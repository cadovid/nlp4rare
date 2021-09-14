class Error(Exception):
    """Base class for other exceptions"""
    pass


class NotValidPath(Error):
    """Raised when the path introduced is not valid"""
    pass


def cohen_kappa(ann1, ann2):
    """
    Computes Cohen kappa for pair-wise annotators
        Input:
            :param ann1 [type:list]: annotations provided by first annotator
            :param ann2 [type:list]: annotations provided by second annotator
        Output:
           :return: Cohen kappa statistic [type:float]
    """
    import numpy as np
    if (len(ann1) == 0) and (len(ann2) == 0):
        return np.nan
    count = 0
    for an1, an2 in zip(ann1, ann2):
        if an1 == an2:
            count += 1
    A = count / len(ann1)  # observed agreement A (Po)
    uniq = set(ann1 + ann2)
    E = 0  # expected agreement E (Pe)
    for item in uniq:
        cnt1 = ann1.count(item)
        cnt2 = ann2.count(item)
        count = ((cnt1 / len(ann1)) * (cnt2 / len(ann2)))
        E += count
    if (1 - E) == 0:
        return 1
    return round((A - E) / (1 - E), 4)


def compute_f(ann1, ann2):
    """
    Computes F-measure for pair-wise annotators
        Input:
            :param ann1 [type:list]: annotations provided by first annotator
            :param ann2 [type:list]: annotations provided by second annotator
        Output:
           :return: Cohen kappa statistic [type:float]
    """
    expected = set(ann1)
    predicted = set(ann2)

    correct = len(expected.intersection(predicted))  # True positive values
    spurious = len(expected.difference(predicted))  # False positive values
    missing = len(predicted.difference(expected))   # False negative values

    keySize = correct + spurious  # Number of positive examples in key set
    resSize = correct + missing  # Number of positive examples in the results set

    if keySize == 0:
        precision = 0
    else:
        precision = correct / keySize
    if resSize == 0:
        recall = 0
    else:
        recall = correct / resSize
    if precision + recall == 0:
        return 0
    else:
        return 2 * precision * recall / (precision + recall)


def iaa_kappa(ann1_path, ann2_path):
    """
    Computes Cohen kappa Inter-Annotator Agreement for two BRAT annotators
        Input:
            :param ann1_path [type:string]: directory for annotator one's annotations
            :param ann2_path [type:string]: directory for annotator two's annotations
        Output:
           :return: Displays visual information [type:None]
    """
    import pandas as pd
    import glob
    import os
    import gc
    import re

    pd.set_option('display.max_columns', 200)
    extension = '*.ann'
    ann1_files = glob.glob(os.path.join(ann1_path, extension))
    ann2_files = glob.glob(os.path.join(ann2_path, extension))
    assert len(ann1_files) == len(ann2_files), "Number of files is not the same for each annotator"
    kappa_per_document = dict()
    kappa_per_label = dict()

    for ann1_file, ann2_file in zip(ann1_files, ann2_files):
        ann1_pattern = '^(.+)\\\(.+)\.ann$'
        ann1_filename = re.search(ann1_pattern, ann1_file).group(2)
        ann2_pattern = '^(.+)\\\(.+)\.ann$'
        ann2_filename = re.search(ann2_pattern, ann2_file).group(2)
        if ann1_filename != ann2_filename:
            print('Warning: Brat files are not the same\n\tann1 filename: {}\n\tann2 filename: {}'.format(ann1_filename,
                                                                                                      ann2_filename))
        header_list = ["id", "type", "text"]

        ann1_df = pd.read_csv(ann1_file, header=None, sep='\t', lineterminator='\n', names=header_list)
        ann1_mask_relations = ann1_df["id"].str.startswith("R")
        ann1_relations_df = ann1_df[ann1_mask_relations].copy()
        ann1_relations_df['text'] = ann1_relations_df['type'].str.split(' ').str.get(0)
        ann1_uniques = list(ann1_relations_df['text'].unique())

        ann2_df = pd.read_csv(ann2_file, header=None, sep='\t', lineterminator='\n', names=header_list)
        ann2_mask_relations = ann2_df["id"].str.startswith("R")
        ann2_relations_df = ann2_df[ann2_mask_relations].copy()
        ann2_relations_df['text'] = ann2_relations_df['type'].str.split(' ').str.get(0)
        ann2_uniques = list(ann2_relations_df['text'].unique())

        ann_uniques = set(ann1_uniques + ann2_uniques)
        kappa_cat_dict = dict()

        for category in ann_uniques:
            if category not in ann1_uniques or category not in ann2_uniques:
                kappa_cat_dict[category] = 0
            else:
                ann1_mask = (ann1_relations_df['text'] == category)
                ann1_cat_df = ann1_relations_df[ann1_mask].copy()
                ann2_mask = (ann2_relations_df['text'] == category)
                ann2_cat_df = ann2_relations_df[ann2_mask].copy()
                category_kappa = cohen_kappa(list(ann1_cat_df['type'].values), list(ann2_cat_df['type'].values))
                kappa_cat_dict[category] = category_kappa
                del ann1_mask, ann2_mask, ann1_cat_df, ann2_cat_df, category_kappa
                gc.collect()

        kappa_per_label[ann1_filename] = kappa_cat_dict

        overall_kappa = cohen_kappa(list(ann1_relations_df['type'].values), list(ann2_relations_df['type'].values))
        kappa_per_document[ann1_filename] = overall_kappa

        del ann1_df, ann2_df, ann1_mask_relations, ann2_mask_relations, ann1_uniques, ann2_uniques, kappa_cat_dict
        gc.collect()

    df_label = pd.DataFrame.from_dict(kappa_per_label)
    grouped_df = pd.DataFrame(df_label.sum(axis=1), columns=['Total Cohen kappa'])
    grouped_df['Cohen kappa'] = grouped_df['Total Cohen kappa'] / (len(df_label.columns) - (df_label.isna().sum(axis=1)))
    grouped_df.drop(columns=['Total Cohen kappa'], inplace=True)

    df_document = pd.DataFrame.from_dict(kappa_per_document, orient='index', columns=['Cohen kappa'])

    print('\n##Agreement per Label and Document:\n{}'.format(df_label))
    print('\n##Agreement per Label:\n{}'.format(grouped_df))
    print('\n##Agreement per Document:\n{}'.format(df_document))
    print('\n##Overall Agreement:\nMean Cohen kappa\t{}'.format(round((df_document.sum(axis=0)
                                                                       / (len(df_document) - df_document.isna().sum(axis=0)))[0], 4)))
    del kappa_per_document, kappa_per_label, ann1_files, ann2_files, df_label, grouped_df, df_document
    gc.collect()


def iaa_fscore(ann1_path, ann2_path):
    """
    Computes F-score Inter-Annotator Agreement for two BRAT annotators
        Input:
            :param ann1_path [type:string]: directory for annotator one's annotations
            :param ann2_path [type:string]: directory for annotator two's annotations
        Output:
           :return: Displays visual information [type:None]
    """
    import pandas as pd
    import glob
    import os
    import gc
    import re

    pd.set_option('display.max_columns', 200)
    extension = '*.ann'
    ann1_files = glob.glob(os.path.join(ann1_path, extension))
    ann2_files = glob.glob(os.path.join(ann2_path, extension))
    assert len(ann1_files) == len(ann2_files), "Number of files is not the same for each annotator"
    f_score_per_label = dict()
    f_score_per_document = dict()

    for ann1_file, ann2_file in zip(ann1_files, ann2_files):
        ann1_pattern = '^(.+)\\\(.+)\.ann$'
        ann1_filename = re.search(ann1_pattern, ann1_file).group(2)
        ann2_pattern = '^(.+)\\\(.+)\.ann$'
        ann2_filename = re.search(ann2_pattern, ann2_file).group(2)
        if ann1_filename != ann2_filename:
            print('Warning: Brat files are not the same\n\tann1 filename: {}\n\tann2 filename: {}'.format(ann1_filename,
                                                                                                          ann2_filename))
        header_list = ["id", "type", "text"]

        ann1_df = pd.read_csv(ann1_file, header=None, sep='\t', lineterminator='\n', names=header_list)
        ann1_mask_relations = ann1_df["id"].str.startswith("R")
        ann1_relations_df = ann1_df[ann1_mask_relations].copy()
        ann1_relations_df['text'] = ann1_relations_df['type'].str.split(' ').str.get(0)
        ann1_uniques = list(ann1_relations_df['text'].unique())

        ann2_df = pd.read_csv(ann2_file, header=None, sep='\t', lineterminator='\n', names=header_list)
        ann2_mask_relations = ann2_df["id"].str.startswith("R")
        ann2_relations_df = ann2_df[ann2_mask_relations].copy()
        ann2_relations_df['text'] = ann2_relations_df['type'].str.split(' ').str.get(0)
        ann2_uniques = list(ann2_relations_df['text'].unique())

        ann_uniques = set(ann1_uniques + ann2_uniques)
        f_score_cat = dict()

        for category in ann_uniques:
            if category not in ann1_uniques or category not in ann2_uniques:
                f_score_cat[category] = 0
            else:
                ann1_mask = (ann1_relations_df['text'] == category)
                ann1_cat_df = ann1_relations_df[ann1_mask].copy()
                ann2_mask = (ann2_relations_df['text'] == category)
                ann2_cat_df = ann2_relations_df[ann2_mask].copy()
                category_f_score = compute_f(list(ann1_cat_df['type'].values), list(ann2_cat_df['type'].values))
                f_score_cat[category] = category_f_score
                del ann1_mask, ann2_mask, ann1_cat_df, ann2_cat_df, category_f_score
                gc.collect()

        f_score_per_label[ann1_filename] = f_score_cat

        overall_f_score = compute_f(list(ann1_relations_df['type'].values), list(ann2_relations_df['type'].values))
        f_score_per_document[ann1_filename] = overall_f_score

        del ann1_df, ann2_df, ann1_mask_relations, ann2_mask_relations, ann1_uniques, ann2_uniques, f_score_cat
        gc.collect()

    df_label = pd.DataFrame.from_dict(f_score_per_label)
    grouped_df = pd.DataFrame(df_label.sum(axis=1), columns=['Total f-score'])
    grouped_df['f-score'] = grouped_df['Total f-score'] / (len(df_label.columns) - (df_label.isna().sum(axis=1)))
    grouped_df.drop(columns=['Total f-score'], inplace=True)

    df_document = pd.DataFrame.from_dict(f_score_per_document, orient='index', columns=['f-score'])

    print('\n##Agreement per Label and Document:\n{}'.format(df_label))
    print('\n##Agreement per Label:\n{}'.format(grouped_df))
    print('\n##Agreement per Document:\n{}'.format(df_document))
    print('\n##Overall Agreement:\nMean f-score\t{}'.format(round((df_document.sum(axis=0)
                                                                   / (len(df_document) - df_document.isna().sum(
                axis=0)))[0], 4)))
    del f_score_per_document, f_score_per_label, ann1_files, ann2_files, df_label, grouped_df, df_document
    gc.collect()


def merge_files(ann1_path, ann2_path):
    """
    Removes different BRAT files between two annotators for computing a correct pair-wise IAA afterwards
        Input:
            :param ann1_path [type:string]: directory for annotator one's annotations
            :param ann2_path [type:string]: directory for annotator two's annotations
        Output:
           :return: Displays visual information [type:None]
    """
    import glob
    import os
    import re

    extension = '*.ann'  # Desired extension
    ann1_files = glob.glob(os.path.join(ann1_path, extension))
    ann2_files = glob.glob(os.path.join(ann2_path, extension))

    ann1_files_set = set()
    ann2_files_set = set()

    for ann1_file in ann1_files:
        ann1_pattern = '^(.+)\\\(.+)\.ann$'
        ann1_filename = re.search(ann1_pattern, ann1_file).group(2)
        ann1_files_set.add(ann1_filename)

    for ann2_file in ann2_files:
        ann2_pattern = '^(.+)\\\(.+)\.ann$'
        ann2_filename = re.search(ann2_pattern, ann2_file).group(2)
        ann2_files_set.add(ann2_filename)

    common_files = ann1_files_set.intersection(ann2_files_set)
    unique_ann1_files = ann1_files_set.difference(common_files)
    unique_ann2_files = ann2_files_set.difference(common_files)

    print('##INFORMATION\n\tTotal ann1 files: {}\n\tTotal ann2 files: {}\n\tCommon files: {}\n\tUnique ann1 files: {}'
          '\n\tUnique ann2 files: {}'.format(len(ann1_files_set), len(ann2_files_set), len(common_files),
                                             len(unique_ann1_files), len(unique_ann2_files)))

    count_ann1 = 0
    count_ann2 = 0
    if len(unique_ann1_files) != 0:
        for filename in unique_ann1_files:
            os.remove(ann1_path + '\\' + filename + '.ann')
            count_ann1 += 1
    if len(unique_ann2_files) != 0:
        for filename in unique_ann2_files:
            os.remove(ann2_path + '\\' + filename + '.ann')
            count_ann2 += 1
    print('##OPERATION\n\tTotal files deleted for ann1: {}\n\tTotal files deleted for ann2: {}'.format(count_ann1,
                                                                                                       count_ann2))


def main():
    """
        Input:
            :param ann1_path [type:string]: directory for annotator one's annotations
            :param ann2_path [type:string]: directory for annotator two's annotations
        Optional Keyword Arguments:
            :bratman.py
                Default. No arguments provided, replace both "dir_path" variables
            :bratman.py arg1 arg2
                arg1 = annotator one's directory
                arg2 = annotator two's directory
            :bratman.py arg1 arg2 arg3
                arg1 = '-mk': merge brat files and compute cohen kappa iaa
                       '-mf': merge brat files and compute f-score iaa
                       '-k': compute cohen kappa iaa
                       '-f': compute f-score iaa
                       '-m': merge brat files
                arg2 = annotator one's directory
                arg3 = annotator two's directory
    """
    import sys
    import os

    if len(sys.argv) == 1:
        try:
            # Replace the next two variable by desired directory paths
            dir_path_one = r"C:\..."
            dir_path_two = r"C:\..."
            if not os.path.isdir(dir_path_one) or not os.path.isdir(dir_path_two):
                raise NotValidPath
            print('Directories path used:\n\t{}\n\t{}'.format(dir_path_one, dir_path_two))
            print('Merge files? [Y/N]\n[WARNING] This operation removes all the different BRAT files between annotators.'
                  'It is recommended to save a copy before the operation.')
            input_order = input()
            merge_files(dir_path_one, dir_path_two) if input_order == 'Y' or input_order == 'y' else print('Execution not performed.')
            print('Select IAA method to compute:\n\t[k] Cohen kappa\n\t[f] F-score')
            input_order = input()
            if input_order == 'K' or input_order == 'k':
                iaa_kappa(dir_path_one, dir_path_two)
            elif input_order == 'F' or input_order == 'f':
                iaa_fscore(dir_path_one, dir_path_two)
            else:
                print('Introduced command <{}> is not valid. Execution not performed.'.format(input_order))
        except NotValidPath:
            print("Directory path introduced is not valid!! Please, introduce a valid path.")

    elif len(sys.argv) == 3:
        try:
            dir_path_one = sys.argv[1]
            dir_path_two = sys.argv[2]
            if not os.path.isdir(dir_path_one) or not os.path.isdir(dir_path_two):
                raise NotValidPath
            print('Directories path used:\n\t{}\n\t{}'.format(dir_path_one, dir_path_two))
            print(
                'Merge files? [Y/N]\n[WARNING] This operation removes all the different BRAT files between annotators.'
                ' It is recommended to save a copy before the operation.')
            input_order = input()
            merge_files(dir_path_one, dir_path_two) if input_order == 'Y' or input_order == 'y' else print(
                'Execution not performed.')
            print('Select IAA method to compute:\n\t[k] Cohen kappa\n\t[f] F-score')
            input_order = input()
            if input_order == 'K' or input_order == 'k':
                iaa_kappa(dir_path_one, dir_path_two)
            elif input_order == 'F' or input_order == 'f':
                iaa_fscore(dir_path_one, dir_path_two)
            else:
                print('Introduced command <{}> is not valid. Execution not performed.'.format(input_order))
        except NotValidPath:
            print("Directory path introduced is not valid!! Please, introduce a valid path.")

    elif len(sys.argv) > 3:
        try:
            dir_path_one = sys.argv[2]
            dir_path_two = sys.argv[3]
            if not os.path.isdir(dir_path_one) or not os.path.isdir(dir_path_two):
                raise NotValidPath
            print('Directories path used:\n\t{}\n\t{}'.format(dir_path_one, dir_path_two))
            if sys.argv[1] == '-m':
                merge_files(dir_path_one, dir_path_two)
            elif sys.argv[1] == '-k':
                iaa_kappa(dir_path_one, dir_path_two)
            elif sys.argv[1] == '-f':
                iaa_fscore(dir_path_one, dir_path_two)
            elif sys.argv[1] == '-mk' or sys.argv[1] == '-km':
                merge_files(dir_path_one, dir_path_two)
                iaa_kappa(dir_path_one, dir_path_two)
            elif sys.argv[1] == '-mf' or sys.argv[1] == '-fm':
                merge_files(dir_path_one, dir_path_two)
                iaa_fscore(dir_path_one, dir_path_two)
            else:
                print('Not valid arguments provided. Check instructions.')
        except NotValidPath:
            print("Directory path introduced is not valid!! Please, introduce a valid path.")

    else:
        print('Not valid arguments provided. Check instructions.')


if __name__ == "__main__":
    main()
