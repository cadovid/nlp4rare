class Error(Exception):
    """Base class for other exceptions"""
    pass


class NotValidPath(Error):
    """Raised when the path introduced is not valid"""
    pass


def update_index_entity(value):
    value = str(value + 1)
    value = 'T'+value
    return value


def update_index_relation(value):
    value = str(value + 1)
    value = 'R'+value
    return value


def clean_id(value):
    value = value.replace('T', '')
    return value


def update_text(df):
    import re
    import pandas as pd
    return pd.Series([
        re.sub(r'Arg1:T[1-9]+\sArg2:T[1-9]+', 'Arg1:T' + str(first_entity) + ' Arg2:T' + str(second_entity), text)
        for (text, first_entity, second_entity) in zip(df['type'], df[2], df[4])
    ])


def order_brat(path):
    """
        Params:
            path: main directory path which contains brat files
        Output:
           Brat files ordered replaced in directory
    """

    try:
        import time
        import pandas as pd
        import numpy as np
        import glob
        import gc  # Garbage Collector to release unreferenced memory
        import os
        start_time = time.time()
        iteration = 0
        extension = '*.ann'
        files = glob.glob(os.path.join(path, extension))
        for file in files:
            file_path = file
            header_list = ["id", "type", "text"]
            ann_df = pd.read_csv(file_path, header=None, sep='\t', lineterminator='\n', names=header_list)
            mask_entities = ann_df["id"].str.startswith("T")
            mask_relations = ann_df["id"].str.startswith("R")
            entities_df = ann_df[mask_entities].copy()
            relations_df = ann_df[mask_relations].copy()
            del ann_df, mask_entities, mask_relations
            gc.collect()

            # Entities dataframe
            split_entities_df = entities_df['type'].str.replace(';', ' ').str.split(' ', expand=True)
            split_entities_df.drop(columns=[0], inplace=True)
            split_entities_df.fillna(value=np.nan, inplace=True)
            split_entities_df = split_entities_df.astype('float32')
            max_len = int(len(split_entities_df.columns))
            char_c_number = []
            full_c_number = [i for i in range(1, max_len + 1)]
            for i in range(0, int(max_len / 2)):
                n = (i + 1) + (1 * i)
                char_c_number.append(n)
            result_entities_df = pd.concat([entities_df, split_entities_df], axis=1)
            del entities_df, split_entities_df
            gc.collect()
            order_list = char_c_number.copy()
            order_list.append(0)
            result_entities_df[0] = pd.Series(result_entities_df['type'].str.split(' ').str.get(0))
            result_entities_df.sort_values(by=order_list, na_position='first', inplace=True)
            result_entities_df.drop(columns=[0], inplace=True)
            result_entities_df.rename(columns={"id": "old_id"}, inplace=True)
            result_entities_df["old_id"] = result_entities_df["old_id"].apply(clean_id).astype('int32')
            result_entities_df.reset_index(drop=True, inplace=True)
            result_entities_df.reset_index(inplace=True)
            result_entities_df.rename(columns={"index": "new_id"}, inplace=True)
            result_entities_df['new_id'] = result_entities_df['new_id'] + 1
            result_entities_df.reset_index(inplace=True)
            result_entities_df['index'] = result_entities_df['index'].apply(update_index_entity)
            ids_dictionary = dict(zip(result_entities_df.old_id, result_entities_df.new_id))
            result_entities_df.drop(columns=['new_id', 'old_id'], inplace=True)
            result_entities_df.drop(columns=full_c_number, inplace=True)
            del char_c_number, full_c_number
            gc.collect()

            # Relations dataframe
            if len(relations_df.index) != 0:
                split_relations_df = relations_df['type'].str.replace('T', '').str.replace(':', ' ') \
                    .str.split(' ', expand=True)
                split_relations_df.drop(columns=[0], inplace=True)
                split_relations_df.fillna(value=np.nan, inplace=True)
                max_len = int(len(split_relations_df.columns))
                non_char_c_number = []
                for i in range(0, int(max_len / 2)):
                    n = (i + 1) + (1 * i)
                    non_char_c_number.append(n)
                char_c_number = [value + 1 for value in non_char_c_number]
                split_relations_df.drop(columns=non_char_c_number, inplace=True)
                split_relations_df = split_relations_df.astype('int32')
                for number in char_c_number:
                    split_relations_df[number] = split_relations_df[number].map(ids_dictionary)
                result_relations_df = pd.concat([relations_df, split_relations_df], axis=1)
                del relations_df, split_relations_df
                gc.collect()
                result_relations_df.reset_index(drop=True, inplace=True)
                result_relations_df['type'] = update_text(result_relations_df)
                result_relations_df.sort_values(by=char_c_number, na_position='first', inplace=True)
                result_relations_df.reset_index(drop=True, inplace=True)
                result_relations_df.reset_index(inplace=True)
                result_relations_df['index'] = result_relations_df['index'].apply(update_index_relation)
                result_relations_df.drop(columns=['id'], inplace=True)
                result_relations_df.drop(columns=char_c_number, inplace=True)
                del non_char_c_number, char_c_number
                gc.collect()

                # Merge both dataframes
                result_df = pd.concat([result_entities_df, result_relations_df], axis=0)
                del result_entities_df, result_relations_df
                gc.collect()
            else:
                result_df = result_entities_df.copy()
                del result_entities_df, relations_df
                gc.collect()
            result_df.to_csv(file_path, index=False, header=False, encoding='utf-8', sep='\t', line_terminator='\n')
            del result_df
            gc.collect()
            iteration += 1
        print(
            '\n****\tORDER OPERATION\t****\nTotal files processed: {}. Elapsed time: {:.2f} seconds.'.format(iteration,
                                                                                                             time.time() - start_time))
        print('Output files saved in:\n\tDirectory: "{}"'.format(path))
    except Exception as e:
        print("Oops!", e.__class__, "occurred. Execution not performed.")


def count_entities(path, count_df=None):
    """
        Params:
            path: main directory path which contains brat files
        Output:
            File with the count of entities inside directory
    """

    try:
        import time
        import pandas as pd
        import glob
        import gc  # Garbage Collector to release unreferenced memory
        import os
        start_time = time.time()
        iteration = 0
        extension = '*.ann'
        files = glob.glob(os.path.join(path, extension))
        for file in files:
            file_path = file
            header_list = ["id", "type", "text"]
            ann_df = pd.read_csv(file_path, header=None, sep='\t', lineterminator='\n', names=header_list)
            split_df = ann_df['type'].str.split(' ', expand=True).copy()
            del header_list
            del ann_df
            gc.collect()
            max_len = int(len(split_df.columns - 1))
            char_c_number = [i for i in range(1, max_len)]
            split_df.drop(columns=char_c_number, inplace=True)
            split_df.rename(columns={0: "Type"}, inplace=True)
            split_grouped_df = split_df.groupby((['Type'])).size().copy()
            result_df = pd.DataFrame(split_grouped_df, columns=['Number'])
            del char_c_number
            del split_df
            del split_grouped_df
            gc.collect()
            if iteration == 0:
                count_df = result_df.copy()
            else:
                count_df = pd.concat([count_df, result_df])
            del result_df
            gc.collect()
            iteration += 1
        total_df = count_df.groupby((['Type']))['Number'].sum().copy()
        del count_df
        gc.collect()
        total_entities = total_df.sum()
        total_df.loc['TOTAL'] = total_entities
        print('\n****\tCOUNT OPERATION\t****\nTotal files processed: {}. Elapsed time: {:.2f} seconds.'.format(iteration, time.time() - start_time))
        output_name = r'!Count_entities_and_relations_result.txt'
        total_df.to_csv(path + '\\' + output_name, index=True, header=True, encoding='utf-8', sep='\t',
                        line_terminator='\n')
        del total_df
        gc.collect()
        print('Output file saved in:\n\tDirectory: "{}"\n\tFile name: "{}"'.format(path, output_name))
    except Exception as e:
        print("Oops!", e.__class__, "occurred. Execution not performed.")


def main():
    """
        Params:
            dir_path: main directory path which contains the brat files to be ordered
        Optional Keyword Arguments:
            bratman.py
                Default. No arguments provided, replace "dir_path" variable
            bratman.py arg1
                arg1 = dir_path
            bratman.py arg1 arg2
                arg1 = '-c': count entities in brat files
                       '-o': order brat files
                       '-co or -oc': order and count brat files (in this order)
                arg2 = dir_path
    """
    import sys
    import os

    if len(sys.argv) == 1:
        try:
            dir_path = r"C:\..." # Replace this variable by desired directory path
            if not os.path.isdir(dir_path):
                raise NotValidPath
            print('Directory path used: {}'.format(dir_path))
            print('Order BRAT files? [Y/N]')
            input_order = input()
            order_brat(dir_path) if input_order == 'Y' or input_order == 'y' else print('Execution not performed.')
            print('Count entities inside BRAT files? [Y/N]')
            input_order = input()
            count_entities(dir_path) if input_order == 'Y' or input_order == 'y' else print('Execution not performed.')
        except NotValidPath:
            print("Directory path introduced is not valid!! Please, introduce a valid path.")

    if len(sys.argv) == 2:
        try:
            dir_path = sys.argv[1]
            if not os.path.isdir(dir_path):
                raise NotValidPath
            print('Directory path used: {}'.format(dir_path))
            print('Order BRAT files? [Y/N]')
            input_order = input()
            order_brat(dir_path) if input_order == 'Y' or input_order == 'y' else print('Execution not performed.')
            print('Count entities inside BRAT files? [Y/N]')
            input_order = input()
            count_entities(dir_path) if input_order == 'Y' or input_order == 'y' else print('Execution not performed.')
        except NotValidPath:
            print("Directory path introduced is not valid!! Please, introduce a valid path.")

    if len(sys.argv) > 2:
        try:
            dir_path = sys.argv[2]
            if not os.path.isdir(dir_path):
                raise NotValidPath
            print('Directory path used: {}'.format(dir_path))
            if sys.argv[1] == '-o':
                order_brat(dir_path)
            elif sys.argv[1] == '-c':
                count_entities(dir_path)
            elif sys.argv[1] == '-co' or sys.argv[1] == '-oc':
                order_brat(dir_path)
                count_entities(dir_path)
            else:
                print('Not valid arguments provided. Check instructions.')
        except NotValidPath:
            print("Directory path introduced is not valid!! Please, introduce a valid path.")


if __name__ == "__main__":
    main()
