This folder contains the scripts used for the manipulation of the RareDis corpus. The main files may be used for management and/or retrieval of information about any NLP corpus annotated in [brat format](https://brat.nlplab.org/).

- `bratman.py`: Python script which contains various methods for extracting information from any corpus containing annotated files using the brat format. It allows sorting mentions on annotations according to their ocurrence in the text, counting of instances on entities and relations and counting of instances on particular entities (nested, overlapped, discontinuous).
- `relations-iaa.py`: Python script which contains the methods necessary to carry out the task of inter annotation agreeement (IAA) between pairs of annotators for any NLP corpus in brat format.


