# Argoverse Python package - Installation

Follow the instructions [here](https://github.com/argoai/argoverse-api) to 
download the dataset files and install the Argoverse API.

After downloading the dataset files, 
extract the contents of each dataset split file such that the final folder 
structure of the dataset looks like this:
```
argoverse
 └──train
      └──Argoverse-Terms_of_Use.txt
      └──data
          └──1.csv
          └──2.csv
          └──...
 └──val
      └──Argoverse-Terms_of_Use.txt
      └──data
          └──1.csv
          └──2.csv
          └──...
 └──test
      └──Argoverse-Terms_of_Use.txt
      └──data
          └──1.csv
          └──2.csv
          └──...
```

Afterwards, run the following to create the h5 files of the dataset:

```
python create_h5_argo.py --raw-dataset-path /path/to/argoverse --split-name [train/val/test] --output-h5-path /path/to/output/h5_files

```
for both train and val.

Time to create and disk space taken:

| Split       | Time        | Final H5 size |
| ----------- | ----------- | ------------- |
| train       | 4 hours     | 4 GB          |
| val         | 1 hour      | 770 MB        |
| test        | 2 hours     | 1 GB          |
