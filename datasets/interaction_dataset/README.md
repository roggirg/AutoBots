# Interaction-Dataset Setup

Before getting started, you need to get access to the dataset from [here](https://interaction-dataset.com/).
Once you receive the email from their team, download and extract the contents of `*multi*.zip` into `multi_agent`. 

Then, run the following to create the h5 files of the dataset:
```
python create_h5_indst.py --output-h5-path /path/to/output/interaction_dataset_h5_file/ --raw-dataset-path /path/to/multi_agent/ --split-name [train/val]
```
for both train and val.
Finally, ensure that the output folder containing the H5 files (for train and val) also has a copy of the `maps` folder.
To make things easy, I would recommend simply setting `--output-h5-path /path/to/multi_agent`.

