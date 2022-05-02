# TrajNet++ Setup

Download the train.zip dataset from [here](https://github.com/vita-epfl/trajnetplusplusdata/releases/tag/v4.0).
Extract the data 
Then run:

```
python create_data_npys.py --raw-dataset-path /path/to/synth_data/ --output-npy-path /path/to/output_npys
```

This script will split the training data into train and validation numpy files called `{split}_orca_synth.npy`.
