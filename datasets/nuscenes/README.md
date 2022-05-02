# nuScenes Installation and Setup

First, you need to download the dataset and the map expansion (v1.3) from [here](https://www.nuscenes.org/nuscenes#download).
To get setup, follow the instructions [here](https://github.com/nutonomy/nuscenes-devkit) to install the nuscenes devkit.

Ensure that the final folder structure looks like this:
```
v1.0-trainval_full
 └──v1.0-trainval
      └──attribute.json
      └──calibrated_sensor.json
      └──category.json
      └──ego_pose.json
      └──instance.json
      └──log.json
      └──map.json
      └──sample.json
      └──sample_annotation.json
      └──sample_data.json
      └──scene.json
      └──sensor.json
      └──visibility.json
 └──maps
      └──basemap
            └──boston-seaport.png
            └──singapore-hollandvillage.png
            └──singapore-onenorth.png
            └──singapore-queenstown.png
      └──expansion
            └──boston-seaport.json
            └──singapore-hollandvillage.json
            └──singapore-onenorth.json
            └──singapore-queenstown.json
      └──prediction
          └──prediction_scenes.json
      └──36092f0b03a857c6a3403e25b4b7aab3.png
      └──37819e65e09e5547b8a3ceaefba56bb2.png
      └──53992ee3023e5494b90c316c183be829.png
      └──93406b464a165eaba6d9de76ca09f5da.png
      
```

Once this is done, run the following to create the h5 files of the dataset:
```
python create_h5_nusc.py --raw-dataset-path /path/to/v1.0-trainval_full --split-name [train/val] --output-h5-path /path/to/output/nuscenes_h5_file/
```
for both train and val.
