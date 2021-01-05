# SPC with instance-level reference

This repo is for developing the spc extension.

## To-Do List

- [x] Enable running spc with Carla server packaged in docker
- [x] Migrate spc from Carla 8 to Carla 9
- [x] Implement better 2d bounding box extraction in spc
- [x] Enable reliable offlane detection 
- [ ] Add 3D-prediction
- [ ] Add scene depth map prediction
- [ ] Add full braking support
- [ ] Improve the shaky driving 
- [ ] Add conditional driving
- [ ] Improve future features prediction quality
- [ ] Refactor code to build unified agent implementation for different CARLA versions

chekc the experiments record on [Google Sheet](https://docs.google.com/spreadsheets/d/1QgFazrKutRdrpIMhWS3BfNrY-ugtqPJJR98PJKgIV00/edit#gid=0)


## Usage

This repo supports three driving simulators: Torcs, CARLA and GTAV. All versions of CALRA8 and CARLA0.9.1 - 0.9.5 are supported respectively.

To run on CARLA9, you should go to the [CARLA website](https://carla.org/) to download the corresponding version of simulator. Then, you should follow the guidelines in their document to install python library. For sub-versions lower than 0.9.5 you could install python lib simply by **pip install carla**. Then, run **script/train_carla.sh** to start the training script. Remember to set the flag **env** in the script to **carla9**.

For more details, you should review the file **main.py**, **train.py** and the agent implementation file **envs/CARLA/carla9.py**.
