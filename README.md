# arm_pose_research_script

## Dataset download
download the datasets from this [Dropbox link](https://www.dropbox.com/sh/etvx8edkweco97u/AABQkjEIGJOwXy09AdAQDqWRa?dl=0)
put the zipped file into folder ./data and unzip it. For example, you can put the test dataset into folder ./data/test_20181024.zip and unzip it. 

## Get started
1. Download the checkpoint for the pretrained model [here](https://www.dropbox.com/s/asi82l7hjdvo1ne/checkpoint.pth.tar?dl=0) and put it into a folder, e.g. ./checkpoint/checkpoint.pth.tar. 
2. Create a folder for result saving, e.g. ./saved_results.
3. Open val_arm_reall.sh. Replace --data-dir, --resume and --save-result-dir with the folder where you put the datasets, the pre-train model and the saved result, respectively. For example,
```bash
--data-dir ./data/test_20181024 --resume ./checkpoint/checkpoint.pth.tar --save-result-dir ./saved_results
```
4. Run test_arm_reall.sh and you can see the accuracy on the real lab dataset.

## Dependencies
1. pytorch with version 0.4.1 or higher
2. OpenCV
