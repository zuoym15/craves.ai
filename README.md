## CRAVES: Controlling Robotic Arm with a Vision-based, Economic System

This is the code for pose estimation module of CRAVES. If you want to test on the OWI-535 hardware, please refer the control module [here](https://github.com/zfw1226/craves_control).

The craves.ai project controls a toy robotic arm (OWI-535) with a single RGB camera. Please see the system pipeline and how it works in [docs/README.md](docs/README.md) first before trying the code. The following animation shows the arm controlling by a mounted camera to reach a goal without relying on any other sensors.

![reach-demo](docs/reach2.gif)

Here are some visualization result from the YouTube dataset:

![youtube_heatmap](docs/youtube_heatmap.png)
 
`./data_generation/load_data_and_vis.py` contains samples on how to visualize the images and their annotations.



## Dataset Download

We created three datasets for this project, namely `synthetic`, `lab` and `youtube`. 

Download the datasets from [here](http://www.cs.jhu.edu/~qiuwch/craves/dataset/).

For the usage of these datasets, please refer to [here](docs/dataset_info.md).

## Pose Estimation

1. Download the checkpoint for the pretrained model [here](http://www.cs.jhu.edu/~qiuwch/craves/) and put it into a folder, e.g. ./checkpoint/checkpoint.pth.tar. 
2. Create a folder for result saving, e.g. `./saved_results`.
3. Open `./scripts/val_arm_reall.sh`. Make sure `--data-dir`, `--resume` and `--save-result-dir` match with the folder where you put the datasets, the pre-train model and the saved result in, respectively. For example,
`--data-dir ../data/test_20181024 --resume ../checkpoint/checkpoint.pth.tar --save-result-dir ../saved_results`

4. `cd ./scripts` then run `sh val_arm_reall.sh` and you can see the accuracy on the real lab dataset.

The output you should expect to see:

```
sh val_arm_reall.sh
=> creating model 'hg', stacks=2, blocks=1
=> loading checkpoint '../checkpoint/checkpoint.pth.tar'
=> loaded checkpoint '../checkpoint/checkpoint.pth.tar' (epoch 30)
    Total params: 6.73M
No. images of dataset 1 : 428
merging 1 datasets, total No. images: 428
No. minibatches in validation set:72

Evaluation only
Processing |################################| (72/72) Data: 0.000000s | Batch: 0.958s | Total: 0:01:08 | ETA: 0:00:01 | Loss: 0.0009 | Acc:  0.9946
```
As you can see, the overall accuracy on the lab dataset is 99.46% under the PCK@0.2 metric.

Other shell scripts you may want to try:

- `train_arm.sh` and `train_arm_concat.sh`: train a model from scratch with synthetic dataset only and with multiple datasets, respectively.
- `val_arm_syn.sh`: evaluate model on synthetic dataset
- `val_arm_reall_with_3D`: evaluate model on synthetic dataset, giving both 2D and 3D output.
- `val_arm_youtube.sh` and `val_arm_youtube_vis_only.sh`: evaluate model on youtube dataset, with all keypoints and only visible keypoints, respectively.

Dependencies: pytorch with version 0.4.1 or higher, OpenCV

The 2D pose estimation module is developed based on [pytorch-pose](https://github.com/bearpaw/pytorch-pose).

## Data Generation from Simulator

Download the binary for [Windows](https://cs.jhu.edu/~qiuwch/craves/sim/arm-win-0808.zip) or [Linux](https://cs.jhu.edu/~qiuwch/craves/sim/arm-linux-0808.zip) (tested in Ubuntu 16.04).

Unzip and run `./LinuxNoEditor/ArmUE4.sh`.

Run the following script to generate images and ground truth

```bash
pip install unrealcv imageio
cd ./data_generation
python demo_capture.py
```
Generated data are saved in `./data/new_data` by default. You can visualize the groundtruth with the script `./data_generation/load_data_and_vis.py`.

## Control System

The control module of CRAVES is hosted in another repo, https://github.com/zfw1226/craves_control.

Please see this repo for hardware drivers, pose estimator, a PID-like controller, and a RL-based controller.

## Citation
If you found CRAVES useful, please consider citing:
```bibtex
@article{zuo2019craves,
  title={CRAVES: Controlling Robotic Arm with a Vision-based, Economic System},
  author={Zuo, Yiming and Qiu, Weichao and Xie, Lingxi and Zhong, Fangwei and Wang, Yizhou and Yuille, Alan L},
  journal={CVPR},
  year={2019}
}
```


## Contact

If you have any question or suggestions, please open an issue in this repo. Thanks.

Disclaimer: authors are a group of scientists working on computer vision research. They are not associated with the company manufactures this arm. If you have a better hardware to recommend, or want to apply this technique to your arm, please contact us.
