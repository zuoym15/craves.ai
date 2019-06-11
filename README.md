## CRAVES: Controlling Robotic Arm with a Vision-based, Economic System

The craves.ai project controls a toy robotic arm (OWI-535) with a single RGB camera. Please see the system pipeline and how it works in [docs/README.md](docs/README.md) first before trying the code. The following animation shows the arm controlling by a mounted camera to reach a goal without relying on any other sensors.

![reach-demo](docs/reach2.gif)

## Dataset Download

Download the datasets from [here](http://www.cs.jhu.edu/~qiuwch/craves/dataset/).

Put the zipped file into folder ./data and unzip it. For example, you can put the test dataset into folder `./data/test_20181024.zip` and unzip it.

- 20181107.zip, synthetic training images and ground truth
- ft_20181105.zip, unlabeled real lab images for fine-tuning
- test_20181024.zip, lab test images with 3D ground truth 
- youtube_20181105.zip, youtube test images with 2D ground truth

## Pose Estimation

1. Download the checkpoint for the pretrained model [here](http://www.cs.jhu.edu/~qiuwch/craves/) and put it into a folder, e.g. ./checkpoint/checkpoint.pth.tar. 
2. Create a folder for result saving, e.g. `./saved_results`.
3. Open `val_arm_reall.sh`. Replace `--data-dir`, `--resume` and `--save-result-dir` with the folder where you put the datasets, the pre-train model and the saved result, respectively. For example,
`--data-dir ./data/test_20181024 --resume ./checkpoint/checkpoint.pth.tar --save-result-dir ./saved_results`

4. Run `test_arm_reall.sh` and you can see the accuracy on the real lab dataset.

Dependencies: pytorch with version 0.4.1 or higher, OpenCV

## Data Generation from Simulator

Download the linux binary from [here](https://cs.jhu.edu/~qiuwch/craves/sim/arm-0610.zip) (tested in Ubuntu 16.04).

Unzip and run `./LinuxNoEditor/ArmUE4.sh`.

Run the following script to generate images and ground truth

```bash
pip install unrealcv imageio
python demo_capture.py frame.png
```

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
