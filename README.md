# Multi-person Pose Estimation

## Introduction

Pose estimation find the keypoints belong to the people in the image. There are two methods exist for pose estimation.

* **Bottom-Up** first finds the keypoints and associates them into different people in the image. (Generally faster and lower accuracy)
* **Top-Down** first detect people in the image and estimate the keypoints. (Generally computationally intensive but better accuracy)

This repo will only incude top-down pose estimation models.

## Features

Datasets
* Body Keypoint
    * 2D
        * [COCO](https://cocodataset.org/#home)
        * [MPII](http://human-pose.mpi-inf.mpg.de/)
        * [MHP](https://lv-mhp.github.io/)
        * [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose) (Crowsed people)
        * [OCHuman](https://github.com/liruilong940607/OCHumanApi) (Occluded people)
    * 3D
        * [Human3.6M](http://vision.imar.ro/human3.6m/description.php)
    * Others
        * [MPII-TRB](https://github.com/kennymckormick/Triplet-Representation-of-human-Body)
        * [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion/LandmarkDetection.html)

* Whole Body Keypoint (Body+Face+Hand+Feet)
    * [COCO-WholeBody](https://github.com/jin-s13/COCO-WholeBody/)
    * [Halpe](https://github.com/Fang-Haoshu/Halpe-FullBody)

* Face Keypoint
    * [WFLW](https://wywu.github.io/projects/LAB/WFLW.html)
    * [COFW](http://www.vision.caltech.edu/xpburgos/ICCV13/)

* Hand Keypoint
    * [FreiHAND](https://lmb.informatik.uni-freiburg.de/projects/freihand/)
    * [CMU Panoptic HandDB](http://domedb.perception.cs.cmu.edu/handdb.html)
    * [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/) (2D/3D)

* Foot Keypoint
    * [Human Foot Keypoint](https://cmu-perceptual-computing-lab.github.io/foot_keypoint_dataset/)

* Animal Keypoint
    * [Animal-Pose](https://sites.google.com/view/animal-pose/) (Dogs+Cats+Sheeps+Horses+Cows)
    * [Horse-10](http://www.mackenziemathislab.org/horse10) (Only Horses)
    * [MacaquePose](http://www.pri.kyoto-u.ac.jp/datasets/macaquepose/index.html) (Only Monkeys)
    * [ATRW](https://cvwc2019.github.io/challenge.html) (Only Tigers)
    * [DeepPoseKit-Zebra](https://github.com/jgraving/DeepPoseKit-Data/tree/master/datasets/zebra) (Only Zebras)
    * [DeepPoseKit-Locust](https://github.com/jgraving/DeepPoseKit-Data/tree/master/datasets/locust) (Only Locusts)
    * [DeepPoseKit-Fly](https://github.com/jgraving/DeepPoseKit-Data/tree/master/datasets/fly) (Only Flies)

* Vehicle Keypoint
    [Apollo Car Instance](http://apolloscape.auto/car_instance.html)

Models

* [RLEPose](https://arxiv.org/abs/2107.11291) [[Code](https://github.com/Jeff-sjtu/res-loglikelihood-regression)]
* [PSA](https://arxiv.org/abs/2107.00782) [[Code](https://github.com/DeLightCMU/PSA)]
* [SimDR](http://arxiv.org/abs/2107.03332) [[Code](https://github.com/leeyegy/SimDR)]
* [HRNet](https://arxiv.org/abs/1908.07919) [[Code](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)]


## Models Comparison

COCO-test-dev 

Model | Backbone | Image Size | AP | AP50 | AP75 | APM | APL | Params (M) | GFLOPs (B)
--- | --- | --- | --- | --- | --- | --- | --- | --- | ---
RLEPose | HRNet-W48 | - | 75.7 | 92.3 | 82.9 | 72.3 | 81.3 | - | -
SimDR* | HRNet-W48 | 256x192 | 75.4 | 92.4 | 82.7 | 71.9 | 81.3 | 66.3 | 14.6
PSA | HRNet-W48 | 256x192 | 78.9 | 93.6 | 85.8 | 76.1 | 83.6 | 70.1 | 15.7
SimDR* | HRNet-W48 | 384x288 | 76.0 | 92.4 | 83.5 | 72.5 | 81.9 | 70.6 | 32.9
PSA | HRNet-W48 | 384x288 | 79.5 | 93.6 | 85.9 | 76.3 | 84.3 | 70.1 | 35.4