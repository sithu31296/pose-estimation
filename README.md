# Pose-Estimation-Pipeline

## Introduction

Pose estimation find the keypoints belong to the people in the image. There are two methods exist for pose estimation.

* **Bottom-Up** first finds the keypoints and associates them into different people in the image. (Generally faster and lower accuracy)
* **Top-Down** first detect people in the image and estimate the keypoints. (Generally computationally intensive but better accuracy)

This repo will only include bottom-up pose estimation methods.


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

Bottom-Up Models
* [OpenPifPaf](https://arxiv.org/abs/2103.02440)
* [Lightweight OpenPose](https://arxiv.org/abs/1811.12004)
* [HigherHRNet](https://arxiv.org/abs/1908.10357)

## Dataset Comparison

Dataset | Images | Instances | Keypoints
--- | --- | --- | ---
COCO | - | - | - 

## Models Comparison

