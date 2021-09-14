# Keypoint Datasets

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



# Dataset Preparation

## COCO

### Info

COCO dataset has 17 keypoints:
* nose
* eye (left, right)
* ear (left, right)
* sholder (left, right)
* elbow (left, right)
* wrist (left, right)
* hip (left, right)
* knee (left, right)
* ankle (left, right)

### Annotation Format

COCO keypoint annotation json file has the following format:

```json
{
    "info": info,                       // not use
    "licenses": [license],              // not use
    "images": [{
        "id": int,                      // image id = image file name w/o 0's
        "width": int,
        "height": int,
        "file_name": str,               // image file name
        "license": int,                 // not use
        "flickr_url": str,              // not use
        "coco_url": str,                // not use
        "date_captured": datetime,      // not use
    }, ...],
    "categories": [{
        "id": 1,                        // category id
        "name": str,                    // category name
        "supercategory": str,           // super category name
        "keypoints": [str],             // keypoint names
        "skeleton": [[int, int],...],   // keypoint connection
    }],
    "annotations": [{
        "area": float,                  // bbox area
        "bbox": [int, int, int, int],   // bbox coordinates
        "category_id": int,             // category id (same as id in categories)
        "image_id": int,                // image id (same as id in images)
        "iscrowd": 0,                   // 0 or 1
        "keypoints": [x1, y1, v1, ...], 
        "num_keypoints": int,
        "segmentation": [polygon],      // not use
    }],
}
```

### Directory Structure

Download the data from [COCO](https://cocodataset.org/#home) and extract them into the following structure.

```
COCO
|__annotations
    |__ person_keypoints_train2017.json
    |__ person_keypoints_val2017.json
|__ train2017
    |__ xxxxxxxxxx.jpg
    |__ ...
|__ val2017
    |__ xxxxxxxxxx.jpg
```

## CrowdPose

### Annotation Format

### Directory Structure

Download the data from [CrowdPose](https://github.com/Jeff-sjtu/CrowdPose). And extract them into the following structure.

```
crowdpose
|__ annotations
    |__ crowdpose_test.json
    |__ crowdpose_train.json
    |__ crowdpose_trainval.json
    |__ crowdpose_val.json
|__ images
    |__ xxxxxxx.jpg
    |__ ...
```