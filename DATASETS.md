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