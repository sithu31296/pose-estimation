# Top-Down Multi-person Pose Estimation

## Introduction

Pose estimation find the keypoints belong to the people in the image. There are two methods exist for pose estimation.

* **Bottom-Up** first finds the keypoints and associates them into different people in the image. (Generally faster and lower accuracy)
* **Top-Down** first detect people in the image and estimate the keypoints. (Generally computationally intensive but better accuracy)

This repo will only include top-down pose estimation models.

## Model Zoo

[hrnet]: https://arxiv.org/abs/1908.07919
[simdr]: http://arxiv.org/abs/2107.03332
[psa]: https://arxiv.org/abs/2107.00782
[rlepose]: https://arxiv.org/abs/2107.11291

[phrnetw32]: https://drive.google.com/file/d/1os6T42ri4zsVPXwceli3J3KtksIaaGgu/view?usp=sharing
[phrnetw48]: https://drive.google.com/file/d/1MbEjiXkV83Pm3G2o_Rni4j9CT_jRDSAQ/view?usp=sharing
[hrnetw32]: https://drive.google.com/file/d/1YlPrQMZdNTMWIX3QJ5iKixN3qd0NCKFO/view?usp=sharing
[hrnetw48]: https://drive.google.com/file/d/1hug4ptbf9Y125h9ZH72x4asY2lHt7NA6/view?usp=sharing

COCO-val with detector AP of 56.4

Model | Backbone | Image Size | AP | AP<sup>50 | AP<sup>75 | Params <br><sup>(M) | GFLOPs | Weights
--- | --- | --- | --- | --- | --- | --- | --- | --- 
[PoseHRNet][hrnet] | HRNet-W32 | 256x192 | 74.4 | 90.5 | 81.9 | 29 | 7 | [pretrained][phrnetw32]\|[backbone][hrnetw32]
| | HRNet-W48 | 256x192 | 75.1 | 90.6 | 82.2 | 64 | 15 | [pretrained][phrnetw48]\|[backbone][hrnetw48]
[SimDR][simdr] | HRNet-W32 | 256x192 | 75.3 | - | - | 31 | 7 | -
| | HRNet-W48 | 256x192 | 75.9 | - | - | 66 | 15 | -


COCO-test-dev with detector AP of 60.9

Model | Backbone | Image Size | AP | AP<sup>50 | AP<sup>75 | Params <br><sup>(M) | GFLOPs | Weights
--- | --- | --- | --- | --- | --- | --- | --- | --- 
[RLEPose][rlepose] | HRNet-W48 | 384x288 | 75.7 | 92.3 | 82.9 | - | - | -
[SimDR*][simdr] | HRNet-W48 | 256x192 | 75.4 | 92.4 | 82.7 | 66 | 15 | -
[PSA][psa] | PSA+HRNet-W48 | 256x192 | 78.9 | 93.6 | 85.8 | 70 | 16 | -

## Requirements

* torch >= 1.8.1
* torchvision >= 0.9.1

Other requirements can be installed with `pip install -r requirements.txt`.

Clone the repository recursively:

```bash
$ git clone --recursive https://github.com/sithu31296/pose-estimation.git
```


## Inference

* Download a YOLOv5m trained on [CrowdHuman](https://www.crowdhuman.org/) dataset from [here](https://drive.google.com/file/d/1gglIwqxaH2iTvy6lZlXuAcMpd_U0GCUb/view?usp=sharing). (The weights are from [deepakcrk/yolov5-crowdhuman](https://github.com/deepakcrk/yolov5-crowdhuman).)
* Download a pose estimation model from the tables.
* Run the following command.

```bash
$ python infer.py --source TEST_SOURCE --det-model DET_MODEL_PATH --pose-model POSE_MODEL_PATH --img-size 640
```

Arguments:

* `source`: Testing sources
    * To test an image, set to image file path. (For example, `assests/test.jpg`)
    * To test a folder containing images, set to folder name. (For example, `assests/`)
    * To test a video, set to video file path. (For example, `assests/video.mp4`)
    * To test with a webcam, set to `0`.

Example inference result (image credit: [Learning to surf](https://www.flickr.com/photos/fotologic/6038911779/in/photostream/)):

![test_out](assests/test_out.jpg)


## References

* https://github.com/leoxiaobin/deep-high-resolution-net.pytorch

## Citations

```
@article{WangSCJDZLMTWLX19,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Jingdong Wang and Ke Sun and Tianheng Cheng and 
          Borui Jiang and Chaorui Deng and Yang Zhao and Dong Liu and Yadong Mu and 
          Mingkui Tan and Xinggang Wang and Wenyu Liu and Bin Xiao},
  journal   = {TPAMI}
  year={2019}
}
```