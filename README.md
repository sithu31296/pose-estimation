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

[hrnetw32]: https://drive.google.com/file/d/1YlPrQMZdNTMWIX3QJ5iKixN3qd0NCKFO/view?usp=sharing
[hrnetw48]: https://drive.google.com/file/d/1hug4ptbf9Y125h9ZH72x4asY2lHt7NA6/view?usp=sharing

[phrnetw32]: https://drive.google.com/file/d/1os6T42ri4zsVPXwceli3J3KtksIaaGgu/view?usp=sharing
[phrnetw48]: https://drive.google.com/file/d/1MbEjiXkV83Pm3G2o_Rni4j9CT_jRDSAQ/view?usp=sharing
[simdrw32]: https://drive.google.com/file/d/1Bd8h2H30tCN8WuLIhuSRF9ViN6zghj29/view?usp=sharing
[simdrw48]: https://drive.google.com/file/d/1WU_9e0MxgrO8X4W6wKo16L8siCdwgLSZ/view?usp=sharing
[sasimdrw48]: https://drive.google.com/file/d/1Tj9bGL7g7XRyL2F1a-uAcWhgYXnXpqBY/view?usp=sharing

<details open>
  <summary><strong>COCO-val with 56.4 Detector AP</strong></summary>

Model | Backbone | Image Size | AP | AP<sup>50 | AP<sup>75 | Params <br><sup>(M) | FLOPs <br><sup>(B) | FPS | Weights
--- | --- | --- | --- | --- | --- | --- | --- | --- | ---
[PoseHRNet][hrnet] | HRNet-w32 | 256x192 | 74.4 | 90.5 | 81.9 | 29 | 7 | 25 | [download][phrnetw32]
| | HRNet-w48 | 256x192 | 75.1 | 90.6 | 82.2 | 64 | 15 | 24 | [download][phrnetw48]
[SimDR][simdr] | HRNet-w32 | 256x192 | 75.3 | - | - | 31 | 7 | 25 | [download][simdrw32]
| | HRNet-w48 | 256x192 | 75.9 | 90.4 | 82.7 | 66 | 15 | 24 | [download][simdrw48]

</details>

> Note: FPS is tested on a GTX1660ti with one person per frame including pre-processing, model inference and post-processing. Both detection and pose models are in PyTorch FP32.

<details>
  <summary><strong>COCO-test with 60.9 Detector AP</strong> (click to expand)</summary>

Model | Backbone | Image Size | AP | AP<sup>50 | AP<sup>75 | Params <br><sup>(M) | FLOPs <br><sup>(B) | Weights
--- | --- | --- | --- | --- | --- | --- | --- | --- 
[SimDR*][simdr] | HRNet-w48 | 256x192 | 75.4 | 92.4 | 82.7 | 66 | 15 | [download][sasimdrw48]
[RLEPose][rlepose] | HRNet-w48 | 384x288 | 75.7 | 92.3 | 82.9 | - | - | -
[UDP+PSA][psa] | HRNet-w48 | 256x192 | 78.9 | 93.6 | 85.8 | 70 | 16 | -

</details>

<br>
<details>
  <summary><strong>Download Backbone Models' Weights</strong> (click to expand)</summary>

Model | Weights
--- | ---
HRNet-w32 | [download][hrnetw32]
HRNet-w48 | [download][hrnetw48]

</details>

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
* Download a pose estimation model's weights from the tables.
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
* `det-model`: YOLOv5 model's weights path
* `pose-model`: Pose estimation model's weights path

Example inference results (image credit: [[1](https://www.flickr.com/photos/fotologic/6038911779/in/photostream/), [2](https://neuralet.com/article/pose-estimation-on-nvidia-jetson-platforms-using-openpifpaf/)]):

![infer_result](assests/infer_results.jpg)


## References

* https://github.com/leoxiaobin/deep-high-resolution-net.pytorch
* https://github.com/ultralytics/yolov5

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

@misc{li20212d,
  title={Is 2D Heatmap Representation Even Necessary for Human Pose Estimation?}, 
  author={Yanjie Li and Sen Yang and Shoukui Zhang and Zhicheng Wang and Wankou Yang and Shu-Tao Xia and Erjin Zhou},
  year={2021},
  eprint={2107.03332},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

```