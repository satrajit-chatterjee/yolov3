# yolov3
yolov3 implementation in PyTorch from https://github.com/ultralytics/yolov3

# Description

The code works on Linux, MacOS and Windows. Training is done on the COCO dataset by default: https://cocodataset.org/#home. **Credit to Joseph Redmon for YOLO** (https://pjreddie.com/darknet/yolo/) and to **Erik Lindernoren for the PyTorch implementation** this work is based on (https://github.com/eriklindernoren/PyTorch-YOLOv3).

# Requirements

Python 3.7 or later with the following `pip3 install -U -r requirements.txt` packages:

- `numpy`
- `torch >= 1.0.0`
- `opencv-python`

# Training

**Start Training:** Run `train.py` to begin training after downloading COCO data with `data/get_coco_dataset.sh`. Training runs about 1 hour per COCO epoch on a 1080 Ti.

**Resume Training:** Run `train.py --resume` to resume training from the most recently saved checkpoint `weights/latest.pt`.

Each epoch trains on 120,000 images from the train and validate COCO sets, and tests on 5000 images from the COCO validate set. Default training settings produce loss plots below, with **training speed of 0.6 s/batch on a 1080 Ti (18 epochs/day)** or 0.45 s/batch on a 2080 Ti.

`from utils import utils; utils.plot_results()`
![Alt](https://user-images.githubusercontent.com/26833433/53494085-3251aa00-3a9d-11e9-8af7-8c08cf40d70b.png "train.py results")

## Image Augmentation

`datasets.py` applies random OpenCV-powered (https://opencv.org/) augmentation to the input images in accordance with the following specifications. Augmentation is applied **only** during training, not during inference. Bounding boxes are automatically tracked and updated with the images. 416 x 416 examples pictured below.

<img src="https://user-images.githubusercontent.com/26833433/50525037-6cbcbc00-0ad9-11e9-8c38-9fd51af530e0.jpg">

# Inference

Run `detect.py` to apply trained weights to an image, such as `zidane.jpg` from the `data/samples` folder:

**YOLOv3:** `detect.py --cfg cfg/yolov3.cfg --weights weights/yolov3.pt`
<img src="https://user-images.githubusercontent.com/26833433/50524393-b0adc200-0ad5-11e9-9335-4774a1e52374.jpg" width="800">

**YOLOv3-tiny:** `detect.py --cfg cfg/yolov3-tiny.cfg --weights weights/yolov3-tiny.pt`
<img src="https://user-images.githubusercontent.com/26833433/50374155-21427380-05ea-11e9-8d24-f1a4b2bac1ad.jpg" width="800">

## Webcam

Run `detect.py` with `webcam=True` to show a live webcam feed.

# Pretrained Weights

- https://drive.google.com/drive/folders/1uxgUBemJVw9wZsdpboYbzUN4bcRhsuAI
