# Real-time Vision Task with a High Degree of Freedom Robot Arm

CS 639 @ UW-Madison: Computer Vision (Fall 2020) Course Project

Group Members:

* Haochen Shi
* Rui Pan
* Chenhao Lu

## Requirements for replicating our result

* Python3
* PyTorch (version > 1.7)
* torchvision
* OpenCV
* [Detectron](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md) for task 2

## Repository Overview

`/docs` contains all source code for the static websites (proposal, mid-term report, final report).

`/panorama` contains the source code for our first task, taking a stable panorama using the robot arm.

`/detection` contains the source code for our second task, real-time object detection.

`/tracking` contains the source code for our third task, real-time object tracking.

Please see the README in each folder for more detailed information on how to replicate our results.

Note that the robotics side of the code is not possible to set up as it requires a version of the RelaxedIK framework which is not open-sourced yet (???). Thus, we only provide instructions on standalone scripts for the vision tasks that can be executed without the robotics framework.

## /panorama

In `main()`, modify the input to `readImgs()` so that it points to a folder containing images to be stitched. Then, do `python3 stitching.py` to stitch the input images. The result image is written to `./images/results.jpg`.

## /detection

We encapsulated the detector in a class `Predictor` in `detectron_predictor.py`. On initialization, `Predictor` sets up detectron2 configurations, download a Mask-RCNN model (wrapped around ResNet50) from detectron2's model zoo, and sets up the default detectron2 predictor wrapper.

`detection.py` is used by our robotics framework -- please ignore it.

### Usage

```python
import detectron_predictor as pr
import cv2
p = pr.Predictor()
input = cv2.imread("")  # numpy.ndarray
output = p.transform(input)
output  # numpy.ndarray of the same size as the input
```

## /tracking

We encapsulated both algorithms (meanshift & camshift) inside the class `Tracker` in `tracker.py`.

On initialization, `Tracker` reads in a local template image for histogram calculations. It also uses OpenCV template matching to find the initial window location and size.

When `track()` is invoked, it takes in an input image as a `numpy.ndarray`, chooses a tracking algorithm to run, and returns the image and the `(x, y, width, height)` coordinates of the bounding box.

### Usage

```python
from tracker import Tracker  # before this, modify the path to the template image in Tracker
import cv2
t = Tracker()
input = cv2.imread("")  # numpy.ndarray
pts, output = t.track(input, method="camshift")  # meanshift/camshift
output  # numpy.ndarray of the same size as the input
pts  # tuple that contains (x, y, width, height) of the bounding box
```