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

`/archive` contains the code we used for the robotics framework integration. 

Please see the README in each folder for more detailed information on how to replicate our results.

Note that the robotics side of the code is not possible to set up as it requires a version of the RelaxedIK framework which is not open-sourced yet (???). Thus, we only provide instructions on standalone scripts for the vision tasks that can be executed without the robotics framework.

