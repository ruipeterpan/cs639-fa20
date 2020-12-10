import torch, torchvision
# see https://pytorch.org/ for installing torch. Use the --upgrade flag to upgrade packages using pip. 
assert torch.__version__.startswith("1.7")
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
import time

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

from timeit import default_timer as timer

class Predictor():
  """
  Initializes a detectron2 predictor
  """
  def __init__(self):
    # create a detectron2 config and a detectron2 predictor to run inference on the image
    self.cfg = get_cfg()
    # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
    self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    self.cfg.MODEL.DEVICE='cpu'  # use cpu when gpu/cuda is not available; otherwise use "cuda"
    self.predictor = DefaultPredictor(self.cfg)

  def transform(self, imgArray):
    """Sends a numpy array through the predictor and outputs a numpy array of the same size

    Args:
        imgArray (numpy.ndarray): Image as a numpy array
    """
    timestamp = timer()
    outputs = self.predictor(imgArray)
    print("Prediction took", round(timer() - timestamp, 4), "seconds")
    v = Visualizer(imgArray[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.0)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite("./test_output.png", out.get_image()[:, :, ::-1])
    return out.get_image()[:, :, ::-1]







