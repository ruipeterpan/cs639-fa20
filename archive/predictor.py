import os
import torch
import torchvision
import numpy
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from PIL import Image


def get_instance_segmentation_model(num_classes):
  # load an instance segmentation model pre-trained on COCO
  model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

  # get the number of input features for the classifier
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  # replace the pre-trained head with a new one
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

  # now get the number of input features for the mask classifier
  in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
  hidden_layer = 256
  # and replace the mask predictor with a new one
  model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                     hidden_layer,
                                                     num_classes)

  return model


class Predictor():
  """
  Usage: 
  import predictor as pr
  p = pr.Predictor()
  p.transform('input.png')
  output image is written to output.png
  """
  def __init__(self):
    num_classes = 2
    model = get_instance_segmentation_model(num_classes)
    if os.path.isfile("model.pth"):
      checkpoint = torch.load("model.pth", map_location=torch.device('cpu'))
      model.load_state_dict(checkpoint["model"])
      torch.set_rng_state(checkpoint["rng_state"])
    self.model = model
    self.device = torch.device('cpu')

  def transform_from_file(self, imgName):
    # read image from local file, writes output to local file
    print("Transforming image", imgName)
    img = Image.open(imgName, mode='r')  # read image from local file
    img = torchvision.transforms.ToTensor()(img)  # transform image to tensor of size torch.Size([3, 512, 512])
    self.model.eval()  # tell the neural net to work in eval mode
    with torch.no_grad():  # deactivates autograd, no backprop
        prediction = self.model([img.to(self.device)])
        # prediction[0]['masks'][0, 0] is the output as a torch tensor
        # it has size torch.Size([512, 512]) and its values are probably from 0 to 1
    mask = Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())  # mask is a PIL image of size 512*512
    # https://discuss.pytorch.org/t/pytorch-pil-to-tensor-and-vice-versa/6312/10
    mask.save("output.png")
    print("Transform of image", imgName, "completed!")

  def transform(self, img):
    # img should be a tensor of size torch.Size([3, 512, 512])
    self.model.eval()  # tell the neural net to work in eval mode
    with torch.no_grad():  # deactivates autograd, no backprop
        prediction = self.model([img.to(self.device)])
        # prediction[0]['masks'][0, 0] is the output as a torch tensor
        # it has size torch.Size([512, 512]) and its values are probably from 0 to 1
        return prediction[0]['masks'][0, 0]  # do .mul(255) if we need the values to be from 0 to 255

