# execute this script on azure

from predictor import Predictor
import cv2
import os

p = Predictor()

while True:
    # try:
    #     input = cv2.imread("/home/azureuser/input.png")
    #     print(input)
    #     output = p.transform(input)
    #     cv2.imwrite("/home/azureuser/output.png", output)
    #     print("Processing image")
    # except Exception as e:
    #     print(e)
    #     continue
    input = cv2.imread("/home/azureuser/input.png")
    output = p.transform(input)
    cv2.imwrite("/home/azureuser/output.png", output)
    print("Processing image")
