# execute this script on azure

from predictor import Predictor
import cv2

p = Predictor()

while True:
    try:
        input = cv2.imread("./input.png")
        output = p.transform(input)
        cv2.imwrite("./output.png", output)
    except:
        print("Something went wrong getting the image, trying again")
        continue
