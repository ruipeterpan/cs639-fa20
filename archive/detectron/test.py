import numpy as np 
import cv2
from tracker import Tracker
  
# Read the input video 
cap = cv2.VideoCapture('./imgs/video.mp4') 

ret, frame = cap.read() 

t = Tracker(frame)

while True:
    ret, frame = cap.read() 
    if not ret:
        break
    window, output_img = t.track(frame)
    print(window)
    cv2.imwrite("./imgs/output.png", output_img)
    