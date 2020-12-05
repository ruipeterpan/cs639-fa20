# import numpy as np
# import time
# from tracker import Tracker
# import cv2

# # cap = cv2.VideoCapture('video0.mp4') 

# # t = Tracker()

# # window_name = 'image'

# # ret, frame = cap.read()

# # print(ret)

# # cv2.imshow(window_name, frame)

# img = cv2.imread("./man.PNG")
# # img = np.array(img, dtype = np.uint8)
# cv2.imshow("window", img)

# time.sleep(3)

# # while True:
# # 	ret, frame = cap.read()
# # 	cv2.imshow(window_name, frame)
# # 	pts, output = t.track(frame)
# # 	cv2.imshow(window_name, output)


import numpy as np 
import cv2
from tracker import Tracker
  
# Read the input video 
cap = cv2.VideoCapture('./video.mp4')

_, frame = cap.read() 

t = Tracker(frame)

while True:
	ret, frame = cap.read() 
	if not ret:
		break
	window, output_img = t.track(frame)
	print(window)
	# cv2.imshow('Canvas', output_img)
  
# # take first frame of the 
# # video 
# ret, frame = cap.read() 
  
# # setup initial region of 
# # tracker 

# x, y, width, height = 600,200,100,300
# # x, y, width, height = cv2.selectROI(frame, False)

# track_window = (x, y,  
#                 width, height) 


  
# # set up the Region of 
# # Interest for tracking 
# # roi = frame[y:y + height, 
# #             x : x + width] 

# roi = cv2.imread("./test.png")
  
# # convert ROI from BGR to 
# # HSV format 
# # hsv_roi = cv2.cvtColor(frame, 
# #                       cv2.COLOR_BGR2HSV) 
# hsv_roi = cv2.cvtColor(roi, 
#                       cv2.COLOR_BGR2HSV) 
  
# # perform masking operation 
# mask = cv2.inRange(hsv_roi,  
#                   np.array((0., 60., 40.)), 
#                   np.array((180., 255., 255))) 
  
# roi_hist = cv2.calcHist([hsv_roi],  
#                        [0],mask, 
#                        [180],  
#                        [0, 180]) 


# # roi_hist = cv2.calcHist([hsv_roi],  
# #                        [0],None, 
# #                        [180],  
# #                        [0, 180]) 
  
# cv2.normalize(roi_hist, roi_hist, 
#              0, 255, cv2.NORM_MINMAX) 
  
  
# # Setup the termination criteria,  
# # either 15 iteration or move by 
# # atleast 2 pt 
# term_crit = ( cv2.TERM_CRITERIA_EPS |  
#              cv2.TERM_CRITERIA_COUNT, 15, 2) 
  
  
# while(1): 
      
#     ret, frame = cap.read() 
      
#     # # Resize the video frames. 
#     # frame = cv2.resize(frame,  
#     #                   (720, 720),  
#     #                   fx = 0, fy = 0, 
#     #                   interpolation = cv2.INTER_CUBIC) 
      
#     # cv2.imshow('Original', frame) 


#     # perform thresholding on  
#     # the video frames 
#     ret1, frame1 = cv2.threshold(frame, 
#                                 180, 155, 
#                                 cv2.THRESH_TOZERO_INV) 
  
#     # convert from BGR to HSV 
#     # format. 
#     hsv = cv2.cvtColor(frame1,  
#                       cv2.COLOR_BGR2HSV) 
  
#     dst = cv2.calcBackProject([hsv],  
#                              [0],  
#                              roi_hist,  
#                              [0, 180], 1) 
      
#     # apply Camshift to get the  
#     # new location 
#     ret2, track_window = cv2.CamShift(dst, 
#                                      track_window, 
#                                      term_crit) 
  
#     # Draw it on image 
#     pts = cv2.boxPoints(ret2) 
      
#     # convert from floating 
#     # to integer 
#     pts = np.int0(pts) 
  
#     # Draw Tracking window on the 
#     # video frame. 
#     Result = cv2.polylines(frame,  
#                           [pts],  
#                           True,  
#                           (0, 255, 255),  
#                           2) 


#     cv2.imshow('Camshift', Result) 
  
#     # set ESC key as the 
#     # exit button. 
#     k = cv2.waitKey(30) & 0xff
      
#     if k == 27: 
#         break
          
  
# # Release the cap object 
# cap.release() 
  
# # close all opened windows 
# cv2.destroyAllWindows() 