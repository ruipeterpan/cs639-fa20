import cv2
import numpy as np
import os

class Tracker:
    """Implementation for object tracker"""

    def __init__(self, firstFrame):
        """Initialize tracker members"""
        # self.x, self.y, self.width, self.height = 600,200,100,300
        # self.track_window = (self.x, self.y, self.width, self.height)
        self.roi = cv2.imread("./template.png")  # self.roi is the template for template matching
        h, w = self.roi.shape[0], self.roi.shape[1]
        method = eval('cv2.TM_CCOEFF')
        res = cv2.matchTemplate(firstFrame,self.roi,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        self.x = top_left[0]
        self.y = top_left[1]
        self.width = w
        self.height = h
        self.track_window = (self.x, self.y, self.width, self.height)



        self.hsv_roi = cv2.cvtColor(self.roi, cv2.COLOR_BGR2HSV) 
        self.mask = cv2.inRange(self.hsv_roi,  
                  np.array((0., 60., 40.)), 
                  np.array((180., 255., 255))) 
        self.roi_hist = cv2.calcHist([self.hsv_roi],  
                       [0],self.mask, 
                       [180],  
                       [0, 180]) 
        self.term_crit = ( cv2.TERM_CRITERIA_EPS |  
             cv2.TERM_CRITERIA_COUNT, 15, 2) 
        self.path_to_src = os.path.dirname(os.path.realpath(__file__))


    def track(self, frame):
        """Takes in an input frame and outputs a frame with the desired object
        surronded with a tracker box
        Args:
            frame (numpy.ndarray): A frame of the video as np array 
        Returns:
            numpy.ndarray: An np array of the same size with the desired object tagged
        """
        # frame = np.float32(frame)  # transform frame from int array to float32
        # frame = frame.astype(np.float32)
        print(frame.dtype)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],self.roi_hist,[0,180],1)
        # apply meanshift to get the new location
        ret, self.track_window = cv2.meanShift(dst, self.track_window, self.term_crit)
        # Draw it on image
        self.x, self.y, self.width, self.height = self.track_window
        img2 = cv2.rectangle(frame, (self.x,self.y), (self.x+self.width,self.y+self.height), 255,2)
        # cv2.imshow('img2',cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        cv2.imwrite('./test.png',img2)

        return self.track_window, img2

