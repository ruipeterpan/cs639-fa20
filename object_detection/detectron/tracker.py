import cv2
import numpy as np
import os

class Tracker:
    """Implementation for object tracker"""

    def __init__(self):
        """Initialize tracker members"""
        self.path_to_src = os.path.dirname(os.path.realpath(__file__))
        self.x, self.y, self.width, self.height = 300,180,50,120
        self.track_window = (self.x, self.y, self.width, self.height)
        self.roi = cv2.imread(self.path_to_src + "/imgs/template.png")
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

    def track(self, frame):
        """Takes in an input frame and outputs a frame with the desired object
        surronded with a tracker box
        Args:
            frame (numpy.ndarray): A frame of the video as np array 
        Returns:
            numpy.ndarray: An np array of the same size with the desired object tagged
        """
        frame = np.uint8(frame)  # transform frame from int array to float32
        print(frame.dtype)
        # frame = frame.astype(np.float32)
        print(frame.shape)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],self.roi_hist,[0,180],1)
        # apply meanshift to get the new location
        ret, self.track_window = cv2.meanShift(dst, self.track_window, self.term_crit)
        # Draw it on image
        self.x, self.y, self.width, self.height = self.track_window
        img2 = cv2.rectangle(frame, (self.x,self.y), (self.x+self.width,self.y+self.height), 255,2)
        # cv2.imshow('img2',img2)
        cv2.imwrite('./test.png',img2)

        return self.track_window, img2

