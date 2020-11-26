#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
import torch
from sensor_msgs.msg import Image
import detectron_predictor as pr
from timeit import default_timer as timer

def list_to_array(image_list, height, width):
    rgb_list = []
    for _, x in enumerate(image_list):
        rgb_list.append(x)
    # r_list = np.reshape(r_list, (height, width))
    # g_list = np.reshape(g_list, (height, width))
    # b_list = np.reshape(b_list, (height, width))
    # return np.array([r_list, g_list, b_list])
    
    if len(rgb_list) == 3 * height * width:
        bgr_array = np.reshape((rgb_list), (height, width, 3))
        rgb_array = np.flip(bgr_array, 2)
        return np.flip(rgb_array, 0)
    else:
        return np.array(image_list)

def array_to_list(array):
    h = array.shape[0]
    w = array.shape[1]
    rgb_list = []
    for i in range(len(array[0]))[::-1]:
        for j in range(len(array[1])):
            rgb_list.append(int(array[i][j][2]))
            rgb_list.append(int(array[i][j][1]))
            rgb_list.append(int(array[i][j][0]))
    return rgb_list, h, w

raw_image_msg = None
def object_detection_callback(msg):
    global raw_image_msg
    raw_image_msg = msg


class Tracker:
    """Implementation for object tracker
    """
    def __init__(self):
        """Initialize tracker members
        """
        self.x, self.y, self.width, self.height = 0, 0, 0, 0
        self.track_window = None
        self.roi = None
        self.hsv_roi = None
        self.mask = None
        self.roi_hist = None
        self.term_crit = None
        self.isFirstIter = True

    def track(self, frame):
        """Takes in an input frame and outputs a frame with the desired object
        surronded with a tracker box

        Args:
            frame (numpy.ndarray): A frame of the video as np array 

        Returns:
            numpy.ndarray: An np array of the same size with the desired object tagged
        """
        if self.isFirstIter: # initialization, set initial tracking region & calculate histogram
            self.isFirstIter = False
            self.x, self.y, self.width, self.height = cv2.selectROI(frame, False)
            self.track_window = (self.x, self.y, self.width, self.height) 

            # set up the Region of 
            # Interest for tracking 
            self.roi = frame[self.y:self.y + self.height, 
                        self.x : self.x + self.width]
            
            # convert ROI from BGR to 
            # HSV format 
            self.hsv_roi = cv2.cvtColor(frame, 
                                cv2.COLOR_BGR2HSV)
            
            # perform masking operation 
            self.mask = cv2.inRange(self.hsv_roi,  
                            np.array((0., 60., 40.)), 
                            np.array((180., 255., 255)))

            self.roi_hist = cv2.calcHist([self.hsv_roi],  
                                [0],self.mask, 
                                [180],  
                                [0, 180])
            
            cv2.normalize(self.roi_hist, self.roi_hist, 
                        0, 255, cv2.NORM_MINMAX) 

            # Setup the termination criteria,  
            # either 15 iteration or move by 
            # atleast 2 pt 
            self.term_crit = (cv2.TERM_CRITERIA_EPS |  
                        cv2.TERM_CRITERIA_COUNT, 15, 2)
        
        _, self.frame1 = cv2.threshold(frame, 
                        180, 155, 
                        cv2.THRESH_TOZERO_INV) 

        # convert from BGR to HSV 
        # format. 
        self.hsv = cv2.cvtColor(self.frame1,  
                        cv2.COLOR_BGR2HSV) 
    
        self.dst = cv2.calcBackProject([self.hsv],  
                                [0],  
                                self.roi_hist,  
                                [0, 180], 1) 
        
        # apply Camshift to get the  
        # new location 
        self.ret2, self.track_window = cv2.CamShift(self.dst, 
                                        self.track_window, 
                                        self.term_crit) 
    
        # Draw it on image 
        pts = cv2.boxPoints(self.ret2) 
        
        # convert from floating 
        # to integer 
        pts = np.int0(pts) 
    
        # Draw Tracking window on the 
        # video frame. 
        output = cv2.polylines(frame,  
                            [pts],  
                            True,  
                            (0, 255, 255),  
                            2)
        
        return output




def main():
    global raw_image_msg

    rospy.init_node('object_detection')
    rospy.Subscriber("/raw_image", Image, object_detection_callback)

    p = pr.Predictor()
    image_pub = rospy.Publisher("/image", Image, queue_size=10)

    while raw_image_msg is None: continue


    tracker = Tracker()


    while not rospy.is_shutdown():
        image_array = list_to_array(raw_image_msg.data, raw_image_msg.height, raw_image_msg.width)
        cv2.imwrite("./test.png", image_array)
        # print(image_array)
        
        # transform image_array to output
        output = tracker.track(image_array)

        # print(np.all(output == 0))
        rgb_list, h, w = array_to_list(output)
        # print(rgb_list)
        image_msg = Image()
        image_msg.header.stamp = rospy.Time.now()
        image_msg.header.frame_id = 'a'
        image_msg.height = h
        image_msg.width = w
        image_msg.encoding = 'bgr8'
        image_msg.is_bigendian = 1
        image_msg.step = 3 * w
        image_msg.data = rgb_list

        image_pub.publish(image_msg)

if __name__ == '__main__':
    main()