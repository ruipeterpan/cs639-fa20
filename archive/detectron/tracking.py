#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
import torch
from sensor_msgs.msg import Image
from timeit import default_timer as timer
from tracker import Tracker

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
def object_tracing_callback(msg):
    global raw_image_msg
    raw_image_msg = msg

def main():
    global raw_image_msg

    rospy.init_node("object_tracing")
    rospy.Subscriber("/raw_image", Image, object_tracing_callback)

    image_pub = rospy.Publisher("/image", Image, queue_size=10)

    while raw_image_msg is None: continue

    image_array = list_to_array(raw_image_msg.data, raw_image_msg.height, raw_image_msg.width)
    while np.all(image_array == 0): 
        print("zeros!")
        image_array = list_to_array(raw_image_msg.data, raw_image_msg.height, raw_image_msg.width)
        continue

    tracker = Tracker(image_array)

    while not rospy.is_shutdown():
        image_array = list_to_array(raw_image_msg.data, raw_image_msg.height, raw_image_msg.width)
        # cv2.imwrite("./input.png", image_array)
        # print(image_array)

        # transform image_array to output
        pts, output = tracker.track(image_array, method="meanshift")
        # pts, output = tracker.track(image_array, method="camshift")
        print(pts)

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
