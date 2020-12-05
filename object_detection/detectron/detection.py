#!/usr/bin/env python3

import cv2
import numpy
import rospy
import torch
from sensor_msgs.msg import Image
import detectron_predictor as pr
from timeit import default_timer as timer

def list_to_array(image_list, height, width):
    rgb_list = []
    for i, x in enumerate(image_list):
        rgb_list.append(x)
    # r_list = numpy.reshape(r_list, (height, width))
    # g_list = numpy.reshape(g_list, (height, width))
    # b_list = numpy.reshape(b_list, (height, width))
    # return numpy.array([r_list, g_list, b_list])
    
    if len(rgb_list) == 3 * height * width:
        bgr_array = numpy.reshape((rgb_list), (height, width, 3))
        rgb_array = numpy.flip(bgr_array, 2)
        return numpy.flip(rgb_array, 0)
    else:
        return numpy.array(image_list)

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

def main():
    global raw_image_msg

    rospy.init_node('object_detection')
    rospy.Subscriber("/raw_image", Image, object_detection_callback)

    p = pr.Predictor()
    image_pub = rospy.Publisher("/image", Image, queue_size=10)

    while raw_image_msg is None: continue

    azure_addr = "137.135.81.74"

    while not rospy.is_shutdown():
        image_array = list_to_array(raw_image_msg.data, raw_image_msg.height, raw_image_msg.width)

        try:
            cv2.imwrite("./input.png", image_array)
            # upload to azure
            os.system("scp -r ./input.png azureuser@{}:/home/azureuser".format(azure_addr))
            # download from azure
            os.system("scp -r azureuser@:/home/azureuser ./output.png".format(azure_addr)) 
            output = cv2.imread("output.png")
        except:
            print("Something went wrong when communicating with azure, trying again")
            continue


        # print(image_array)
        # output = p.transform(image_array)
        # print(numpy.all(output == 0))
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