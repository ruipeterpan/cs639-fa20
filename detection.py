#!/usr/bin/env python3

import numpy
import rospy
import torch
from sensor_msgs.msg import Image
import detectron_predictor as pr
p = pr.Predictor()
p.transform('input.png')
image_pub = rospy.Publisher("/image", Image, queue_size=10)

def list_to_tensor(image_list, height, width):
    r_list = []
    g_list = []
    b_list = []
    for i, x in enumerate(image_list):
        if i % 3 == 0:
            r_list.append(x)
        elif i % 3 == 1:
            g_list.append(x)
        else:
            b_list.append(x)
    r_list = numpy.reshape(r_list, (height, width))
    g_list = numpy.reshape(g_list, (height, width))
    b_list = numpy.reshape(b_list, (height, width))
    return torch.tensor(numpy.array([r_list, g_list, b_list])).div(255)

def tensor_to_list(tensor):
    tensor = torch.flatten(tensor)
    rgb_list = []
    for x in tensor:
        for i in range(3):
            rgb_list.append(int(x))
    return rgb_list
    
def object_detection_callback(msg):
    image_list = msg.data
    tensor = list_to_tensor(image_list, msg.height, msg.width)
    print(tensor.size())
    output = p.transform(tensor)
    h = list(output.size())[0]
    w = list(output.size())[1]

    img_msg = Image()
    img_msg.header.stamp = rospy.Time.now()
    img_msg.header.frame_id = 'a'
    img_msg.height = h
    img_msg.width = w
    img_msg.encoding = 'rgb8'
    img_msg.is_bigendian = 1
    img_msg.step = 3 * w

    rgb_list = tensor_to_list(output)
    # print(list(filter(lambda x: (x > 0), rgb_list)))
    img_msg.data = rgb_list

    image_pub.publish(img_msg)

if __name__ == '__main__':
    rospy.init_node('object_detection')
    rospy.Subscriber("/raw_image", Image, object_detection_callback)
    rospy.spin()