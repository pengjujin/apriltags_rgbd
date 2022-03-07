#!/usr/bin/env python3
# Author: Amy Phung

# TODO: move yaml parsing to utils
# TODO: remove hardcoded tag size

# Tell python where to find apriltags_rgbd code
import sys
import os
fpath = os.path.join(os.path.dirname(__file__), "apriltags_rgbd")
sys.path.append(fpath)

# Python Imports
import numpy as np
import cv2
import rgb_depth_fuse as fuse

# ROS Imports
import rospy
import tf2_ros
from geometry_msgs.msg import Transform, Vector3
from std_msgs.msg import Bool
from sensor_msgs.msg import Image, CameraInfo
from apriltag_msgs.msg import ApriltagArrayStamped
from message_filters import TimeSynchronizer, Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge, CvBridgeError


class ApriltagsRgbdNode():
    def __init__(self):
        rospy.init_node("apriltags_rgbd")
        self.rate = rospy.Rate(10) # 10 Hz

        # Subscribers
        tss = ApproximateTimeSynchronizer([
            Subscriber("/kinect2/hd/camera_info", CameraInfo),
            Subscriber("/kinect2/hd/image_color_rect", Image),
            Subscriber("/kinect2/hd/image_depth_rect", Image),
            Subscriber("/kinect2/hd/tags", ApriltagArrayStamped)], 1 ,0.5)
        tss.registerCallback(self.tagCallback)

        self.camera_info_data = None
        self.rgb_data = None
        self.depth_data = None
        self.tag_data = None

        # cfg_param = rospy.get_param("~apriltags_rbgd_config")
        # self.config = self.parseConfig(cfg_param)


# camera info
# rgb_image_path
# depth_image_path
# Need to convert to opencv


    def tagCallback(self, camera_info_data, rgb_data, depth_data, tag_data):
        self.camera_info_data = camera_info_data
        self.rgb_data = rgb_data
        self.depth_data = depth_data
        self.tag_data = tag_data

        # print("here")
        # # print(self.camera_info)
        # print(camera_info_data)
        # try:
        #     rgb_image = self.bridge.imgmsg_to_cv2(rgb_data, "bgr8")
        #     depth_image = self.bridge.imgmsg_to_cv2(depth_data, "16UC1")
        # except CvBridgeError as e:
        #     print(e)
        #
        # all_detections = tag_data.detections
        # # allstring = ''
        # # for current_detection in all_detections:
        # #     detection_string = self.format_AprilTagDetections(current_detection)
        # #     allstring = allstring + detection_string
        # self.rgb_image = rgb_image
        # self.depth_image = depth_image
        # # self.detection_string = allstring
        # cv2.imshow('Image', rgb_image)
        # key = cv2.waitKey(1)
        # if (key > -1):
        #     self.key_press()

    # def parseConfig(self, cfg_param):
    #     config = {}
    #     config["ids"] = [1,2,3,4]
    #
    #     return config

    def run(self):
        while not rospy.is_shutdown():

            # Estimate pose
            nrvec, ntvec = fuse.solvePnP_RGBD(rgb_image, depth_image, object_pts, image_pts, mtx, dist, 0)
        	print("Test rvec:")
        	print(nrvec)
        	print("Test tvec:")
        	print(ntvec)

            self.rate.sleep()

if __name__ == '__main__':
    node = ApriltagsRgbdNode()
    node.run()
