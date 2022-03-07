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
from geometry_msgs.msg import TransformStamped, Vector3, Quaternion
from std_msgs.msg import Bool
from sensor_msgs.msg import Image, CameraInfo
from apriltag_msgs.msg import ApriltagArrayStamped
from message_filters import TimeSynchronizer, Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge, CvBridgeError
import tf_conversions


# TODO: Remove hardcode
# Experiment Parameters
# mtx = Camera Matrix
# dist = Camera distortion
# tag_size = Apriltag size
# tag_radius = Apriltag size / 2
MTX = np.array([1078.578826404335, 0.0, 959.8136576469886, 0.0, 1078.9620428822643, 528.997658280927, 0.0, 0.0, 1.0]).reshape(3,3)
DIST = np.array([0.09581801408471438, -0.17355242497569437, -0.002099527275158767, -0.0002485026755700764, 0.08403352203200236]).reshape((5,1))
TAG_SIZE = 0.06925
TAG_RADIUS = TAG_SIZE / 2.0

TAG_PREFIX = "detected_"
class ApriltagsRgbdNode():
    def __init__(self):
        rospy.init_node("apriltags_rgbd")
        self.rate = rospy.Rate(10) # 10 Hz

        # CV bridge
        self.bridge = CvBridge()

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

        self.new_data = False

        # ROS tf
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()


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
        self.new_data = True

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
            # Check for new data
            if not self.new_data:
                self.rate.sleep()
                continue

            self.new_data = False

            # Convert ROS images to OpenCV frames
            try:
                rgb_image = self.bridge.imgmsg_to_cv2(self.rgb_data, "bgr8")
                depth_image = self.bridge.imgmsg_to_cv2(self.depth_data, "16UC1")
            except CvBridgeError as e:
                print(e)
                continue

            # Extract metadata
            header = self.camera_info_data.header

            # Estimate pose of each tag
            for tag in self.tag_data.apriltags:
                tag_id, object_pts, image_pts = self.parseTag(tag)

                # Estimate pose
                nrvec, ntvec = fuse.solvePnP_RGBD(rgb_image, depth_image,
                    object_pts, image_pts, MTX, DIST, 0)
                print("Test rvec:")
                print(nrvec)
                print("Test tvec:")
                print(ntvec)

                # Update tf tree
                output_tf = self.composeTfMsg(ntvec, nrvec, header, tag_id)
                self.tf_broadcaster.sendTransform(output_tf)

            self.rate.sleep()

    def parseTag(self, tag):
        id = tag.id

        ob_pt0 = [-TAG_RADIUS, -TAG_RADIUS, 0.0]
        ob_pt1 = [ TAG_RADIUS, -TAG_RADIUS, 0.0]
        ob_pt2 = [ TAG_RADIUS,  TAG_RADIUS, 0.0]
        ob_pt3 = [-TAG_RADIUS,  TAG_RADIUS, 0.0]
        ob_pts = ob_pt0 + ob_pt1 + ob_pt2 + ob_pt3
        object_pts = np.array(ob_pts).reshape(4,3)

        im_pt0 = [tag.corners[0].x, tag.corners[0].y]
        im_pt1 = [tag.corners[1].x, tag.corners[1].y]
        im_pt2 = [tag.corners[2].x, tag.corners[2].y]
        im_pt3 = [tag.corners[3].x, tag.corners[3].y]

        im_pts = im_pt0 + im_pt1 + im_pt2 + im_pt3
        image_pts = np.array(im_pts).reshape(4,2)

        return id, object_pts, image_pts

    def composeTfMsg(self, trans, rot, header, tag_id):
        output_tf = TransformStamped()

        # Update header info
        output_tf.header = header
        output_tf.child_frame_id = TAG_PREFIX + str(tag_id)

        # Populate translation info
        output_tf.transform.translation = Vector3(*trans)

        # Populate rotation info
        q = tf_conversions.transformations.quaternion_from_euler(*rot)
        output_tf.transform.rotation = Quaternion(*q)

        print(output_tf)
        return output_tf

if __name__ == '__main__':
    node = ApriltagsRgbdNode()
    node.run()
