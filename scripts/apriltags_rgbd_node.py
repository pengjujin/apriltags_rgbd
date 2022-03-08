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
import math

# ROS Imports
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped, Vector3, Quaternion, Pose
from std_msgs.msg import Bool
from sensor_msgs.msg import Image, CameraInfo
from apriltag_msgs.msg import ApriltagArrayStamped
from message_filters import TimeSynchronizer, Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge, CvBridgeError
import tf_conversions
from tf.transformations import quaternion_from_matrix
from visualization_msgs.msg import MarkerArray, Marker


# For debugging
DEBUG = True
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32


# TODO: Remove hardcode
# Experiment Parameters
# mtx = Camera Matrix
# dist = Camera distortion
# tag_size = Apriltag size
# tag_radius = Apriltag size / 2
MTX = np.array([1078.578826404335, 0.0, 959.8136576469886, 0.0, 1078.9620428822643, 528.997658280927, 0.0, 0.0, 1.0]).reshape(3,3)
DIST = np.zeros((5,1))#np.array([0.09581801408471438, -0.17355242497569437, -0.002099527275158767, -0.0002485026755700764, 0.08403352203200236]).reshape((5,1))
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

        # Publishers
        if DEBUG:
            self.tag_pt_pub = rospy.Publisher("extracted_tag_pts", PointCloud, queue_size=10)
            self.corner_pt_pub = rospy.Publisher("corner_tag_pts", PointCloud, queue_size=10)
            self.marker_arr_pub = rospy.Publisher("visualization_marker_array", MarkerArray, queue_size=10)


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

            if DEBUG:
                tag_pts = PointCloud()
                corner_pts = PointCloud()
                marker_array_msg = MarkerArray()

            # Estimate pose of each tag
            for tag in self.tag_data.apriltags:
                tag_id, object_pts, image_pts = self.parseTag(tag)

                # Estimate pose
                depth_plane_est, all_pts = fuse.sample_depth_plane(depth_image, image_pts, MTX)
                depth_points = fuse.getDepthPoints(image_pts, depth_plane_est, depth_image, MTX)





                # nrvec, ntvec = fuse.solvePnP_RGBD(rgb_image, depth_image,
                #     object_pts, image_pts, MTX, DIST, 0)

                # vec_a = depth_points[0] - depth_points[1]
                # vec_b = depth_points[2] - depth_points[1]
                #
                # vec_a_len = np.linalg.norm(vec_a)
                # vec_b_len = np.linalg.norm(vec_b)
                #
                # norm = np.cross(vec_a, vec_b)
                #
                # q_xyz = np.cross(vec_a, vec_b)
                # q_w =  np.sqrt(vec_a_len**2 * vec_b_len**2) + np.dot(vec_a, vec_b)
                # print(np.dot(vec_a, vec_b))
                # quat = [*q_xyz, q_w]
                # quat = quat / np.linalg.norm(quat)
                # quat = [0,0,0,1]

                # q_xyz = vec_a
                # q_w
                # print(quat)
                # vec_a = vec_a / np.linalg.norm(vec_a)
                # # print("Test rvec:")
                # # print(nrvec)
                # # print("Test tvec:")
                # # print(ntvec)
                # #
                # angle = math.atan2( vec_a[0], vec_a[1] )
                # # print(angle)
                # # angle =
                #
                # qx = norm[0] * math.sin(angle/2)
                # qy = norm[1] * math.sin(angle/2)
                # qz = norm[2] * math.sin(angle/2)
                # qw = math.cos(angle/2)
                # quat = [qx, qy, qz, qw]
                # quat = quat / np.linalg.norm(quat)

                # Compute center of plane
                center_pt = np.mean(depth_points, axis=0)

                # Plane Normal Vector
                n_vec = depth_plane_est.mean.n
                # n_vec = np.array([1,0,0])#depth_plane_est.mean.n
                n_norm = np.linalg.norm(n_vec)

                # Compute point in direction of x-axis
                x_pt = (depth_points[1] + depth_points[2]) / 2
                # x_pt = center_pt + np.array([0,1,0])

                # Compute first orthogonal vector - project point onto plane
                u = x_pt - center_pt
                v = n_vec
                v_norm = n_norm
                x_vec = u - (np.dot(u, v)/v_norm**2)*v

                print(x_vec)

                # Compute second orthogonal vector - take cross product
                y_vec = np.cross(n_vec, x_vec)

                # Normalize vectors
                x_vec = x_vec / np.linalg.norm(x_vec)
                y_vec = y_vec / np.linalg.norm(y_vec)
                n_vec = n_vec / np.linalg.norm(n_vec)

                # TODO: this can be optimized
                x_vec1 = list(x_vec) + [0]
                y_vec1 = list(y_vec) + [0]
                n_vec1 = list(n_vec) + [0]



                #
                # x_vec = [0,1,0]
                # y_vec = [0,0,1]
                # n_vec = [1,0,0]
                #
                # # Compute quaternion from rotation matrix
                # q_w = 1/2 * np.sqrt(1 + x_vec[0] + y_vec[1] + n_vec[2])
                # q_x = 1/4 * q_w * (y_vec[2] - n_vec[1])
                # q_y = 1/4 * q_w * (n_vec[0] - x_vec[2])
                # q_z = 1/4 * q_w * (x_vec[1] - y_vec[0])
                R_t = np.array([x_vec1,y_vec1,n_vec1,[0,0,0,1]])

                quat = quaternion_from_matrix(R_t.transpose())#[q_x, q_y, q_z, q_w]

                # Normalize quaternion
                quat = quat / np.linalg.norm(quat)
                print(quat)

                # Update tf tree
                output_tf = self.composeTfMsg(center_pt, quat, header, tag_id)
                self.tf_broadcaster.sendTransform(output_tf)

                if DEBUG:
                    tag_pts.header = header
                    corner_pts.header = header

                    for i in range(len(all_pts)):
                        tag_pts.points.append(Point32(*all_pts[i]))

                    for i in range(len(depth_points)):
                        corner_pts.points.append(Point32(*depth_points[i]))

                    marker = Marker()
                    marker.header = header
                    marker.id = tag_id
                    marker.type = 0
                    marker.pose.position = Point32(0,0,0)
                    marker.pose.orientation = Quaternion(0,0,0,1)
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                    marker.color.a = 1.0
                    marker.scale.x = 0.005
                    marker.scale.y = 0.01
                    marker.scale.z = 0.0
                    pt1 = Point32(*center_pt)
                    pt2 = Point32(*(center_pt + n_vec/5))
                    marker.points = [pt1, pt2]
                    marker_array_msg.markers.append(marker)

            if DEBUG:
                self.tag_pt_pub.publish(tag_pts)
                self.corner_pt_pub.publish(corner_pts)
                self.marker_arr_pub.publish(marker_array_msg)

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

    def composeTfMsg(self, trans, q, header, tag_id):
        output_tf = TransformStamped()

        # Update header info
        output_tf.header = header
        output_tf.child_frame_id = TAG_PREFIX + str(tag_id)

        # Populate translation info
        output_tf.transform.translation = Vector3(*trans)

        # Populate rotation info
        # q = tf_conversions.transformations.quaternion_from_euler(*rot)
        output_tf.transform.rotation = Quaternion(*q)

        print(output_tf)
        return output_tf

if __name__ == '__main__':
    node = ApriltagsRgbdNode()
    node.run()
