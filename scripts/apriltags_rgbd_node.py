#!/usr/bin/env python3
# Author: Amy Phung

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
from geometry_msgs.msg import TransformStamped, Vector3, Quaternion, Pose, Point32
from std_msgs.msg import Bool
from sensor_msgs.msg import Image, CameraInfo, PointCloud
from apriltag_msgs.msg import ApriltagArrayStamped
from message_filters import TimeSynchronizer, Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge, CvBridgeError
import tf_conversions
from tf.transformations import quaternion_from_matrix
from visualization_msgs.msg import MarkerArray, Marker

# For filtering
ENABLE_FILTER = True
from tag_detection_filter import TagDetectionFilter

# TODO: Remove hardcode
TAG_PREFIX = ""#"detected_"

class ApriltagsRgbdNode():
    def __init__(self):
        rospy.init_node("apriltags_rgbd")
        self.rate = rospy.Rate(60) # 10 Hz

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
        self.k_mtx = None

        self.new_data = False

        # ROS tf
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Point filtering
        if ENABLE_FILTER:
            self.filter = TagDetectionFilter()

        # Publishers
        self.tag_tf_pub = rospy.Publisher("/apriltags_rgbd/tag_tfs", TransformStamped, queue_size=10)
        self.tag_pt_pub = rospy.Publisher("/apriltags_rgbd/extracted_tag_pts", PointCloud, queue_size=10)
        self.corner_pt_pub = rospy.Publisher("/apriltags_rgbd/corner_tag_pts", PointCloud, queue_size=10)
        self.marker_arr_pub = rospy.Publisher("/apriltags_rgbd/visualization_marker_array", MarkerArray, queue_size=10)

    def tagCallback(self, camera_info_data, rgb_data, depth_data, tag_data):
        self.camera_info_data = camera_info_data
        self.rgb_data = rgb_data
        self.depth_data = depth_data
        self.tag_data = tag_data
        self.new_data = True
        # TODO: check timestamps here

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
            self.k_mtx = np.array(self.camera_info_data.K).reshape(3,3)

            # Create messages for point info
            tag_pts = PointCloud()
            corner_pts = PointCloud()
            marker_array_msg = MarkerArray()

            # Estimate pose of each tag
            for tag in self.tag_data.apriltags:
                tag_id, image_pts = self.parseTag(tag)

                # Fit plane and compute corner positions
                depth_plane_est, all_pts = fuse.sample_depth_plane(depth_image, image_pts, self.k_mtx)
                depth_points = fuse.getDepthPoints(image_pts, depth_plane_est, depth_image, self.k_mtx)

                # Plane Normal Vector
                n_vec = -depth_plane_est.mean.n # Negative because plane points
                                                # in opposite direction

                if ENABLE_FILTER:
                    # Filter points
                    depth_points, n_vec = self.filter.updateEstimate(tag_id, depth_points, n_vec)

                    # Check results from filter
                    if depth_points == None or np.isnan(np.sum(n_vec)):
                        continue

                # Compute center of plane
                center_pt = np.mean(depth_points, axis=0)

                # Compute magnitude of normal
                n_norm = np.linalg.norm(n_vec)

                # Compute point in direction of x-axis
                x_pt = (depth_points[1] + depth_points[2]) / 2

                # Compute first orthogonal vector - project point onto plane
                u = x_pt - center_pt
                v = n_vec
                v_norm = n_norm
                x_vec = u - (np.dot(u, v)/v_norm**2)*v

                # Compute second orthogonal vector - take cross product
                y_vec = np.cross(n_vec, x_vec)

                # Normalize vectors
                x_vec = x_vec / np.linalg.norm(x_vec)
                y_vec = y_vec / np.linalg.norm(y_vec)
                n_vec = n_vec / np.linalg.norm(n_vec)

                # TODO: Optimize this
                x_vec1 = list(x_vec) + [0]
                y_vec1 = list(y_vec) + [0]
                n_vec1 = list(n_vec) + [0]

                R_t = np.array([x_vec1,y_vec1,n_vec1,[0,0,0,1]])

                quat = quaternion_from_matrix(R_t.transpose()) # [q_x, q_y, q_z, q_w]

                # Normalize quaternion
                quat = quat / np.linalg.norm(quat)

                # Update tf tree
                output_tf = self.composeTfMsg(center_pt, quat, header, tag_id)
                self.tf_broadcaster.sendTransform(output_tf)
                self.tag_tf_pub.publish(output_tf)

                # Save tag and corner points
                tag_pts.header = header
                corner_pts.header = header

                for i in range(len(all_pts)):
                    tag_pts.points.append(Point32(*all_pts[i]))

                for i in range(len(depth_points)):
                    corner_pts.points.append(Point32(*depth_points[i]))

                # Create visualization markers
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

            # Publish data
            self.tag_pt_pub.publish(tag_pts)
            self.corner_pt_pub.publish(corner_pts)
            self.marker_arr_pub.publish(marker_array_msg)

            self.rate.sleep()

    def parseTag(self, tag):
        id = tag.id

        im_pt0 = [tag.corners[0].x, tag.corners[0].y]
        im_pt1 = [tag.corners[1].x, tag.corners[1].y]
        im_pt2 = [tag.corners[2].x, tag.corners[2].y]
        im_pt3 = [tag.corners[3].x, tag.corners[3].y]

        im_pts = im_pt0 + im_pt1 + im_pt2 + im_pt3
        image_pts = np.array(im_pts).reshape(4,2)

        return id, image_pts

    def composeTfMsg(self, trans, q, header, tag_id):
        output_tf = TransformStamped()

        # Update header info
        output_tf.header = header
        output_tf.child_frame_id = TAG_PREFIX + str(tag_id)

        # Populate translation info
        output_tf.transform.translation = Vector3(*trans)

        # Populate rotation info
        output_tf.transform.rotation = Quaternion(*q)

        return output_tf

if __name__ == '__main__':
    node = ApriltagsRgbdNode()
    node.run()
