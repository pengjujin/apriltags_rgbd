#!/usr/bin/env python3
# Author: Amy Phung

# Tell python where to find apriltags_rgbd and utils code
import sys
import os
fpath = os.path.join(os.path.dirname(__file__), "apriltags_rgbd")
sys.path.append(fpath)
fpath = os.path.join(os.path.dirname(__file__), "utils")
sys.path.append(fpath)

# Python Imports
import numpy as np
import cv2
import rgb_depth_fuse as fuse
import math
from threading import Thread, Lock
mutex = Lock()

# Custom Imports
import tf_utils

# ROS Imports
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped, Vector3, Quaternion, Pose, Point32
from std_msgs.msg import Bool
from sensor_msgs.msg import Image, CameraInfo, PointCloud
from apriltag_ros.msg import AprilTagDetectionArray
from message_filters import TimeSynchronizer, Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge, CvBridgeError
import tf_conversions
from tf.transformations import quaternion_from_matrix
from visualization_msgs.msg import MarkerArray, Marker

# Custom messages
from apriltags_rgbd.msg import PointArray, LabeledPointArray

# For filtering
ENABLE_FILTER = True
from tag_detection_filter import TagDetectionFilter

# TODO: Remove hardcode
TAG_PREFIX = ""#"detected_"

class ApriltagsRgbdNode():
    def __init__(self):
        rospy.init_node("apriltags_rgbd")
        self.rate = rospy.Rate(60) # 10 Hz

        # Load ROS parameters
        self.rgbd_pos_flag = rospy.get_param("~use_rgbd_position")
        self.rgbd_rot_flag = rospy.get_param("~use_rgbd_rotation")

        # CV bridge
        self.bridge = CvBridge()

        # Subscribers
        tss = ApproximateTimeSynchronizer([
            Subscriber("/kinect2/hd/camera_info", CameraInfo),
            Subscriber("/kinect2/hd/image_color_rect", Image),
            Subscriber("/kinect2/hd/image_depth_rect", Image),
            Subscriber("/tag_detections", AprilTagDetectionArray)], 1 ,0.5)
        tss.registerCallback(self.tagCallback)

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
        self.corner_pt_labeled_pub = rospy.Publisher("/apriltags_rgbd/corner_tag_pts_labeled", LabeledPointArray, queue_size=10)
        self.marker_arr_pub = rospy.Publisher("/apriltags_rgbd/visualization_marker_array", MarkerArray, queue_size=10)

    def tagCallback(self, camera_info_data, rgb_data, depth_data, tag_data):
        with mutex:
            camera_info_data = camera_info_data
            rgb_data = rgb_data
            depth_data = depth_data
            tag_data = tag_data

            # TODO: check timestamps here

            # Convert ROS images to OpenCV frames
            try:
                rgb_image = self.bridge.imgmsg_to_cv2(rgb_data, "bgr8")
                depth_image = self.bridge.imgmsg_to_cv2(depth_data, "16UC1")
            except CvBridgeError as e:
                print(e)
                return

            # Extract metadata
            header = camera_info_data.header
            k_mtx = np.array(camera_info_data.K).reshape(3,3)

            # Create messages for point info
            tag_pts = PointCloud()
            corner_pts = PointCloud()
            marker_array_msg = MarkerArray()
            corner_pts_labeled_array = LabeledPointArray()

            tag_pts.header = header
            corner_pts.header = header
            corner_pts_labeled_array.header = header

            # Estimate pose of each tag
            for tag_idx, tag in enumerate(tag_data.detections):
                tag_id, image_pts = self.parseTag(tag)

                depth_pts = self.extractDepthPoints(depth_image, image_pts, k_mtx)

                # try:
                #     # Fit plane and compute corner positions
                #     depth_plane_est, all_pts = fuse.sample_depth_plane(depth_image, image_pts, k_mtx)
                #     depth_points = fuse.getDepthPoints(image_pts, depth_plane_est, depth_image, k_mtx)
                # except:
                #     rospy.logwarn("Error in plane fitting - skipping tag " + str(tag_id))
                #     continue

                # print(len(all_pts))
                # # Plane Normal Vector
                # n_vec = -depth_plane_est.mean.n # Negative because plane points
                #                                 # in opposite direction

                # # TODO: left off here - this filter results in really clean detections, but is too aggressive
                # if ENABLE_FILTER:
                #     # Filter points
                #     depth_points, n_vec = self.filter.updateEstimate(tag_id, depth_points, n_vec)

                #     # Check results from filter
                #     if depth_points == None:
                #         rospy.loginfo("invalid depth points")
                #         continue
                    
                #     if np.isnan(np.sum(n_vec)):
                #         rospy.loginfo("malformed normal vector")
                #         continue

                # Compute pose
                # center_pt, quat = self.computePoseFromDepth(depth_points, n_vec)


                # TODO: figure out why this piece of code causes jumpy pose estimates
                # # Override pose with detection based on camera if necessary
                # cam_center_pt, cam_quat = self.extractPoseFromMsg(tag)
                # if not self.rgbd_pos_flag:
                #     center_pt = cam_center_pt 
                # if not self.rgbd_rot_flag:
                #     quat = cam_quat

                # Update tf tree
                output_tf = TransformStamped()
                output_tf.header = header
                output_tf.child_frame_id = TAG_PREFIX + str(tag_id)

                # Estimate tag position based on average depth measurement
                output_tf.transform.translation = Vector3(*np.mean(depth_pts, axis=0))

                # Estimate tag orientation based on apriltag detection and camera intrinsics 
                output_tf.transform.rotation = tag.pose.pose.pose.orientation

                self.tf_broadcaster.sendTransform(output_tf)
                self.tag_tf_pub.publish(output_tf)

                # # Save tag and corner points
                # corner_pts_tag = PointArray()

                # for i in range(len(all_pts)):
                #     tag_pts.points.append(Point32(*all_pts[i]))

                # for i in range(len(depth_points)):
                #     corner_pts.points.append(Point32(*depth_points[i]))
                #     corner_pts_tag.points.append(Point32(*depth_points[i]))

                # corner_pts_labeled_array.labels.append(str(tag_id))
                # corner_pts_labeled_array.point_arrays.append(corner_pts_tag)

                # # Create visualization markers
                # marker = Marker()
                # marker.header = header
                # marker.id = int(tag_id)
                # marker.type = 0
                # marker.pose.position = Point32(0,0,0)
                # marker.pose.orientation = Quaternion(0,0,0,1)
                # marker.color.r = 1.0
                # marker.color.g = 0.0
                # marker.color.b = 0.0
                # marker.color.a = 1.0
                # marker.scale.x = 0.005
                # marker.scale.y = 0.01
                # marker.scale.z = 0.0
                # pt1 = Point32(*center_pt)
                # pt2 = Point32(*(center_pt + n_vec/5))
                # marker.points = [pt1, pt2]
                # marker_array_msg.markers.append(marker)

            # Publish data
            # self.tag_pt_pub.publish(tag_pts)
            # self.corner_pt_pub.publish(corner_pts)
            # self.corner_pt_labeled_pub.publish(corner_pts_labeled_array)
            # self.marker_arr_pub.publish(marker_array_msg)

            self.rate.sleep()

    def extractDepthPoints(self, depth_image, image_pts, K):
        ## Generate the depth samples from the depth image
        fx = K[0][0]
        fy = K[1][1]
        px = K[0][2]
        py = K[1][2]
        rows, cols = depth_image.shape
        hull_pts = image_pts.reshape(4,1,2).astype(int)
        rect = cv2.convexHull(hull_pts)
        all_pts = []
        xcoord = image_pts[:, 0]
        ycoord = image_pts[: ,1]
        xmin = int(np.amin(xcoord))
        xmax = int(np.amax(xcoord))
        ymin = int(np.amin(ycoord))
        ymax = int(np.amax(ycoord))
        for j in range(ymin, ymax):
            for i in range(xmin, xmax):
                if (cv2.pointPolygonTest(rect, (i,j), False) > 0):
                    depth = depth_image[j,i] / 1000.0
                    if(depth != 0):
                        x = (i - px) * depth / fx
                        y = (j - py) * depth / fy
                        all_pts.append([x,y,depth])
        samples_depth = np.array(all_pts)
        return samples_depth

    def loop(self):
        self.rate.sleep()

    def computePoseFromDepth(self, depth_points, n_vec):
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
        return center_pt, quat

    def extractPoseFromMsg(self, detection):           
        pose = detection.pose.pose.pose
        center_pt, quat = tf_utils.pose_to_pq(pose)
        return center_pt, quat

    def parseTag(self, tag):
        id = str(tag.id[0])

        im_pt0 = tag.pix_tl
        im_pt1 = tag.pix_tr
        im_pt2 = tag.pix_br
        im_pt3 = tag.pix_bl

        im_pts = im_pt0 + im_pt1 + im_pt2 + im_pt3
        image_pts = np.array(im_pts).reshape(4,2)

        return id, image_pts

    # def composeTfMsg(self, trans, q, header, tag_id):
    #     output_tf = TransformStamped()

    #     # Update header info
    #     output_tf.header = header
    #     output_tf.child_frame_id = TAG_PREFIX + str(tag_id)

    #     # Populate translation info
    #     output_tf.transform.translation = Vector3(*trans)

    #     # Populate rotation info
    #     output_tf.transform.rotation = Quaternion(*q)

    #     return output_tf

if __name__ == '__main__':
    node = ApriltagsRgbdNode()

    while not rospy.is_shutdown():
        with mutex:
            node.loop()
