#!/usr/bin/env python3
# Author: Amy Phung

# Tell python where to find utils code
import sys
import os
fpath = os.path.join(os.path.dirname(__file__), "utils")
sys.path.append(fpath)

# Python Imports
import numpy as np
import cv2
import math
import time

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
import tf.transformations as tr
from visualization_msgs.msg import MarkerArray, Marker

# Custom messages
from apriltags_rgbd.msg import PointArray, LabeledPointArray

# Custom Imports
import tf_utils
import icp

TF_TIMEOUT = 2 # Maximum age of tfs to use in estimate (in seconds)

class BodyPoseEstimationNode():
    def __init__(self):
        rospy.init_node("body_pose_estimator")
        self.rate = rospy.Rate(10)

        cfg_param = rospy.get_param("~apriltags_rbgd_config")
        self.cfg = self.parseConfig(cfg_param)
        if not self.cfg:
            sys.exit("Invalid configuration")

        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Subscribers
        self.tag_sub = rospy.Subscriber("/apriltags_rgbd/tag_tfs", TransformStamped, self.tagCallback)
        self.corner_sub = rospy.Subscriber("/apriltags_rgbd/corner_tag_pts_labeled", LabeledPointArray, self.cornerCallback)

        # Publishers (visualization only)
        self.body_pts_pub =  rospy.Publisher("/apriltags_rgbd/icp_body_pts", PointCloud, queue_size=10)
        self.detected_pts_pub =  rospy.Publisher("/apriltags_rgbd/icp_detected_pts", PointCloud, queue_size=10)

        # Private vars
        self.frame_id = ""
        self.bodies = list(set(self.cfg['bodies']))
        self.body_pts = []
        self.prev_ts = []
        self.tag_corner_info = None

        for b in self.bodies:
            # Compute points in each body for ICP
            self.body_pts.append(self.computeBodyPts(b))
            # Initialize ts to 0
            self.prev_ts.append(0)

    def parseConfig(self, cfg):
        formatted_cfg = {
            "tags": [],
            "sizes": [],
            "bodies": [],
            "timestamps": [],
            "tf_camera2tag": [],
            "tf_tag2body": []
        }
        for body in cfg['bodies']:
            name = list(body.keys())[0]
            if name in formatted_cfg["bodies"]:
                rospy.logerror("Duplicate body " + name + "found")
                return False
            for tag in body[name]['tags']:
                if tag['id'] in formatted_cfg["tags"]:
                    rospy.logerror("Duplicate tag ID " + str(tag['id']) + "found")
                    return False

                formatted_cfg["tags"].append(str(tag['id']))
                formatted_cfg["sizes"].append(tag['size'])
                formatted_cfg["bodies"].append(name)

                # Position of body to tag
                p_b_t = [tag['pose']['position']['x'],
                         tag['pose']['position']['y'],
                         tag['pose']['position']['z']]

                # Rotation from body to tag
                a_b_t = [tag['pose']['rotation']['x'],
                         tag['pose']['rotation']['y'],
                         tag['pose']['rotation']['z']]

                # Compute transformation between tag to body
                X_b_t = tr.compose_matrix(angles=a_b_t, translate=p_b_t)
                X_t_b = tr.inverse_matrix(X_b_t)
                formatted_cfg["tf_tag2body"].append(X_t_b)

                # Add placeholders for vars
                formatted_cfg['tf_camera2tag'].append(None)
                formatted_cfg['timestamps'].append(0.0)

        # Use np array for compute efficiency
        formatted_cfg['tags'] = np.array(formatted_cfg['tags'])
        formatted_cfg['bodies'] = np.array(formatted_cfg['bodies'])
        formatted_cfg['timestamps'] = np.array(formatted_cfg['timestamps'])
        formatted_cfg['tf_camera2tag'] = np.array(formatted_cfg['tf_camera2tag'])
        formatted_cfg['tf_tag2body'] = np.array(formatted_cfg['tf_tag2body'])

        return formatted_cfg

    def getCornerPoints(self, tag_size, use_3d=True):
        """Compute position for apriltag corners in tag frame

        @param:
        - tag_size: width of tag in meters
        - use_3d: whether to return 2d or 3d points. 3d points will set
        z value to 0
        @return corners: list of corners in 0,1,2,3 order according to this
        layout:

           +y
            ^
            |
        3___|___2
        |       |
        |       | ----> +x
        |_______|
        0       1
        """
        tag_radius = tag_size / 2

        if use_3d:
            corners = [[-tag_radius, -tag_radius, 0.0],
                       [ tag_radius, -tag_radius, 0.0],
                       [ tag_radius,  tag_radius, 0.0],
                       [-tag_radius,  tag_radius, 0.0]]
        else:
            corners = [[-tag_radius, -tag_radius],
                       [ tag_radius, -tag_radius],
                       [ tag_radius,  tag_radius],
                       [-tag_radius,  tag_radius]]
        return corners

    def computeBodyPts(self, body):
        """Compute theoretical 3d points of apriltag corners in body frame

        @param body: name of body to compute points for
        @return pts: points of corners for all apriltags in body based on
        config, as a numpy array
        """
        # Initialize output
        pts = []

        # Get indices of tags relevant to current body
        idxs = np.argwhere(self.cfg['bodies'] == body).T[0]

        # Compute position of each corner in each tag relative to body
        for idx in idxs:
            corners = self.getCornerPoints(self.cfg['sizes'][idx])
            X_b_t = tr.inverse_matrix(self.cfg['tf_tag2body'][idx])
            for c in corners:
                 X_t_p = tr.translation_matrix(c)
                 X_b_p = tr.concatenate_matrices(X_b_t, X_t_p)
                 _, _, _, pos, _ = tr.decompose_matrix(X_b_p)
                 pts.append(pos)
        return np.array(pts)

    def tagCallback(self, tf_data):
        # Extract position and quaternion for camera to tag
        p_c_t, q_c_t = tf_utils.transform_to_pq(tf_data.transform)
        a_c_t = tr.euler_from_quaternion(q_c_t)

        # Compute transformation between camera to tag
        X_c_t = tr.compose_matrix(angles=a_c_t, translate=p_c_t)

        # Save transform
        if tf_data.child_frame_id not in self.cfg['tags']:
            rospy.loginfo("Unused tf " + tf_data.child_frame_id)
            return

        idx = np.where(self.cfg['tags'] == tf_data.child_frame_id)[0][0]
        self.cfg['tf_camera2tag'][idx] = X_c_t
        self.cfg['timestamps'][idx] = tf_data.header.stamp.to_sec()

        # TODO: make this more robust
        self.frame_id = tf_data.header.frame_id

    def cornerCallback(self, corner_data):
        # Save corner data
        self.tag_corner_info = corner_data

    def computeInitialPose(self, body, time_mask):
        """Compute initial pose to use for ICP - use body pose estimate
        based on first body tag (chosen arbitrarily)
        """
        # Create bool array for body
        b_arr_body = [self.cfg['bodies'] == body]

        # Get tags with valid timestamps for this body
        b_arr = np.logical_and(time_mask, b_arr_body)[0]

        # Use first index
        idx = np.where(b_arr)[0][0]

        # Get tag info
        X_c_t = self.cfg['tf_camera2tag'][idx]
        X_t_b = self.cfg['tf_tag2body'][idx]
        tag_ts = self.cfg['timestamps'][idx]
        tag_id = self.cfg['tags'][idx]

        # Compute estimate of camera to body tf
        X_c_b = tr.concatenate_matrices(X_c_t, X_t_b)

        return X_c_b

    def extractBodyPoints(self, body):
        # Get list of corner points for body
        body_idx = self.bodies.index(body)
        body_pts = self.body_pts[body_idx]
        return body_pts

    def extractDetectedPoints(self, body, time_mask):
        # Initialize corner points array for body
        detected_pts = []

        # Create bool array for body
        b_arr_body = [self.cfg['bodies'] == body]

        # Get tags with valid timestamps for this body
        b_arr = np.logical_and(time_mask, b_arr_body)[0]
        idxs = np.where(b_arr)[0]

        # Get list of detected corner points
        for idx in idxs:
            tag_id = self.cfg['tags'][idx]
            if tag_id not in self.tag_corner_info.labels:
                rospy.logwarn("No corner info for tag " + tag_id)
                continue
            c_idx = self.tag_corner_info.labels.index(tag_id)
            c_pts = self.tag_corner_info.point_arrays[c_idx].points

            # Add points to array
            for c in c_pts:
                pt = tf_utils.point_to_p(c)
                detected_pts.append(pt)

        detected_pts = np.array(detected_pts)
        return detected_pts

    def constructOutputMsg(self, body, se3):
        output_msg = TransformStamped()

        # Use most recent timestamp for body
        ts = max(self.cfg['timestamps'][self.cfg['bodies'] == body])
        output_msg.header.stamp = rospy.Time.from_sec(ts)
        output_msg.header.frame_id = self.frame_id
        output_msg.child_frame_id = "detected_" + body

        output_msg.transform = tf_utils.se3_to_msg(se3)
        return output_msg

    def checkNewData(self, body):
        ts = max(self.cfg['timestamps'][self.cfg['bodies'] == body])
        b_idx = self.bodies.index(body)
        if self.prev_ts[b_idx] == ts:
            return False
        else:
            # Update ts
            self.prev_ts[b_idx] = ts
            return True

    def run(self):
        while not rospy.is_shutdown():
            if self.tag_corner_info == None:
                continue
            # Create bool array of valid timestamps
            b_arr_time = [rospy.Time.now().to_sec() - self.cfg['timestamps'] < TF_TIMEOUT]

            # Get bodies with valid timestamps
            bodies_list = self.cfg['bodies'][tuple(b_arr_time)]
            bodies = set(bodies_list)

            # Estimate tf of each body
            for body in bodies:
                # Check if we have new data
                if not self.checkNewData(body):
                    continue

                initial_pose = self.computeInitialPose(body, b_arr_time)
                detected_pts = self.extractDetectedPoints(body, b_arr_time)
                body_pts = self.extractBodyPoints(body)

                # Ensure we have valid data
                if len(detected_pts) == 0:
                    rospy.logwarn("No usable corner detections")
                    continue

                # Run ICP to compute transform from camera to body
                # TODO: use initial pose to help ensure we don't get the mirror
                X_c_b, distances, iterations = icp.icp(body_pts, detected_pts,
                    init_pose=initial_pose, max_iterations=100, tolerance=0.000000001)

                # Publish to ROS
                tf_msg = self.constructOutputMsg(body, X_c_b)
                self.tf_broadcaster.sendTransform(tf_msg)


                # Publish body and detected points for visualization
                body_pts_msg = PointCloud()
                body_pts_msg.header = tf_msg.header
                detected_pts_msg = PointCloud()
                detected_pts_msg.header = tf_msg.header

                for i in range(len(body_pts)):
                    body_pts_msg.points.append(Point32(*body_pts[i]))

                for i in range(len(detected_pts)):
                    detected_pts_msg.points.append(Point32(*detected_pts[i]))

                self.body_pts_pub.publish(body_pts_msg)
                self.detected_pts_pub.publish(detected_pts_msg)

                #         self.cfg['timestamps'][idx] = tf_data.header.stamp.to_sec()
                #
                #         self.frame_id
                # print(T)



                #
                # # TODO: Compute icp initial estimate
                # print(b_arr)
                # body_tags = self.cfg['tags'][b_arr]
                # # print("here")
                # # print(body_tags)
                # # print(self.tag_corner_info.labels)
                #
                # for tag in body_tags:
                #     if tag in self.tag_corner_info.labels:
                #         tag

                # for idx in idxs:
                #     # Get tag info
                #     X_c_t = self.cfg['tf_camera2tag'][idx]
                #     X_t_b = self.cfg['tf_tag2body'][idx]
                #     tag_ts = self.cfg['timestamps'][idx]
                #     tag_id = self.cfg['tags'][idx]
                #
                #     # Compute position of corner points in camera frame
                #     # X_t_p = translation_matrix((1, 2, 3)) # points in tag frame
                #     #
                #     # X_c_t*X_t_p
                #
                #     # TODO: need to add error checking for initial messages
                #
                #     print(np.array(self.tag_corner_info.labels) == '4')
                #             # print(msg.header.stamp.to_sec())
                #             # print(rospy.Time.now().to_sec())
                #             # print(X_c_t)
                #             # self.cfg
                #             # self.cfg
                #             # print(p)
                #
                #             # trans1_mat = tft.translation_matrix(msg.transform.translation)
                #             # print(trans1_mat)
                #             # print(q)
                #
                #
                #     # Compute estimate of camera to body tf
                #     X_c_b = tr.concatenate_matrices(X_c_t, X_t_b)
                #
                #     # Publish for visualization
                #     msg = tf_utils.se3_to_msg(X_c_b)
                #     tf_msg = TransformStamped()
                #     tf_msg.header.frame_id = self.frame_id
                #     tf_msg.header.stamp = rospy.Time.from_sec(tag_ts)
                #     tf_msg.child_frame_id = body + "_" + str(tag_id)
                #     tf_msg.transform = msg
                #
                #     self.tf_broadcaster.sendTransform(tf_msg)


            # print(bodies)



                     # print(self.cfg['tags'].shape)
                     # print(self.cfg['bodies'].shape)
                     # print(self.cfg['timestamps'].shape)
                     # print(self.cfg['tf_camera2tag'].shape)
                     # print(self.cfg['tf_tag2body'].shape)
                     #
                     # print(self.cfg['tf_tag2body'][0])
                     # print(idx)


            # Multiply X_c_t X_t_b to get X_c_b
            #
            # Record X_c_b
            # Take average of X_c_b values between tags


            self.rate.sleep()

if __name__=="__main__":
    est = BodyPoseEstimationNode()
    est.run()
