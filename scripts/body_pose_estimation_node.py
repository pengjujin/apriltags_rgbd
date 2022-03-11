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
import tf.transformations as tr
from visualization_msgs.msg import MarkerArray, Marker

# Custom Imports
import tf_utils

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

        self.tag_tf_sub = rospy.Subscriber("/apriltags_rgbd/tag_tfs", TransformStamped, self.tagTfCallback)

        self.frame_id = ""

    def tagTfCallback(self, msg):
        # Extract position and quaternion for camera to tag
        p_c_t, q_c_t = tf_utils.transform_to_pq(msg.transform)
        a_c_t = tr.euler_from_quaternion(q_c_t)

        # Compute transformation between camera to tag
        X_c_t = tr.compose_matrix(angles=a_c_t, translate=p_c_t)

        # Save transform
        if int(msg.child_frame_id) not in self.cfg['tags']:
            rospy.loginfo("Unused tf " + msg.child_frame_id)
            return

        idx = np.where(self.cfg['tags'] == int(msg.child_frame_id))[0][0]
        self.cfg['tf_camera2tag'][idx] = X_c_t
        self.cfg['timestamps'][idx] = msg.header.stamp.to_sec()


        # TODO: make this more robust
        self.frame_id = msg.header.frame_id

        # print(msg.header.stamp.to_sec())
        # print(rospy.Time.now().to_sec())
        # print(X_c_t)
        # self.cfg
        # self.cfg
        # print(p)

        # trans1_mat = tft.translation_matrix(msg.transform.translation)
        # print(trans1_mat)
        # print(q)

    def parseConfig(self, cfg):
        formatted_cfg = {
            "tags": [],
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
            print(name)
            for tag in body[name]['tags']:
                if tag['id'] in formatted_cfg["tags"]:
                    rospy.logerror("Duplicate tag ID " + str(tag['id']) + "found")
                    return False

                formatted_cfg["tags"].append(tag['id'])
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
                formatted_cfg['timestamps'].append(0)

        # Use np array for compute efficiency
        formatted_cfg['tags'] = np.array(formatted_cfg['tags'])
        formatted_cfg['bodies'] = np.array(formatted_cfg['bodies'])
        formatted_cfg['timestamps'] = np.array(formatted_cfg['timestamps'])
        formatted_cfg['tf_camera2tag'] = np.array(formatted_cfg['tf_camera2tag'])
        formatted_cfg['tf_tag2body'] = np.array(formatted_cfg['tf_tag2body'])

        return formatted_cfg


    def run(self):
        while not rospy.is_shutdown():
            # Create bool array of valid timestamps
            b_arr_time = [rospy.Time.now().to_sec() - self.cfg['timestamps'] < TF_TIMEOUT]

            # Get bodies with valid timestamps
            bodies_list = self.cfg['bodies'][b_arr_time]
            bodies = set(bodies_list)

            # Estimate tf of each body
            for body in bodies:
                # Create bool array for body
                b_arr_body = [self.cfg['bodies'] == body]

                # Get idxs with valid timestamps for this body
                b_arr = np.logical_and(b_arr_time, b_arr_body)
                idxs = np.where(b_arr)[1]

                for idx in idxs:
                    # Get tag info
                    X_c_t = self.cfg['tf_camera2tag'][idx]
                    X_t_b = self.cfg['tf_tag2body'][idx]
                    tag_ts = self.cfg['timestamps'][idx]
                    tag_id = self.cfg['tags'][idx]

                    # Compute estimate of camera to body tf
                    X_c_b = tr.concatenate_matrices(X_c_t, X_t_b)

                    # Publish for visualization
                    msg = tf_utils.se3_to_msg(X_c_b)
                    tf_msg = TransformStamped()
                    tf_msg.header.frame_id = self.frame_id
                    tf_msg.header.stamp = rospy.Time.from_sec(tag_ts)
                    tf_msg.child_frame_id = body + "_" + str(tag_id)
                    tf_msg.transform = msg

                    self.tf_broadcaster.sendTransform(tf_msg)
                    print(msg)


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
                # print("a")
                # blocking

if __name__=="__main__":
    est = BodyPoseEstimationNode()
    est.run()

#         (trans1, rot1) = tf.lookupTransform(l2, l1, t)
# trans1_mat = tf.transformations.translation_matrix(trans1)
# rot1_mat   = tf.transformations.quaternion_matrix(rot1)
# mat1 = numpy.dot(trans1_mat, rot1_mat)
#
# (trans2, rot2) = tf.lookupTransform(l4, l3, t)
# trans2_mat = tf.transformations.translation_matrix(trans2)
# rot2_mat    = tf.transformations.quaternion_matrix(rot2)
# mat2 = numpy.dot(trans2_mat, rot2_mat)
#
# mat3 = numpy.dot(mat1, mat2)
# trans3 = tf.transformations.translation_from_matrix(mat3)
# rot3 = tf.transformations.quaternion_from_matrix(mat3)
#
# br = tf.TransformBroadcaster()
# br.sendTransform(
#   trans3,
#   rot3,
#   t,
#   "target",
#   "source");
#
# # TODO: move yaml parsing to utils
# # TODO: remove hardcoded tag size
#
# # Tell python where to find apriltags_rgbd code
# import sys
# import os
# fpath = os.path.join(os.path.dirname(__file__), "apriltags_rgbd")
# sys.path.append(fpath)
#
# # Python Imports
# import numpy as np
# import cv2
# import rgb_depth_fuse as fuse
# import math
#
# # ROS Imports
# import rospy
# import tf2_ros
# from geometry_msgs.msg import TransformStamped, Vector3, Quaternion, Pose
# from std_msgs.msg import Bool
# from sensor_msgs.msg import Image, CameraInfo
# from apriltag_msgs.msg import ApriltagArrayStamped
# from message_filters import TimeSynchronizer, Subscriber, ApproximateTimeSynchronizer
# from cv_bridge import CvBridge, CvBridgeError
# import tf_conversions
# from tf.transformations import quaternion_from_matrix
# from visualization_msgs.msg import MarkerArray, Marker
#
#
# # For debugging
# DEBUG = True
# from sensor_msgs.msg import PointCloud
# from geometry_msgs.msg import Point32
#
#
# # TODO: Remove hardcode
# TAG_PREFIX = "detected_"
#
# class ApriltagsRgbdNode():
#     def __init__(self):
#         rospy.init_node("apriltags_rgbd")
#         self.rate = rospy.Rate(10) # 10 Hz
#
#         # CV bridge
#         self.bridge = CvBridge()
#
#         # Subscribers
#         tss = ApproximateTimeSynchronizer([
#             Subscriber("/kinect2/hd/camera_info", CameraInfo),
#             Subscriber("/kinect2/hd/image_color_rect", Image),
#             Subscriber("/kinect2/hd/image_depth_rect", Image),
#             Subscriber("/kinect2/hd/tags", ApriltagArrayStamped)], 1 ,0.5)
#         tss.registerCallback(self.tagCallback)
#
#         self.camera_info_data = None
#         self.rgb_data = None
#         self.depth_data = None
#         self.tag_data = None
#         self.k_mtx = None
#
#         self.new_data = False
#
#         # ROS tf
#         self.tf_buffer = tf2_ros.Buffer()
#         self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
#         self.tf_broadcaster = tf2_ros.TransformBroadcaster()
#
#         # Publishers
#         if DEBUG:
#             self.tag_pt_pub = rospy.Publisher("extracted_tag_pts", PointCloud, queue_size=10)
#             self.corner_pt_pub = rospy.Publisher("corner_tag_pts", PointCloud, queue_size=10)
#             self.marker_arr_pub = rospy.Publisher("visualization_marker_array", MarkerArray, queue_size=10)
#
#     def tagCallback(self, camera_info_data, rgb_data, depth_data, tag_data):
#         self.camera_info_data = camera_info_data
#         self.rgb_data = rgb_data
#         self.depth_data = depth_data
#         self.tag_data = tag_data
#         self.new_data = True
#
#         # print("here")
#         # # print(self.camera_info)
#         # print(camera_info_data)
#         # try:
#         #     rgb_image = self.bridge.imgmsg_to_cv2(rgb_data, "bgr8")
#         #     depth_image = self.bridge.imgmsg_to_cv2(depth_data, "16UC1")
#         # except CvBridgeError as e:
#         #     print(e)
#         #
#         # all_detections = tag_data.detections
#         # # allstring = ''
#         # # for current_detection in all_detections:
#         # #     detection_string = self.format_AprilTagDetections(current_detection)
#         # #     allstring = allstring + detection_string
#         # self.rgb_image = rgb_image
#         # self.depth_image = depth_image
#         # # self.detection_string = allstring
#         # cv2.imshow('Image', rgb_image)
#         # key = cv2.waitKey(1)
#         # if (key > -1):
#         #     self.key_press()
#
#     # def parseConfig(self, cfg_param):
#     #     config = {}
#     #     config["ids"] = [1,2,3,4]
#     #
#     #     return config
#
#     def run(self):
#         while not rospy.is_shutdown():
#             # Check for new data
#             if not self.new_data:
#                 self.rate.sleep()
#                 continue
#
#             self.new_data = False
#
#             # Convert ROS images to OpenCV frames
#             try:
#                 rgb_image = self.bridge.imgmsg_to_cv2(self.rgb_data, "bgr8")
#                 depth_image = self.bridge.imgmsg_to_cv2(self.depth_data, "16UC1")
#             except CvBridgeError as e:
#                 print(e)
#                 continue
#
#             # Extract metadata
#             header = self.camera_info_data.header
#             self.k_mtx = np.array(self.camera_info_data.K).reshape(3,3)
#
#             if DEBUG:
#                 tag_pts = PointCloud()
#                 corner_pts = PointCloud()
#                 marker_array_msg = MarkerArray()
#
#             # Estimate pose of each tag
#             for tag in self.tag_data.apriltags:
#                 tag_id, image_pts = self.parseTag(tag)
#
#                 # Estimate pose
#                 depth_plane_est, all_pts = fuse.sample_depth_plane(depth_image, image_pts, self.k_mtx)
#                 depth_points = fuse.getDepthPoints(image_pts, depth_plane_est, depth_image, self.k_mtx)
#
#                 # Compute center of plane
#                 center_pt = np.mean(depth_points, axis=0)
#
#                 # Plane Normal Vector
#                 n_vec = -depth_plane_est.mean.n # Negative because plane points
#                                                 # in opposite direction
#                 n_norm = np.linalg.norm(n_vec)
#
#                 # Compute point in direction of x-axis
#                 x_pt = (depth_points[1] + depth_points[2]) / 2
#
#                 # Compute first orthogonal vector - project point onto plane
#                 u = x_pt - center_pt
#                 v = n_vec
#                 v_norm = n_norm
#                 x_vec = u - (np.dot(u, v)/v_norm**2)*v
#
#                 # Compute second orthogonal vector - take cross product
#                 y_vec = np.cross(n_vec, x_vec)
#
#                 # Normalize vectors
#                 x_vec = x_vec / np.linalg.norm(x_vec)
#                 y_vec = y_vec / np.linalg.norm(y_vec)
#                 n_vec = n_vec / np.linalg.norm(n_vec)
#
#                 # TODO: this can be optimized
#                 x_vec1 = list(x_vec) + [0]
#                 y_vec1 = list(y_vec) + [0]
#                 n_vec1 = list(n_vec) + [0]
#
#                 R_t = np.array([x_vec1,y_vec1,n_vec1,[0,0,0,1]])
#
#                 quat = quaternion_from_matrix(R_t.transpose()) # [q_x, q_y, q_z, q_w]
#
#                 # Normalize quaternion
#                 quat = quat / np.linalg.norm(quat)
#
#                 # Update tf tree
#                 output_tf = self.composeTfMsg(center_pt, quat, header, tag_id)
#                 self.tf_broadcaster.sendTransform(output_tf)
#
#                 if DEBUG:
#                     tag_pts.header = header
#                     corner_pts.header = header
#
#                     for i in range(len(all_pts)):
#                         tag_pts.points.append(Point32(*all_pts[i]))
#
#                     for i in range(len(depth_points)):
#                         corner_pts.points.append(Point32(*depth_points[i]))
#
#                     marker = Marker()
#                     marker.header = header
#                     marker.id = tag_id
#                     marker.type = 0
#                     marker.pose.position = Point32(0,0,0)
#                     marker.pose.orientation = Quaternion(0,0,0,1)
#                     marker.color.r = 1.0
#                     marker.color.g = 0.0
#                     marker.color.b = 0.0
#                     marker.color.a = 1.0
#                     marker.scale.x = 0.005
#                     marker.scale.y = 0.01
#                     marker.scale.z = 0.0
#                     pt1 = Point32(*center_pt)
#                     pt2 = Point32(*(center_pt + n_vec/5))
#                     marker.points = [pt1, pt2]
#                     marker_array_msg.markers.append(marker)
#
#             if DEBUG:
#                 self.tag_pt_pub.publish(tag_pts)
#                 self.corner_pt_pub.publish(corner_pts)
#                 self.marker_arr_pub.publish(marker_array_msg)
#
#             self.rate.sleep()
#
#     def parseTag(self, tag):
#         id = tag.id
#
#         im_pt0 = [tag.corners[0].x, tag.corners[0].y]
#         im_pt1 = [tag.corners[1].x, tag.corners[1].y]
#         im_pt2 = [tag.corners[2].x, tag.corners[2].y]
#         im_pt3 = [tag.corners[3].x, tag.corners[3].y]
#
#         im_pts = im_pt0 + im_pt1 + im_pt2 + im_pt3
#         image_pts = np.array(im_pts).reshape(4,2)
#
#         return id, image_pts
#
#     def composeTfMsg(self, trans, q, header, tag_id):
#         output_tf = TransformStamped()
#
#         # Update header info
#         output_tf.header = header
#         output_tf.child_frame_id = TAG_PREFIX + str(tag_id)
#
#         # Populate translation info
#         output_tf.transform.translation = Vector3(*trans)
#
#         # Populate rotation info
#         output_tf.transform.rotation = Quaternion(*q)
#
#         return output_tf
#
# if __name__ == '__main__':
#     node = ApriltagsRgbdNode()
#     node.run()
