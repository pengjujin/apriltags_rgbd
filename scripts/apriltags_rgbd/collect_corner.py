#!/usr/bin/env python3
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from apriltags.msg import AprilTagDetections

from cv_bridge import CvBridge, CvBridgeError

class image_capture:
	def __init__(self):
			self.apriltag_sub = rospy.Subscriber("/apriltags_kinect2/detections", AprilTagDetections, self.tag_callback)
			self.dic;
	def format_AprilTagDetections(self, data):
		detection_id = data.id
		tag_corners = data.corners2d
		corners1_x = tag_corners[0].x 
		corners1_y = tag_corners[0].y 
		corners1_z = tag_corners[0].z 
		corners2_x = tag_corners[1].x 
		corners2_y = tag_corners[1].y 
		corners2_z = tag_corners[1].z 
		corners3_x = tag_corners[2].x 
		corners3_y = tag_corners[2].y 
		corners3_z = tag_corners[2].z 
		corners4_x = tag_corners[3].x 
		corners4_y = tag_corners[3].y 
		corners4_z = tag_corners[3].z 
		tag_positions = (corners1_x, corners1_y, corners2_x, corners2_y, corners3_x, corners3_y, corners4_x, corners4_y)
		return tag_positions

	def tag_callback(self, data):
		all_detections = data.detections
		with open(self.tag_filepath, 'w') as f:
			for current_detection in all_detections:
				detection_corners = self.format_AprilTagDetections(current_detection)


def main(args):
	ic = image_capture()
	rospy.init_node('corner_collect', anonymous=True)
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")

if __name__ == '__main__':
	main(sys.argv)