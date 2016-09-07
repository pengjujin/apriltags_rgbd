#!/usr/bin/env python
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class image_capture:
	def __init__(self):
			self.bridge = CvBridge()
			self.image_rgb_sub = rospy.Subscriber("/head/kinect2/qhd/image_color_rect", Image, self.rgb_callback)
			self.image_depth_sub = rospy.Subscriber("/head/kinect2/qhd/image_depth_rect", Image, self.depth_callback)

	def rgb_callback(self, data):
		try:
			rgb_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)
		cv2.imwrite("rgb_frame.png", rgb_image)
	
	def depth_callback(self, data):
		try:
			depth_image = self.bridge.imgmsg_to_cv2(data, "16UC1")
		except CvBridgeError as e:
			print(e)
		cv2.imwrite("depth_frame.png", depth_image)

def main(args):
	ic = image_capture()
	rospy.init_node('image capturer', anonymous=True)
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")

if __name__ == '__main__':
	main(sys.argv)