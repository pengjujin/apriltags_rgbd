#!/usr/bin/env python
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from apriltags.msg import AprilTagDetections
from message_filters import TimeSynchronizer, Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge, CvBridgeError

class image_capture:
	def __init__(self):
			self.bridge = CvBridge()
			# self.image_rgb_sub = rospy.Subscriber("/head/kinect2/qhd/image_color_rect", Image, self.rgb_callback)
			# self.image_depth_sub = rospy.Subscriber("/head/kinect2/qhd/image_depth_rect", Image, self.depth_callback)
			# self.apriltag_sub = rospy.Subscriber("/apriltags_kinect2/detections", AprilTagDetections, self.tag_callback)
			tss = ApproximateTimeSynchronizer([Subscriber("/head/kinect2/qhd/image_color_rect", Image),
											Subscriber("/head/kinect2/qhd/image_depth_rect", Image),
											Subscriber("/apriltags_kinect2/detections", AprilTagDetections)], 1 ,0.5)
			tss.registerCallback(self.processtag_callback)

			self.tag_filepath = '../data/iros_data/apriltag_info_%04d.txt'
			self.rgb_filepath = '../data/iros_data/rgb_frame%04d.png'
			self.depth_filepath = '../data/iros_data/depth_frame%04d.png'
			self.counter = 1

	def processtag_callback(self, rgb_data, depth_data, tag_data):
		try:
			rgb_image = self.bridge.imgmsg_to_cv2(rgb_data, "bgr8")
			depth_image = self.bridge.imgmsg_to_cv2(depth_data, "16UC1")
		except CvBridgeError as e:
			print(e)

		all_detections = tag_data.detections
		allstring = ''
		for current_detection in all_detections:
			detection_string = self.format_AprilTagDetections(current_detection)
			allstring = allstring + detection_string
		self.rgb_image = rgb_image
		self.depth_image = depth_image
		self.detection_string = allstring
		cv2.imshow('Image', rgb_image)
		key = cv2.waitKey(1)
		if (key > -1):
			self.key_press()

	def key_press(self):
		print(("saving data %04d" % (self.counter, )))
		cv2.imwrite((self.rgb_filepath % (self.counter,)), self.rgb_image)
		cv2.imwrite((self.depth_filepath % (self.counter,)), self.depth_image)
		temppath = self.tag_filepath % (self.counter, ) 
		with open(temppath, 'w') as f:
			f.write(self.detection_string)
		f.close()
		self.counter = self.counter + 1

	def rgb_callback(self, data):
		try:
			rgb_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
		except CvBridgeError as e:
			print(e)
		cv2.imwrite("../data/rgb_frame.png", rgb_image)
	
	def depth_callback(self, data):
		try:
			depth_image = self.bridge.imgmsg_to_cv2(data, "16UC1")
		except CvBridgeError as e:
			print(e)
		cv2.imwrite("../data/depth_frame.png", depth_image)

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
		tag_size = data.tag_size
		position_x = data.pose.position.x
		position_y = data.pose.position.y 
		position_z = data.pose.position.z
		rotation_x = data.pose.orientation.x 
		rotation_y = data.pose.orientation.y 
		rotation_z = data.pose.orientation.z 
		rotation_w = data.pose.orientation.w 
		template_string = '''
			detection_id: {0} 
			point1:
			corner_x: {1} 
			corner_y: {2}
			corner_z: {3}
			point2:
			corner_x: {4} 
			corner_y: {5}
			corner_z: {6}
			point3:
			corner_x: {7} 
			corner_y: {8}
			corner_z: {9}
			point4:
			corner_x: {10} 
			corner_y: {11}
			corner_z: {12}

			tag_size: {13}
			position_x: {14}
			position_y: {15}
			position_z: {16}
			rotation_x: {17}
			rotation_y: {18}
			rotation_z: {19}
			rotation_w: {20}
			'''
		output_string = template_string.format(detection_id, corners1_x, corners1_y, corners1_z,
												corners2_x, corners2_y, corners2_z, 
												corners3_x, corners3_y, corners3_z, 
												corners4_x, corners4_y, corners4_z,  
											   tag_size, position_x, position_y, position_z,
											   rotation_x, rotation_y, rotation_z, rotation_w)
		return output_string

	def tag_callback(self, data):
		all_detections = data.detections
		with open(self.tag_filepath, 'w') as f:
			for current_detection in all_detections:
				detection_string = self.format_AprilTagDetections(current_detection)
				f.write(detection_string)
		f.close()

def main(args):
	ic = image_capture()
	rospy.init_node('image_capture', anonymous=True)
	try:
		rospy.spin()
	except KeyboardInterrupt:
		print("Shutting down")

if __name__ == '__main__':
	main(sys.argv)