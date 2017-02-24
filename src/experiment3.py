import numpy as np
import cv2
import glob 
import transformation as tf
import math 
import rgb_depth_fuse as fuse 
import LM_minimize as lm 
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import parse
import time


def get_rotation_matrix(rx, ry, rz):
	mrx = [1, 0, 0, 0, np.cos(rx), -np.sin(rx), 0, np.sin(rx), np.cos(rx)]
	mry = [np.cos(ry), 0, np.sin(ry), 0, 1, 0, -np.sin(ry), 0, np.cos(ry)]
	mrz = [np.cos(rz), -np.sin(rz), 0, np.sin(rz), np.cos(rz), 0, 0, 0, 1]

	mrx = np.array(mrx).reshape(3,3)
	mry = np.array(mry).reshape(3,3)
	mrz = np.array(mrz).reshape(3,3)
	rotation = np.dot(np.dot(mrx, mry), mrz)
	return rotation



tag_size = 0.0480000004172
tag_radius = tag_size / 2.0
ob_pt1 = [-tag_radius, -tag_radius, 0.0]
ob_pt2 = [ tag_radius, -tag_radius, 0.0]
ob_pt3 = [ tag_radius,  tag_radius, 0.0]
ob_pt4 = [-tag_radius,  tag_radius, 0.0]
ob_pts = ob_pt1 + ob_pt2 + ob_pt3 + ob_pt4
object_pts = np.array(ob_pts).reshape(4,3)

mtx = np.array([529.2945, 0.0, 466.9604, 0.0, 531.2834, 273.25937, 0, 0, 1]).reshape(3,3) 
dist = np.zeros((5,1))

empty_image = np.zeros((540, 960, 3), np.uint8)

rx = math.radians(0)
ry = math.radians(40)
rz = math.radians(0)
rotation = get_rotation_matrix(rx, ry, rz)
rotation_vec, jacob = cv2.Rodrigues(rotation)
translation = np.array([0, 0, 1.0])
image_pts, jacob = cv2.projectPoints(object_pts, rotation_vec, translation, mtx, dist)
for i in range(4):
	x = int(image_pts[i][0][0])
	y = int(image_pts[i][0][1])
	cv2.circle(empty_image, (x,y), 3, (0,0,255), -1)
image_pts = image_pts.reshape(4,2)
print image_pts

cv2.namedWindow('img', 1)
cv2.imshow('img', empty_image)
cv2.waitKey(0)

def simulate_projection(theta, d):
	rx = math.radians(0)
	ry = math.radians(theta)
	rz = math.radians(0)
	rotation = get_rotation_matrix(rx, ry, rz)
	rotation_vec, jacob = cv2.Rodrigues(rotation)
	translation = np.array([0, 0, d])
	image_pts, jacob = cv2.projectPoints(object_pts, rotation_vec, translation, mtx, dist)
	image_pts = image_pts.reshape(4,2)
	sample_size = 1000
	n = 4
	baseline_diff = []
	for k in range(sample_size):
		modified_img_pts = image_pts
		normal_noise = np.random.normal(0, 0.5, 8).reshape(4,2)
		modified_img_pts = modified_img_pts + normal_noise
		retval, cv2rvec, cv2tvec = cv2.solvePnP(object_pts, modified_img_pts, mtx, dist, flags=cv2.ITERATIVE)
		baseline_rvec_difference = lm.quatAngleDiff(cv2rvec, rotation_vec)
		baseline_tvec_difference = np.linalg.norm(cv2tvec - rotation_vec)
		baseline_diff = baseline_diff + [[baseline_rvec_difference, baseline_tvec_difference]]

	counter = 0
	baseline_diff = np.array(baseline_diff)
	baseline_rot = baseline_diff[:, 0]
	for current_rot in baseline_rot:
			if current_rot > 30:
				counter = counter + 1.0

	error = counter / len(baseline_rot)
	# print("Baseline Rotational Difference:")
	# print error
	return error

all_error = []
# for i in np.arange(0.6, 1.8, 0.05):
for i in np.arange(0, 90, 1):
	error = simulate_projection(i, 0.8)
	all_error.append(error)

print all_error