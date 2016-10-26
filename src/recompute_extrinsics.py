#!/usr/bin/env python
import sys
import cv2
import numpy as np
import bayesplane
import plane
import transformation as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math

def print_att(att, mat):
	print att + ":"
	print mat

def normal_transfomation(init_normal, goal_normal):
	vector_init = init_normal
	vector_goal = goal_normal
	vector_cross = np.cross(vector_init, vector_goal)
	vector_sin = np.linalg.norm(vector_cross)
	vector_cos = np.dot(vector_init, vector_goal)
	vector_skew = np.array([[0, -vector_cross[2], vector_cross[1]],
							   [vector_cross[2], 0, -vector_cross[0]],
							   [-vector_cross[1], vector_cross[0], 0]])
	vector_eye = np.eye(3)
	R = vector_eye + vector_skew + np.linalg.matrix_power(vector_skew, 2) * (1 - vector_cos) / (vector_sin * vector_sin)
	print_att("depth_R", R)
	[rvec, job] = cv2.Rodrigues(R)
	return rvec

def plot_vector(vector, ax):
	soa =np.array( [vector]) 
	X,Y,Z,U,V,W = zip(*soa)
	print_att("x", X)
	ax.quiver(X,Y,Z,U,V,W, length = 0.05)

	return ax

def plot_samples(samples, ax):
	ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c='g')
	return ax

def main(args):
	# Declare Test Variables
	# Camera Intrinsics
	tag_size = 0.0480000004172
	tag_radius = tag_size / 2.0
	fx = 529.29
	fy = 531.28
	px = 466.96
	py = 273.26
	I = np.array([fx, 0 , px, 0, fy, py, 0, 0, 1]).reshape(3,3)
	D = np.zeros((5,1))
	im_pt1 = [584.5,268.5]
	im_pt2 = [603.5,274.5]
	im_pt3 = [604.5,254.5]
	im_pt4 = [586.5,249.5]
	im_pts = im_pt1 + im_pt2 + im_pt3 + im_pt4
	image_pts = np.array(im_pts).reshape(4,2,1)
	ob_pt1 = [-tag_radius, -tag_radius, 0.0]
	ob_pt2 = [ tag_radius, -tag_radius, 0.0]
	ob_pt3 = [ tag_radius,  tag_radius, 0.0]
	ob_pt4 = [-tag_radius,  tag_radius, 0.0]
	ob_pts = ob_pt1 + ob_pt2 + ob_pt3 + ob_pt4
	object_pts = np.array(ob_pts).reshape(4,3,1)
	print_att("object_pts", object_pts)
	retval, rvec, tvec = cv2.solvePnP(object_pts, image_pts, I, D, flags=cv2.ITERATIVE)
	print "rvec:"
	print rvec
	print "tvec:"
	print tvec
	rotM = cv2.Rodrigues(rvec)[0]
	camera_extrinsics = np.eye(4)
	camera_extrinsics[0:3, 0:3] = rotM
	camera_extrinsics[0:3, 3:4] = tvec
	print "Extrinsics using rgb corner: "
	print camera_extrinsics

	## Generate the depth samples from the depth image
	x_start = 584
	x_end = 600
	y_start = 256
	y_end = 266
	rgb_image = cv2.imread("../data/rgb_frame2.png")
	depth_image = cv2.imread("../data/depth_frame2.png", cv2.IMREAD_ANYDEPTH)
	april_tag_rgb = rgb_image[y_start:y_end, x_start:x_end]
	april_tag_depth = depth_image[y_start:y_end, x_start:x_end]
	# cv2.imshow('april_tag', april_tag_rgb)
	# cv2.waitKey(0)
	all_pts = []
	for i in range(x_start, x_end):
		for j in range(y_start, y_end):
			depth = depth_image[j,i] / 1000.0
			if(depth != 0):
				x = (i - px) * depth / fx
				y = (j - py) * depth / fy
				all_pts.append([x,y,depth])
	sample_cov = 0.9
	samples_depth = np.array(all_pts)
	cov = np.asarray([sample_cov] * samples_depth.shape[0])
	depth_plane_est = bayesplane.fit_plane_bayes(samples_depth, cov)
	vector_depth = depth_plane_est.mean.vectorize()[0:3]
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	ax = plot_samples(samples_depth, ax)
	print samples_depth.shape
	depth_center = samples_depth[75, :]
	depth_normal = depth_plane_est.mean.vectorize()[0 : 3] * 0.05
	depth_center = [depth_center[0], depth_center[1], depth_center[2]]
	print start
	start = [x + y for x, y in zip(depth_center, depth_normal)] 
	end = [depth_normal[0], depth_normal[1], depth_normal[2]]
	print end
	depth_normal_vec = start+end
	print_att("depth_normal_vec", depth_normal_vec)
	depthplane = depth_plane_est.mean.plot(center=np.array(depth_center), scale= 0.01, color='g', ax=ax)
	ax = plot_vector(depth_normal_vec, ax)
	# For now hard code the test data x y values
	# Generate homogenous matrix for pose from the message
	x_r = 0.970358818444
	y_r = 0.105224751742
	z_r = 0.145592452085
	w_r = 0.161661229068
	M = tf.quaternion_matrix([w_r,x_r,y_r,z_r]) 
	x_t = 0.290623142918
	y_t = -0.0266687975686
	z_t = 1.20030737138
	M[0, 3] = x_t
	M[1, 3] = y_t
	M[2, 3] = z_t
	M_d = np.delete(M, 3, 0)
	print "Extrinsics"
	print M # pose extrinsics
	
	# Calculating the new pose based on the depth
	init_vector = [0,0,1]
	normal_z = [0,0,0,0,0,1]
	# ax = plot_vector(normal_z, ax)
	rvec_init = normal_transfomation(init_vector, vector_depth)
	print_att("rvec_init", rvec_init)
	retval, rvec, tvec = cv2.solvePnP(object_pts, image_pts, I, D, rvec=rvec_init, flags=cv2.ITERATIVE)
	rotM = cv2.Rodrigues(rvec)[0]

	camera_extrinsics = np.eye(4)

	camera_extrinsics[0:3, 0:3] = rotM
	camera_extrinsics[0:3, 3:4] = tvec
	print "extrinsics_depth_corrected: "
	print camera_extrinsics	

	plt.show()

if __name__ == '__main__':
	main(sys.argv)