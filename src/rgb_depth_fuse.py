#!/usr/bin/env python
import sys
import cv2
import numpy as np
import bayesplane
import plane
import transformation as tf
import math
import LM_minimize as lm
import rigid_transform as rtrans

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
	[rvec, job] = cv2.Rodrigues(R)
	return rvec

def sample_depth_plane(K, depth_image):
	## Generate the depth samples from the depth image
	fx = K[0][0]
	fy = K[1][1]
	px = K[0][2]
	py = K[1][2]
	x_start = 584
	x_end = 600
	y_start = 256
	y_end = 266
	# rgb_image = cv2.imread("../data/rgb_frame2.png")
	depth_image = cv2.imread("../data/depth_frame2.png", cv2.IMREAD_ANYDEPTH)

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
	return depth_plane_est, samples_depth

def computeZ (n, d, x, y):
	sum = n[0] * x + n[1] * y
	z = (d - sum) / n[2]
	return z

def generate_depth_correspondence(pixel_point, depth_plane_est, K):
	fx = K[0][0]
	fy = K[1][1]
	px = K[0][2]
	py = K[1][2]
	depth_image = cv2.imread("../data/depth_frame2.png", cv2.IMREAD_ANYDEPTH)
	x = pixel_point[0]
	y = pixel_point[1]
	depth = depth_image[y, x] / 1000.0
	if(depth != 0):
		X = (x - px) * depth / fx
		Y = (y - py) * depth / fy
	n = depth_plane_est.mean.n
	d = depth_plane_est.mean.d
	Z = computeZ(n, d, X, Y)
	return [X, Y, Z]

def getDepthPoints(image_pts, depth_plane_est, K)

def computeExtrinsics(object_pts, image_pts, depth_plane_est, K, D, verbose=0):

	depth_points = getDepthPoints()
	rdepth, tdepth = trans.rigid_transform_3D(object_pts, depth_points)
	if(verbose > 0):
		print rdepth
		print tdepth

	depthH = np.eye(4)
	depthH[0:3, 0:3] = rdepth
	depthH[0:3, 3:4] = tdepth.reshape(3,1)
	if(verbose > 0):
		print depthH

	rvec_init, jacob = cv2.Rodrigues(rdepth)
	tvec_init = tdepth.reshape(3,1)
	nrvec, ntvec = lm.PnPMin(rvec_init, tvec_init, object_pts, image_pts, K, D)
	return nrvec, ntvec