#!/usr/bin/env python3
import sys
import cv2
import numpy as np
import bayesplane
import plane
import transformation as tf
import math
import LM_minimize as lm
import rigid_transform as trans

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

def sample_depth_plane(depth_image, image_pts, K):
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
	sample_cov = 0.9
	samples_depth = np.array(all_pts)
	cov = np.asarray([sample_cov] * samples_depth.shape[0])
	depth_plane_est = bayesplane.fit_plane_bayes(samples_depth, cov)
	return depth_plane_est, all_pts

def computeZ (n, d, x, y):
	sum = n[0] * x + n[1] * y
	z = (d - sum) / n[2]
	return z

def getDepthPoints(image_pts, depth_plane_est, depth_image, K):
	fx = K[0][0]
	fy = K[1][1]
	px = K[0][2]
	py = K[1][2]
	all_depth_points = []
	dim = image_pts.shape
	n = depth_plane_est.mean.n
	d = depth_plane_est.mean.d
	for i in range(dim[0]):
		x = int(image_pts[i, 0])
		y = int(image_pts[i, 1])
		# print(x)
		depth = depth_image[y, x] / 1000.0 + 0.00001
		if(depth != 0):
			X = (x - px) * depth / fx
			Y = (y - py) * depth / fy
			Z = computeZ(n, d, X, Y)
			all_depth_points = all_depth_points + [[X, Y, Z]]
	all_depth_points = np.array(all_depth_points)
	return all_depth_points

# def computeExtrinsics(object_pts, image_pts, depth_points, K, D, verbose=0):
#
# 	rdepth, tdepth = trans.rigid_transform_3D(object_pts, depth_points)
# 	if(verbose > 0):
# 		print(rdepth)
# 		print(tdepth)
#
# 	depthH = np.eye(4)
# 	depthH[0:3, 0:3] = rdepth
# 	depthH[0:3, 3:4] = tdepth.reshape(3,1)
# 	if(verbose > 0):
# 		print(depthH)
#
# 	rvec_init, jacob = cv2.Rodrigues(rdepth)
# 	tvec_init = tdepth.reshape(3,1)
# 	nrvec, ntvec = lm.PnPMin(rvec_init, tvec_init, object_pts, image_pts, K, D)
# 	nrvec = nrvec.reshape(3,1)
# 	ntvec = ntvec.reshape(3,1)
# 	return nrvec, ntvec

# def solvePnP_RGBD(rgb_image, depth_image, object_pts, image_pts, K, D, verbose = 0):
# 	depth_plane_est, all_pts = sample_depth_plane(depth_image, image_pts, K)
# 	depth_points = getDepthPoints(image_pts, depth_plane_est, depth_image, K)
# 	return computeExtrinsics(object_pts, image_pts, depth_points, K, D, verbose)

# def main():
# 	gt_rvec = np.array([[ 3.33005081],[ 0.21025803], [-1.34587401]])
# 	gt_tvec = np.array([[ 0.28346233],[-0.02539108], [ 1.177536  ]])
# 	tag_size = 0.0480000004172
# 	tag_radius = tag_size / 2.0
# 	fx = 529.2945
# 	fy = 0.0
# 	px = 466.9604
# 	py = 273.25937
# 	K = np.array([fx, 0 , px, 0, fy, py, 0, 0, 1]).reshape(3,3)
# 	D = np.zeros((5,1))
# 	im_pt1 = [584.5,268.5]
# 	im_pt2 = [603.5,274.5]
# 	im_pt3 = [604.5,254.5]
# 	im_pt4 = [585.5,249.5]    #586.5 bad 585.5 good
# 	im_pts = im_pt1 + im_pt2 + im_pt3 + im_pt4
# 	image_pts = np.array(im_pts).reshape(4,2)
# 	ob_pt1 = [-tag_radius, -tag_radius, 0.0]
# 	ob_pt2 = [ tag_radius, -tag_radius, 0.0]
# 	ob_pt3 = [ tag_radius,  tag_radius, 0.0]
# 	ob_pt4 = [-tag_radius,  tag_radius, 0.0]
# 	ob_pts = ob_pt1 + ob_pt2 + ob_pt3 + ob_pt4
# 	object_pts = np.array(ob_pts).reshape(4,3)

# 	rgb_image = cv2.imread("../../data/rgb_frame2.png", 0)
# 	depth_image = cv2.imread("../../data/depth_frame2.png", cv2.IMREAD_ANYDEPTH)

# 	nrvec, ntvec = solvePnP_RGBD(rgb_image, depth_image, object_pts, image_pts, K, D, 0)
# 	print("nrev:")
# 	print nrvec
# 	print("ntvec:")
# 	print ntvec

# 	retval, cv2rvec, cv2tvec = cv2.solvePnP(object_pts, image_pts, K, D, flags=cv2.ITERATIVE)

# 	print("CV2 rev:")
# 	print cv2rvec
# 	print("CV2 tvec:")
# 	print cv2tvec

# 	rot_difference = lm.quatAngleDiff(cv2rvec, gt_rvec)
# 	print("Computed Rotational Difference:")
# 	print rot_difference

# 	trans_difference = np.linalg.norm(cv2tvec - gt_tvec)
# 	print("Computed Translation Difference:")
# 	print trans_difference


# 	##### Experiement 1
# 	all_diff = []
# 	n = 4 #number of points
# 	for i in range(n):
# 		for j in range(2):
# 			for k in range(2):
# 				modified_img_pts = image_pts
# 				if k == 1:
# 					modified_img_pts[i, j] = modified_img_pts[i, j] + 1
# 				else:
# 					modified_img_pts[i, j] = modified_img_pts[i, j] - 1
# 				retval, cv2rvec, cv2tvec = cv2.solvePnP(object_pts, modified_img_pts, K, D, flags=cv2.ITERATIVE)
# 				rot_difference = lm.quatAngleDiff(cv2rvec, gt_rvec)
# 				trans_difference = np.linalg.norm(cv2tvec - gt_tvec)
# 				all_diff = all_diff + [[rot_difference, trans_difference]]
# 	all_diff = np.array(all_diff)
# 	print "Experiement 1 result:"
# 	print all_diff

# 	RotationalErrors = all_diff[:, 0]
# 	print RotationalErrors

# 	##### Experiement 2
# 	all_diff = []
# 	sample_size = 1000
# 	n = 4 #number of points
# 	for k in range(sample_size):
# 		modified_img_pts = image_pts
# 		normal_noise = np.random.normal(0, 0.6, 8).reshape(4,2)
# 		modified_img_pts = modified_img_pts + normal_noise
# 		retval, cv2rvec, cv2tvec = cv2.solvePnP(object_pts, modified_img_pts, K, D, flags=cv2.ITERATIVE)
# 		rot_difference = lm.quatAngleDiff(cv2rvec, gt_rvec)
# 		trans_difference = np.linalg.norm(cv2tvec - gt_tvec)
# 		all_diff = all_diff + [[rot_difference, trans_difference]]

# 	all_diff = np.array(all_diff)
# 	print "Experiement 2 result:"
# 	print all_diff

# 	RotationalErrors = all_diff[:, 0]
# 	bad_count = 0
# 	for i in range(sample_size):
# 		if(RotationalErrors[i] > 50):
# 			bad_count = bad_count + 1
# 	print bad_count
# 	# print RotationalErrors

# 	##### Experiment 3
# 	all_diff = []
# 	sample_size = 1000
# 	n = 4 #number of points
# 	for k in range(sample_size):
# 		modified_img_pts = image_pts
# 		normal_noise = np.random.normal(0, 0.6, 8).reshape(4,2)
# 		modified_img_pts = modified_img_pts + normal_noise
# 		nrvec, ntvec = solvePnP_RGBD(rgb_image, depth_image, object_pts, modified_img_pts, K, D, 0)
# 		rot_difference = lm.quatAngleDiff(nrvec, gt_rvec)
# 		trans_difference = np.linalg.norm(ntvec - gt_tvec)
# 		all_diff = all_diff + [[rot_difference, trans_difference]]

# 	all_diff = np.array(all_diff)
# 	print "Experiement 3 result:"
# 	print all_diff

# 	RotationalErrors = all_diff[:, 0]
# 	bad_count = 0
# 	for i in range(sample_size):
# 		if(RotationalErrors[i] > 50):
# 			bad_count = bad_count + 1
# 	print bad_count

# 	##### Experiment 4


# if __name__ == "__main__":
# 	main()
