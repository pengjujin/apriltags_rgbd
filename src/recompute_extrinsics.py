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
import LM_minimize as lm
import rigid_transform as rtrans
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
	# print_att("depth_R", R)
	[rvec, job] = cv2.Rodrigues(R)
	return rvec, R

def plot_vector(vector, ax):
	soa =np.array( [vector]) 
	X,Y,Z,U,V,W = zip(*soa)
	# print_att("x", X)
	ax.quiver(X,Y,Z,U,V,W, length = 1)

	return ax

def plot_samples(samples, ax, color):
	ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c=color)
	return ax

def sample_rgb_plane():
	fx = 529.29
	fy = 531.28
	px = 466.96
	py = 273.26
	I = np.array([fx, 0 , px, 0, fy, py, 0, 0, 1]).reshape(3,3)
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
	# print "Extrinsics"
	# print M # pose extrinsics
	origin = np.array([0,0,0,1])
	np.transpose(origin)
	C = np.dot(I, M_d)
	coord = np.dot(C, origin)
	x_coord = coord[0] / coord[2]
	y_coord = coord[1] / coord[2]
	x_samples = np.linspace(-0.024, 0.024, num = 10)
	y_samples = np.linspace(-0.024, 0.024, num = 10)
	sample_points = []
	for i in x_samples:
		for j in y_samples:
			sample_points.append([i,j,0,1])
	sample_points = np.transpose(np.array(sample_points))
	sample_points_viz = np.dot(C, sample_points)
	sample_rgb = np.transpose(np.dot(M_d, sample_points))
	cov = np.asarray([0.9] * sample_rgb.shape[0])
	rgb_plane_est = bayesplane.fit_plane_bayes(sample_rgb, cov)
	return rgb_plane_est, sample_rgb

def sample_depth_plane():
	## Generate the depth samples from the depth image
	fx = 529.29
	fy = 531.28
	px = 466.96
	py = 273.26
	x_start = 584
	x_end = 600
	y_start = 256
	y_end = 266
	rgb_image = cv2.imread("../data/rgb_frame2.png")
	depth_image = cv2.imread("../data/depth_frame2.png", cv2.IMREAD_ANYDEPTH)
	april_tag_rgb = rgb_image[y_start:y_end, x_start:x_end]
	april_tag_depth = depth_image[y_start:y_end, x_start:x_end]
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

def generate_depth_correspondence(pixel_point, depth_plane_est):
	fx = 529.29
	fx = 529.29
	fy = 531.28
	px = 466.96
	py = 273.26
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
	im_pt4 = [585.5,249.5]    #586.5 bad 585.5 good
	im_pts = im_pt1 + im_pt2 + im_pt3 + im_pt4
	image_pts = np.array(im_pts).reshape(4,2)
	ob_pt1 = [-tag_radius, -tag_radius, 0.0]
	ob_pt2 = [ tag_radius, -tag_radius, 0.0]
	ob_pt3 = [ tag_radius,  tag_radius, 0.0]
	ob_pt4 = [-tag_radius,  tag_radius, 0.0]
	ob_pts = ob_pt1 + ob_pt2 + ob_pt3 + ob_pt4
	object_pts = np.array(ob_pts).reshape(4,3)
	# print_att("object_pts", object_pts)
	# print_att("image_pts", image_pts)
	retval, rvec, tvec = cv2.solvePnP(object_pts, image_pts, I, D, flags=cv2.ITERATIVE)
	print "cv2 rvec:"
	print rvec
	print "cv2 tvec:"
	print tvec
	cv2rvec = rvec
	rotM = cv2.Rodrigues(rvec)[0]
	camera_extrinsics = np.eye(4)
	camera_extrinsics[0:3, 0:3] = rotM
	camera_extrinsics[0:3, 3:4] = tvec
	cv2H = camera_extrinsics
	# print "Extrinsics using rgb corner: "
	# print camera_extrinsics

	## Init the plots
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')

	## plot depth plane and normals
	depth_plane_est, samples_depth = sample_depth_plane()
	depth_normal = depth_plane_est.mean.vectorize()[0:3]
	ax = plot_samples(samples_depth, ax, 'g')
	depth_center = samples_depth[75, :]
	depth_center = [depth_center[0], depth_center[1], depth_center[2]]
	start = [x + y for x, y in zip([0,0,0], depth_normal)] 
	end = [depth_normal[0], depth_normal[1], depth_normal[2]]
	depth_normal_vec = start+end
	# print_att("depth_normal_vec", depth_normal_vec)
	# depthplane = depth_plane_est.mean.plot(center=np.array(depth_center), scale= 1.5, color='g', ax=ax)
	# ax = plot_vector(depth_normal_vec, ax)

	# Sample rgb plane and normals
	rgb_plane_est, samples_rgb = sample_rgb_plane()
	rgb_normal = rgb_plane_est.mean.vectorize()[0:3]
	rgb_normal = [rgb_normal[0], rgb_normal[1], rgb_normal[2]]
	ax = plot_samples(samples_rgb, ax, 'r')
	# depth_center = samples_rgb[75, :]
	# depth_center = [depth_center[0], depth_center[1], depth_center[2]]
	start = [x + y for x, y in zip([0,0,0], rgb_normal)] 
	end = [rgb_normal[0], rgb_normal[1], rgb_normal[2]]
	rgb_normal_vec = start+end
	# print_att("rgb_normal_vec", rgb_normal_vec)
	# rgb_plane = rgb_plane_est.mean.plot(center=np.array([0,0,0]), scale= 1.5, color='r', ax=ax)
	# ax = plot_vector(rgb_normal_vec, ax)
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
	# print "Extrinsics"
	# print M # pose extrinsics
	
	# Calculating the new pose based on the depth
	init_vector = [0,0,10]
	normal_z = [0,0,0,0,0,1]
	# ax = plot_vector(normal_z, ax)
	testPoint1 = generate_depth_correspondence(im_pt1, depth_plane_est) 
	testPoint2 = generate_depth_correspondence(im_pt2, depth_plane_est)
	testPoint3 = generate_depth_correspondence(im_pt3, depth_plane_est) 
	testPoint4 = generate_depth_correspondence(im_pt4, depth_plane_est)
	print_att("TEST POINT1", testPoint1)
	print_att("TEST POINT2", testPoint2)
	print_att("TEST POINT3", testPoint3)
	print_att("TEST POINT4", testPoint4)
	test_ob_point1 = np.array([-tag_radius, -tag_radius, 0.0, 1.0])
	test_ob_point2 = np.array([ tag_radius, -tag_radius, 0.0, 1.0])
	test_ob_point3 = np.array([ tag_radius,  tag_radius, 0.0, 1.0])
	test_ob_point4 = np.array([-tag_radius,  tag_radius, 0.0, 1.0])
	# result_pt1 = np.dot(cv2H, test_ob_point1.reshape(4,1))
	# result_pt2 = np.dot(cv2H, test_ob_point2.reshape(4,1))
	# result_pt3 = np.dot(cv2H, test_ob_point3.reshape(4,1))
	# result_pt4 = np.dot(cv2H, test_ob_point4.reshape(4,1))
	# print_att("FROM PIXEL1", result_pt1.reshape(1,4))
	# print_att("FROM PIXEL2", result_pt2.reshape(1,4))
	# print_att("FROM PIXEL3", result_pt3.reshape(1,4))
	# print_att("FROM PIXEL4", result_pt4.reshape(1,4))
	test_pts = testPoint1 + testPoint2 + testPoint3 + testPoint4
	depth_points = np.array(test_pts).reshape(4,3)
	tag_pts = np.hstack((test_ob_point1, test_ob_point2, test_ob_point3, test_ob_point4)).reshape(4, 4)
	# print depth_points
	# print tag_pts
	print_att('object_pts', object_pts)
	print_att('depth_pts', depth_points)
	Rdepth, t_depth = rtrans.rigid_transform_3D(object_pts, depth_points)
	print Rdepth
	print t_depth
	depthH = np.eye(4)
	depthH[0:3, 0:3] = Rdepth
	depthH[0:3, 3:4] = t_depth.reshape(3,1)
	print depthH
	result_pt1 = np.dot(depthH, test_ob_point1.reshape(4,1))
	result_pt2 = np.dot(depthH, test_ob_point2.reshape(4,1))
	result_pt3 = np.dot(depthH, test_ob_point3.reshape(4,1))
	result_pt4 = np.dot(depthH, test_ob_point4.reshape(4,1))
	print_att("same as test1", result_pt1.reshape(1,4))
	print_att("same as test2", result_pt2.reshape(1,4))
	print_att("same as test3", result_pt3.reshape(1,4))
	print_att("same as test4", result_pt4.reshape(1,4))
	rvec_init, R = normal_transfomation(init_vector, depth_normal)
	
	rotated_vector = np.dot(R, np.array(init_vector).reshape(3,1))
	plot_norm2 = [rotated_vector[0,0], rotated_vector[1,0], rotated_vector[2,0]]
	# ax = plot_vector(plot_norm2 + plot_norm2, ax)
	tvec_init = np.array(depth_center).reshape(3,1)
	# print_att("rvec_init", rvec_init.reshape(1, 3))
	# print_att("tvec_init", tvec_init.reshape(1, 3))
	nrvec, ntvec = lm.PnPMin(rvec_init, tvec_init, object_pts, image_pts, I, D)
	# print_att("new rvec:", nrvec) 
	# print_att("new tvec:", ntvec)
	# for x in range(0, 100):
	# 	tvec_init = np.array(depth_center).reshape(3,1)
	# 	rvec_init = np.random.rand(3,1)
	# 	print rvec_init
	# 	rvec, tvec, inliners = cv2.solvePnPRansac(object_pts, image_pts, I, D, rvec=rvec_init, tvec=tvec_init, flags=cv2.ITERATIVE)
	# 	rotM = cv2.Rodrigues(rvec)[0]

	# 	camera_extrinsics = np.eye(4)

	# 	camera_extrinsics[0:3, 0:3] = rotM
	# 	camera_extrinsics[0:3, 3:4] = tvec
	# 	print "extrinsics_depth_corrected: "
	# 	print camera_extrinsics	



	## rotating the norm purely based off of the rotation matrix
	rotM = cv2.Rodrigues(nrvec)[0]
	init_norm = [0,0, 1]
	init_norm = np.array(init_norm).reshape(3,1)
	rotated_norm = np.dot(rotM, init_norm)
	# print_att("rotated_norm", rotated_norm)
	plot_norm = [rotated_norm[0,0], rotated_norm[1,0], rotated_norm[2,0]]

	# ax = plot_vector(plot_norm + plot_norm, ax)

	rotM2 = cv2.Rodrigues(cv2rvec)[0]
	# init_norm = [0,0,-1]
	# init_norm = np.array(init_norm).reshape(3,1)
	# rotM2 = np.eye(3)
	rotated_norm2 = np.dot(rotM2, init_norm)
	# print_att("rotated_norm", rotated_norm)
	plot_norm2 = [rotated_norm2[0,0], rotated_norm2[1,0], rotated_norm2[2,0]]
	# ax = plot_vector(plot_norm2 + plot_norm2, ax)
	print rotM
	print rotM2

	plt.show()


if __name__ == '__main__':
	main(sys.argv)