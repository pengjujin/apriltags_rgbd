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

def main(args):
	# Declare Test Variables
	# Camera Intrinsics
	fx = 529.29
	fy = 531.28
	px = 466.96
	py = 273.26
	I = np.array([fx, 0 , px, 0, fy, py, 0, 0, 1]).reshape(3,3)
	
	x_start = 586
	x_end = 603
	y_start = 255
	y_end = 268
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
	print "Sample points from the depth sensor"
	print samples_depth[0:5, :]
	cov = np.asarray([sample_cov] * samples_depth.shape[0])
	depth_plane_est = bayesplane.fit_plane_bayes(samples_depth, cov)

	# For now hard code the test data x y values
	# Generate homogenous matrix for pose 
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
	M = np.delete(M, 3, 0)
	print "Extrinsics"
	print M # pose extrinsics
	origin = np.array([0,0,0,1])
	np.transpose(origin)
	C = np.dot(I, M)
	coord = np.dot(C, origin)
	x_coord = coord[0] / coord[2]
	y_coord = coord[1] / coord[2]
	# cv2.circle(rgb_image, (int(x_coord), int(y_coord)), 3, (255, 0,0))
	# cv2.imshow('april_tag', rgb_image)
	# cv2.waitKey(5)
	# cv2.destroyAllWindows()

	x_samples = np.linspace(-0.01, 0.01, num = 10)
	y_samples = np.linspace(-0.01, 0.01, num = 10)
	sample_points = []
	sample_points_test = []
	for i in x_samples:
		for j in y_samples:
			sample_points.append([i,j,0,1])
			sample_points_test.append([i,j, 0])
	sample_points = np.transpose(np.array(sample_points))
	sample_points_viz = np.dot(C, sample_points)
	sample_rgb = np.transpose(np.dot(M, sample_points))
	sample_points_test = np.array(sample_points_test)
	for i in range(0, 50):
		x_coord = sample_points_viz[0, i] / sample_points_viz[2, i]
		y_coord = sample_points_viz[1, i] / sample_points_viz[2, i]
		cv2.circle(rgb_image, (int(x_coord), int(y_coord)), 3, (255, 0,0))
	# cv2.imshow('april_tag', rgb_image)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	print "Sample points from the RGB sensor"
	print  sample_rgb[0:5, :]
	cov = np.asarray([0.9] * sample_rgb.shape[0])
	rgb_plane_est = bayesplane.fit_plane_bayes(sample_rgb, cov)
	
	## Plotting for visual effects
	print "rgb_plane_est cov: "
	print rgb_plane_est.cov
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	
	#ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2], c='b')
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')
	ax.scatter(sample_rgb[:, 0], sample_rgb[:, 1], sample_rgb[:, 2], c='b')
	ax.scatter(samples_depth[:, 0], samples_depth[:, 1], samples_depth[:, 2], c='g')
   	#rgbplane = rgb_plane_est.plot(10, center=np.array([0.190, -0.450, 1.59]), scale= 0.01, color='r', ax=ax)
	#plt.show()

	## Kalman Update stage
	mean_rgb = rgb_plane_est.mean.vectorize()[:, np.newaxis].T
	mean_depth = depth_plane_est.mean.vectorize()[:, np.newaxis].T
	#cov_rgb = rgb_plane_est.cov
	cov_depth = depth_plane_est.cov
	print "cov_depth: "
	print depth_plane_est.cov
	print "cov_rgb: "
	print rgb_plane_est.cov
	cov_rgb = np.eye(4)
	cov_depth = np.eye(4)
	cov_rgb_sq = np.dot(cov_rgb.T, cov_rgb)
	cov_depth_sq = np.dot(cov_depth.T, cov_depth)
	mean_fused = np.dot((np.dot(mean_rgb, cov_rgb_sq) + np.dot(mean_depth, cov_depth_sq)) , np.linalg.inv(cov_rgb_sq + cov_depth_sq))
	mean_fused = mean_fused.flatten()
	fuse_plane = plane.Plane(mean_fused[0:3], mean_fused[3])
	fuse_plane_plot = fuse_plane.plot(center=np.array([0.26, -0.03, 1.16]), scale= 0.01, color='b', ax=ax)
	average_mean = (rgb_plane_est.mean.vectorize() + depth_plane_est.mean.vectorize()) / 2
	average_plane =  plane.Plane(average_mean[0:3], average_mean[3])
	average_plane_plot = average_plane.plot(center=np.array([0.26, -0.03, 1.16]), scale= 0.01, color='r', ax=ax)
	print "mean_rgb: "
	print mean_rgb
	print "mean_depth: "
	print mean_depth
	print "mean_fused: "
	print mean_fused
	print average_mean
	plt.show()


if __name__ == '__main__':
	main(sys.argv)