from scipy.optimize import least_squares
import numpy as np
import matplotlib.pyplot as plt
import cv2
import transformation as tf
import math
# def residual(params, x, data=None, eps_data=None):
# 	amp = params['amp']
# 	pshift = params['phase']
# 	freq = params['frequency']
# 	decay = params['decay']
# 	model = amp * sin(x * freq + pshift) * exp(-x*x*decay)

# 	return (data-model) / eps_data

# params = Parameters()
# params.add('amp', value=10)
# params.add('decay', value=0.007)
# params.add('phase', value=0.2)
# params.add('frequency', value=3.0)

# out = minimize(residual, params, args=(x, data, eps_data))

def quatAngleDiff(rvec1, rvec2):
	rot1, jac1 = cv2.Rodrigues(rvec1)
	rot2, jac2 = cv2.Rodrigues(rvec2)
	quat1 = tf.quaternion_from_matrix(rot1)
	quat2 = tf.quaternion_from_matrix(rot2)

	# print quat1 / tf.vector_norm(quat1)
	# print quat2 / tf.vector_norm(quat2)

	dtheta = math.acos(2*(np.dot(quat1, quat2)**2)-1)
	return math.degrees(dtheta) 

def model(x, K, D, object_pt):
	rot1 = x[0]
	rot2 = x[1]
	rot3 = x[2]
	trans1 = x[3]
	trans2 = x[4]
	trans3 = x[5]
	rvec = np.array([rot1, rot2, rot3])
	tvec = np.array([trans1, trans2, trans3])
	err, jacob = cv2.projectPoints(object_pt, rvec, tvec, K, D)
	ret = np.array([err[0,0], err[1,0], err[2,0], err[3,0]])
	return ret

def residual(x, K, D, object_pt, image_pt):
	diff = model(x, K, D, object_pt) - image_pt
	# sum_diff = np.array([0,0])
	# for i in diff:
	# 	sum_diff = np.square(sum_diff) + i
	sum_diff = np.concatenate([diff[0], diff[1], diff[2], diff[3]])
	return sum_diff

def jac(x, K, D, object_pt, image_pt):
	rot1 = x[0]
	rot2 = x[1]
	rot3 = x[2]
	trans1 = x[3]
	trans2 = x[4]
	trans3 = x[5]
	rvec = np.array([rot1, rot2, rot3])
	tvec = np.array([trans1, trans2, trans3])
	err, jacob = cv2.projectPoints(object_pt, rvec, tvec, K, D)
	return jacob

def PnPMin(rvec, tvec, object_pt, image_pt, I, D):
	x0 = np.append(rvec, tvec)
	K = I
	D = np.zeros((5,1))
	relaxation = 100
	bounds = ([x0[0]-relaxation, x0[1]-relaxation, x0[2]-relaxation, -np.inf, -np.inf, -np.inf], 
			  [x0[0]+relaxation, x0[1]+relaxation, x0[2]+relaxation, np.inf, np.inf, np.inf])
	# bounds = ([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf], 
	# 	  	  [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf])
	res = least_squares(residual, x0, args=(K, D, object_pt, image_pt), verbose = 0, bounds=bounds)
	# res = least_squares(residual, x0, args=(K, D, object_pt, image_pt), method='lm', verbose = 1)
	rvec_r = res.x[0:3]
	tvec_r = res.x[3:6]
	return rvec_r, tvec_r

# tag_size = 0.0480000004172
# tag_radius = tag_size / 2.0
# ob_pt1 = [-tag_radius, -tag_radius, 0.0]
# ob_pt2 = [ tag_radius, -tag_radius, 0.0]
# ob_pt3 = [ tag_radius,  tag_radius, 0.0]
# ob_pt4 = [-tag_radius,  tag_radius, 0.0]
# ob_pts = ob_pt1 + ob_pt2 + ob_pt3 + ob_pt4
# object_pt = np.array(ob_pts).reshape(4,3)
# im_pt1 = [584.5,268.5]
# im_pt2 = [603.5,274.5]
# im_pt3 = [604.5,254.5]
# im_pt4 = [585.5,249.5]
# im_pts = [im_pt1] + [im_pt2] + [im_pt3] + [im_pt4]
# image_pt = np.array(im_pts).reshape(4,2)
# fx = 529.29
# fy = 531.28
# px = 466.96
# py = 273.26
# I = np.array([fx, 0 , px, 0, fy, py, 0, 0, 1]).reshape(3,3)
# K = I
# D = np.zeros((5,1))
# # x0 = [-0.32106359, 0.29009044, 0.9015352, 0.26856702, -0.02644549, 1.146]
# # x0 = [-1.2 , -1.9, 0.0, 0.26856702, -0.02644549, 1.146]

# # x0 = [2.76072895, 0.3005424, 0.42028413, 0.28941606, -0.026083, 1.19823801]
# # x0 = [ 3.2788047, 0.24224965, -1.34404619,  0.28906966, -0.02585935,  1.19846304]
# print object_pt
# print image_pt
# test_model = model(x0, K, D, object_pt)
# print test_model
# test = residual(x0, K, D, object_pt, image_pt)
# print test
# res = least_squares(residual, x0, args=(K, D, object_pt, image_pt), verbose = 1)
# print res.x

# print np.sum(np.square(test)) * 0.5

# r1 = np.array([ 3.33005081,  0.21025803, -1.34587401])
# r2 = np.array([ 4.21444723e-01,   6.75306829e-01,  -1.54594988e-17])

# ang = quatAngleDiff(r1, r2)
# print math.degrees(ang)