import lmfit
import numpy as np
import matplotlib.pyplot as plt
import cv2
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
x = np.linspace(1, 10, 250)
np.random.seed(0)
y = 3.0 * np.exp(-x / 2) - 5.0 * np.exp(-(x - 0.1) / 10.) + 0.1 * np.random.randn(len(x))

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
	return err

def residual(p, K, D, object_pt, img_pt):
	return model(p, K, D, object_pt) - img_pt

def jac(p, K, D, object_pt, img_pt):
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

x0 = 

res = least_squares(residual, x0, jac=jac, args=(K, D, object_pt, img_pt), verbose = 1, method='lm')