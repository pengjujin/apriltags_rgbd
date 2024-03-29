import numpy as np
import cv2
import glob 
import transformation as tf
import math 

mtx = np.array([529.2945, 0.0, 466.9604, 0.0, 531.2834, 273.25937, 0, 0, 1]).reshape(3,3)
dist = np.zeros((5,1))
square_size = 0.029 

def cv2hom(rvec, tvec):
	rmat, jacob = cv2.Rodrigues(rvec)
	H = np.eye(4)
	H[0:3][:,0:3] = rmat
	H[0:3][:, 3] = tvecs.reshape(3)
	return H

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    print corner
    cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

def draw_origin(img, origin, imgpts):
	origin = tuple(origin.ravel())
	cv2.line(img, origin, tuple(imgpts[0].ravel()), (255,0,0), 5)
	cv2.line(img, origin, tuple(imgpts[1].ravel()), (0,255,0), 5)
	cv2.line(img, origin, tuple(imgpts[2].ravel()), (0,0,255), 5)
	return img

def quatAngleDiff(rot1, rot2):
	quat1 = tf.quaternion_from_matrix(rot1)
	quat2 = tf.quaternion_from_matrix(rot2)

	dtheta = math.acos(2*(np.dot(quat1, quat2)**2)-1)
	return math.degrees(dtheta) 

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((5*7, 3), np.float32)
objp[:,:2] = np.mgrid[0:7, 0:5].T.reshape(-1,2)
objp = objp * square_size
axis = np.float32([[0.08,0,0], [0,0.08,0], [0,0,-0.08]]).reshape(-1,3)
objpoints = []
imgpoints = []

img = cv2.imread("chessboard/rgb_frame.png")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', gray)
# cv2.waitKey(0)
ret, corners = cv2.findChessboardCorners(gray, (7,5), None)
if ret == True:
	corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1, -1), criteria)
	rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners, mtx, dist)
	imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
	img = draw(img, corners, imgpts)
	board_h = cv2hom(rvecs, tvecs) # The pose of the board


	aptag_h = np.eye(4)
	aptag_offset = np.array([0.305, -0.006, 0]).reshape(3,1)  # Board measurements
	aptag_h[0:3][:, 3] = aptag_offset.reshape(3)
	aptag_h[0:3][:, 0:3] = np.array([1, 0, 0, 0, -1, 0,0,0,-1]).reshape(3,3) # Orientation offset
	groundtruth_h = np.dot(board_h, aptag_h) # The pose of the tag measured from the board
	print "groundtruth h"
	print groundtruth_h
	groundtruth_rvec, _ = cv2.Rodrigues(groundtruth_h[0:3][:,0:3])
	groundtruth_tvec = groundtruth_h[0:3][:,3].reshape(3,1)
	

	exp_rmat = tf.quaternion_matrix([0.116816084143, 0.991342961954, 0.00679171280228, -0.0595567536713])
	exp_h = exp_rmat
	exp_rvec, _ = cv2.Rodrigues(exp_h[0:3][:, 0:3])
	exp_tvec = np.array([0.0343233400823, -0.00234641109975, 0.918781027116]).reshape(3)
	exp_h[0:3][:, 3] = exp_tvec #The pose of the tag measured from the Rosnode
	print exp_h
	angle_diff = quatAngleDiff(groundtruth_h[0:3][:,0:3], exp_h[0:3][:, 0:3])
	print angle_diff

	origin_point = np.float32([[0,0,0]])
	origin_pt, _ = cv2.projectPoints(origin_point, groundtruth_rvec, groundtruth_tvec, mtx, dist)
	x = origin_pt[0][0][0]
	y = origin_pt[0][0][1]
	cv2.circle(img, (x,y), 5, (0,0,255), -1)

	imgpts2, _ = cv2.projectPoints(axis, exp_rvec, exp_tvec, mtx, dist)
	print imgpts2
	img = draw_origin(img, origin_pt, imgpts2)
	cv2.imshow('img', img)
	k = cv2.waitKey(0)


# cv2.destroyAllWindows()
