import numpy as np
import cv2
import glob 
import transformation as tf
import math 
import rgb_depth_fuse as fuse 
import LM_minimize as lm 
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
# Experiment Parameters
# mtx = Camera Matrix
# dist = Camera distortion
# square_size = Chessboard square size
# tag_size = Apriltag size
# tag_radius = Apriltag size / 2
mtx = np.array([529.2945, 0.0, 466.9604, 0.0, 531.2834, 273.25937, 0, 0, 1]).reshape(3,3) 
dist = np.zeros((5,1))
square_size = 0.047  
tag_size = 0.0480000004172
tag_radius = tag_size / 2.0

rgb_image = cv2.imread("../data/iros_data/rgb_frame0000.png")
depth_image = cv2.imread("../data/iros_data/depth_frame0000.png", cv2.IMREAD_ANYDEPTH)

def cv2hom(rvec, tvec):
	rmat, jacob = cv2.Rodrigues(rvec)
	H = np.eye(4)
	H[0:3][:,0:3] = rmat
	H[0:3][:, 3] = tvecs.reshape(3)
	return H

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
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

# Getting Apriltag groundtruth information from the chessboard 

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((3*4, 3), np.float32)
objp[:,:2] = np.mgrid[0:4, 0:3].T.reshape(-1,2)
objp = objp * square_size
axis = np.float32([[0.08,0,0], [0,0.08,0], [0,0,-0.08]]).reshape(-1,3)
objpoints = []
imgpoints = []

gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (4,3), None)
print ret
if ret == True:
	corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1, -1), criteria)
	rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners, mtx, dist)
	imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
	rgb_image = draw(rgb_image, corners, imgpts)
	board_h = cv2hom(rvecs, tvecs) # The pose of the board


	aptag_h = np.eye(4)
	aptag_offset = np.array([0.265, -0.003, 0]).reshape(3,1)  # Board measurements
	aptag_h[0:3][:, 3] = aptag_offset.reshape(3)
	aptag_h[0:3][:, 0:3] = np.array([1, 0, 0, 0, -1, 0,0,0,-1]).reshape(3,3) # Orientation offset
	groundtruth_h = np.dot(board_h, aptag_h) # The pose of the tag measured from the board
	print "Groundtruth H:"
	print groundtruth_h
	gt_rvec, _ = cv2.Rodrigues(groundtruth_h[0:3][:,0:3])
	gt_tvec = groundtruth_h[0:3][:,3].reshape(3,1)
	

	exp_rmat = tf.quaternion_matrix([0.0594791427379, 0.885966679653, -0.0187847792584, -0.459534989083])
	exp_h = exp_rmat
	exp_rvec, _ = cv2.Rodrigues(exp_h[0:3][:, 0:3])
	exp_tvec = np.array([0.201772100798, -0.139835385971, 1.27532936921]).reshape(3)
	exp_h[0:3][:, 3] = exp_tvec #The pose of the tag measured from the Rosnode
	print "Experimental H:"
	print exp_h

	angle_diff = quatAngleDiff(groundtruth_h[0:3][:,0:3], exp_h[0:3][:, 0:3])
	print angle_diff

	origin_point = np.float32([[0,0,0]])
	origin_pt, _ = cv2.projectPoints(origin_point, gt_rvec, gt_tvec, mtx, dist)
	x = origin_pt[0][0][0]
	y = origin_pt[0][0][1]
	cv2.circle(rgb_image, (x,y), 2, (0,0,255), -1)

	# cv2.namedWindow('img', 1)
	# imgpts2, _ = cv2.projectPoints(axis, exp_rvec, exp_tvec, mtx, dist)
	# rgb_image = draw_origin(rgb_image, origin_pt, imgpts2)
	# cv2.imshow('img', rgb_image)
	# cv2.waitKey(0)

#### Experiment 
im_pt1 = [543.5,226.5]
im_pt2 = [557.5,223.5]
im_pt3 = [557.5,203.5]
im_pt4 = [543.5,207.5]    
im_pts = im_pt1 + im_pt2 + im_pt3 + im_pt4
image_pts = np.array(im_pts).reshape(4,2)
ob_pt1 = [-tag_radius, -tag_radius, 0.0]
ob_pt2 = [ tag_radius, -tag_radius, 0.0]
ob_pt3 = [ tag_radius,  tag_radius, 0.0]
ob_pt4 = [-tag_radius,  tag_radius, 0.0]
ob_pts = ob_pt1 + ob_pt2 + ob_pt3 + ob_pt4
object_pts = np.array(ob_pts).reshape(4,3)

retval, cv2rvec, cv2tvec = cv2.solvePnP(object_pts, image_pts, mtx, dist, flags=cv2.ITERATIVE)

################# Baseline ###########################
print "----------- Basic Test ----------------"
print("Baseline rvec:")
print cv2rvec
print("Baseline tvec:")
print cv2tvec

nrvec, ntvec = fuse.solvePnP_RGBD(rgb_image, depth_image, object_pts, image_pts, mtx, dist, 0)
print("Test rvec:")
print nrvec
print("Test tvec:")
print ntvec

rot_difference = lm.quatAngleDiff(cv2rvec, gt_rvec)
print("Baseline Rotational Difference:")
print rot_difference

trans_difference = np.linalg.norm(cv2tvec - gt_tvec)
print("Baseline Translation Difference:")
print trans_difference

rot_difference = lm.quatAngleDiff(nrvec, gt_rvec)
print("Baseline Rotational Difference:")
print rot_difference

trans_difference = np.linalg.norm(ntvec - gt_tvec)
print("Baseline Translation Difference:")
print trans_difference

################ Experiment 1 ########################
print "----------- 0.5 Noise ----------------"
baseline_diff = []
test_diff = []
sample_size = 1000
n = 4
for k in range(sample_size):
	modified_img_pts = image_pts
	normal_noise = np.random.normal(0, 1, 8).reshape(4,2)
	modified_img_pts = modified_img_pts + normal_noise
	retval, cv2rvec, cv2tvec = cv2.solvePnP(object_pts, modified_img_pts, mtx, dist, flags=cv2.ITERATIVE)
	baseline_rvec_difference = lm.quatAngleDiff(cv2rvec, gt_rvec)
	baseline_tvec_difference = np.linalg.norm(cv2tvec - gt_tvec)
	baseline_diff = baseline_diff + [[baseline_rvec_difference, baseline_tvec_difference]]
	nrvec, ntvec = fuse.solvePnP_RGBD(rgb_image, depth_image, object_pts, modified_img_pts, mtx, dist, 0)
	test_rvec_difference = lm.quatAngleDiff(nrvec, gt_rvec)
	test_tvec_difference = np.linalg.norm(ntvec - gt_tvec)
	test_diff = test_diff + [[test_rvec_difference, test_tvec_difference]]

bins_define = np.arange(0,180, 2)
# baseline_diff = np.array(baseline_diff)
# baseline_rot = baseline_diff[:, 0]
# plt.figure(1)
# n, bins, patches = plt.hist(baseline_rot, bins_define, normed = 1, facecolor='orange', alpha=0.75)
# plt.xlabel('Rotation Error')
# plt.ylabel('Frequecy')
# plt.title(r'Test Histogram')
# plt.axis([0, 180, 0, 0.15])
# plt.grid(True)

test_diff = np.array(test_diff)
test_rot = test_diff[:,0]
n, bins, patches = plt.hist(test_rot, bins_define, normed = 1, facecolor='blue', alpha=0.75)
plt.xlabel('Rotation Error')
plt.ylabel('Frequecy')
plt.title(r'Test Histogram')
plt.axis([0, 180, 0, 0.2])
plt.grid(True)

plt.show()