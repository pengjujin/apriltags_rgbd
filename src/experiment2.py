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

def info_parser(info_file):
	f = open(info_file, 'r')
	all_file = f.read()
	format_string = '''
			detection_id: {0} 
			point1:
			corner_x: {1} 
			corner_y: {2}
			corner_z: {3}
			point2:
			corner_x: {4} 
			corner_y: {5}
			corner_z: {6}
			point3:
			corner_x: {7} 
			corner_y: {8}
			corner_z: {9}
			point4:
			corner_x: {10} 
			corner_y: {11}
			corner_z: {12}

			tag_size: {13}
			position_x: {14}
			position_y: {15}
			position_z: {16}
			rotation_x: {17}
			rotation_y: {18}
			rotation_z: {19}
			rotation_w: {20}
			'''
	parsed = parse.parse(format_string, all_file)
	return parsed

def cv2hom(rvec, tvec):
	rmat, jacob = cv2.Rodrigues(rvec)
	H = np.eye(4)
	H[0:3][:,0:3] = rmat
	H[0:3][:, 3] = tvec.reshape(3)
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

def run_experiment(data_index):
	print ("----------- Begin Exp:" + str(data_index))
	rgb_image_path = ("../data/iros_data2/rgb_frame%04d.png" % (data_index, ))
	depth_image_path = ("../data/iros_data2/depth_frame%04d.png" % (data_index,))
	info_file_path = ("../data/iros_data2/apriltag_info_%04d.txt" % (data_index,))
	rgb_image = cv2.imread(rgb_image_path)
	depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)
	info_file = info_file_path


	# Getting Apriltag groundtruth information from the chessboard 
	info_array = info_parser(info_file)
	corner1 = [float(info_array[1]), float(info_array[2])]
	corner2 = [float(info_array[4]), float(info_array[5])]
	corner3 = [float(info_array[7]), float(info_array[8])]
	corner4 = [float(info_array[10]), float(info_array[11])]

	px = float(info_array[14])
	py = float(info_array[15])
	pz = float(info_array[16])
	rx = float(info_array[17])
	ry = float(info_array[18])
	rz = float(info_array[19])
	rw = float(info_array[20])

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
		

		exp_rmat = tf.quaternion_matrix([rw, rx, ry, rz])
		exp_h = exp_rmat
		exp_rvec, _ = cv2.Rodrigues(exp_h[0:3][:, 0:3])
		exp_tvec = np.array([px, py, pz]).reshape(3)
		exp_h[0:3][:, 3] = exp_tvec #The pose of the tag measured from the Rosnode
		print "Experimental H:"
		print exp_h

		angle_diff = quatAngleDiff(groundtruth_h[0:3][:,0:3], exp_h[0:3][:, 0:3])
		print angle_diff

		origin_point = np.float32([[0,0,0]])
		origin_pt, _ = cv2.projectPoints(origin_point, exp_rvec, exp_tvec, mtx, dist)
		origin_pt2, _ = cv2.projectPoints(origin_point, gt_rvec, gt_tvec, mtx, dist)
		print origin_pt
		print origin_pt2
		x = origin_pt[0][0][0]
		y = origin_pt[0][0][1]
		x2 = origin_pt2[0][0][0]
		y2 = origin_pt2[0][0][1]
		cv2.circle(rgb_image, (x,y), 6, (0,0,255), -1)
		cv2.circle(rgb_image, (x,y), 2, (0,255,0), -1)
		cv2.namedWindow('img', 1)
		imgpts2, _ = cv2.projectPoints(axis, exp_rvec, exp_tvec, mtx, dist)
		# rgb_image = draw_origin(rgb_image, origin_pt, imgpts2)
		# cv2.imshow('img', rgb_image)
		# cv2.waitKey(0)
	else:
		print("Cannot localize using Chessboard")
		return False
	#### Experiment 
	print corner1
	im_pt1 = corner1
	im_pt2 = corner2
	im_pt3 = corner3
	im_pt4 = corner4

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
	nrvec2, ntvec2 = fuse.solvePnP_D(rgb_image, depth_image, object_pts, image_pts, mtx, dist, 0)

	print("Tvec RGBD:")
	print ntvec

	print("Tvec D:")
	print ntvec2


	print("GroundTruth:")
	print exp_tvec
	test_tvec = exp_tvec.reshape(3,1)

	rot_difference1 = lm.quatAngleDiff(nrvec, gt_rvec)
	print("RGBD Rotational Difference:")
	print rot_difference1

	trans_difference1 = np.linalg.norm(ntvec - test_tvec)
	print("RGBD Translation Difference:")
	print trans_difference1

	rot_difference2 = lm.quatAngleDiff(nrvec2, gt_rvec)
	print("Depth Rotational Difference:")
	print rot_difference2

	trans_difference2 = np.linalg.norm(ntvec2 - test_tvec)
	print("Depth Translation Difference:")
	print trans_difference2

	return rot_difference1, rot_difference2
	################ Experiment 1 ########################
	# print "----------- 0.5 Noise ----------------"
	# baseline_diff = []
	# test_diff = []
	# test_diff2 = []
	# sample_size = 100
	# n = 4
	# for k in range(sample_size):
	# 	modified_img_pts = image_pts
	# 	normal_noise = np.random.normal(0, 0.5, 8).reshape(4,2)
	# 	modified_img_pts = modified_img_pts + normal_noise
	# 	retval, cv2rvec, cv2tvec = cv2.solvePnP(object_pts, modified_img_pts, mtx, dist, flags=cv2.ITERATIVE)
	# 	baseline_rvec_difference = lm.quatAngleDiff(cv2rvec, gt_rvec)
	# 	baseline_tvec_difference = np.linalg.norm(cv2tvec - gt_tvec)
	# 	baseline_diff = baseline_diff + [[baseline_rvec_difference, baseline_tvec_difference]]
	# 	nrvec, ntvec = fuse.solvePnP_RGBD(rgb_image, depth_image, object_pts, modified_img_pts, mtx, dist, 0)
	# 	test_rvec_difference = lm.quatAngleDiff(nrvec, gt_rvec)
	# 	test_tvec_difference = np.linalg.norm(ntvec - gt_tvec)
	# 	test_diff = test_diff + [[test_rvec_difference, test_tvec_difference]]
	# 	nrvec2, ntvec2 = fuse.solvePnP_D(rgb_image, depth_image, object_pts, modified_img_pts, mtx, dist, 0)
	# 	test_rvec_difference2 = lm.quatAngleDiff(nrvec2, gt_rvec)
	# 	test_tvec_difference2 = np.linalg.norm(ntvec2 - gt_tvec)
	# 	test_diff2 = test_diff2 + [[test_rvec_difference2, test_tvec_difference2]]

	# test_diff = np.array(test_diff)
	# rvec_diff = test_diff[:, 0] 
	# tvec_diff = test_diff[:, 1]
	# rvec_diff_avg = np.sum(rvec_diff) / sample_size
	# tvec_diff_avg = np.sum(tvec_diff) / sample_size
	# print(rvec_diff_avg)
	# print(tvec_diff_avg)

	# test_diff2 = np.array(test_diff2)
	# rvec_diff2 = test_diff2[:, 0] 
	# tvec_diff2 = test_diff2[:, 1]
	# rvec_diff_avg2 = np.sum(rvec_diff2) / sample_size
	# tvec_diff_avg2 = np.sum(tvec_diff2) / sample_size
	# print(rvec_diff_avg2)
	# print(tvec_diff_avg2)
	# bins_define = np.arange(0,180, 1)
	# baseline_diff = np.array(baseline_diff)
	# baseline_rot = baseline_diff[:, 0]
	# test_diff = np.array(test_diff)
	# test_rot = test_diff[:,0]
	# counter = 0
	# counter2 = 0
	# for current_rot in baseline_rot:
	# 	if current_rot > 30:
	# 		counter = counter + 1.0
	# for current_rot in test_rot:
	# 	if current_rot > 30:
	# 		counter2 = counter2 + 1.0
	
	# error = counter / len(baseline_rot)
	# error2 = counter2 / len(test_rot)
	# print "Error percentage:"
	# print error
	# f, axarr = plt.subplots(2, sharex=True, sharey=True)
	# axarr[0].hist(baseline_rot, bins_define, normed = 1, facecolor='red', alpha=0.75)
	# plt.xlabel('Rotation Error')
	# plt.ylabel('Frequnecy')
	# axarr[0].axis([0, 180, 0, 0.15])
	# axarr[0].grid(True)


	# axarr[1].hist(test_rot, bins_define, normed = 1, facecolor='blue', alpha=0.75)
	# axarr[1].axis([0, 180, 0, 0.15])
	# axarr[1].grid(True)
	# savepath = ("../data/iros_results/distance/result_%04d.png" % (data_index, ))
	# f.savefig(savepath)
	# return error, error2	
	# return True

def main():
	all_error_base = []
	all_error_test = []
	for ind in range(1, 23):
		current_error1, current_error2 = run_experiment(ind)
		all_error_base = all_error_base + [current_error1]
		all_error_test = all_error_test + [current_error2]
	print "----------- Averages ----------------"
	print np.average(all_error_base)
	print np.var(all_error_base)
	print np.average(all_error_test)
	print np.var(all_error_test)

main()