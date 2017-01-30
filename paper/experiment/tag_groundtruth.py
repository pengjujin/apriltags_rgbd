import numpy as np
import cv2
import glob 

mtx = np.array([529.2945, 0.0, 466.9604, 0.0, 531.2834, 273.25937, 0, 0, 1]).reshape(3,3)
dist = np.zeros((5,1))
square_size = 0.029 
def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    cv2.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    cv2.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    cv2.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((5*7, 3), np.float32)
objp[:,:2] = np.mgrid[0:7, 0:5].T.reshape(-1,2)
objp = objp * square_size
axis = np.float32([[0.08,0,0], [0,0.08,0], [0,0,-0.08]]).reshape(-1,3)
objpoints = []
imgpoints = []

img = cv2.imread("chessboard/board_aptag.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', gray)
# cv2.waitKey(0)
ret, corners = cv2.findChessboardCorners(gray, (7,5), None)
if ret == True:
	corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1, -1), criteria)
	rvecs, tvecs, inliers = cv2.solvePnPRansac(objp, corners, mtx, dist)
	imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)
	img = draw(img, corners, imgpts)

	measured = np.array([0.305, -0.006, 0]).reshape(3,1)
	

	cv2.imshow('img', img)
	k = cv2.waitKey(0)


# cv2.destroyAllWindows()
