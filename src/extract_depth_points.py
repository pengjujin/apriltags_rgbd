#!/usr/bin/env python
import sys
import cv2
import numpy as np
import bayesplane
import plane
import transformation as tf

def plot(plane, center=None, scale=1.0, color='r', alpha=1.0, ax=None):
    '''
    2D or 3D render of plane box centered and scaled
    @param center - center of plane box, (2,) or (3,) vector
                    defaults to origin
    @param scale - scale of box, defaults to 1.0
    @param color - color of box, defaults to red
    @param alpha - alpha of box, defaults to 1.0
    @param ax - axis handle, defaults to creating a new one
    @return axis handle
    '''
    import matplotlib.pyplot as plt

    numd = plane.dim()
    if not ((numd == 2) or (numd == 3)):
        raise TypeError('Plotting restricted to 2D or 3D')
    if numd == 3:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from mpl_toolkits.mplot3d import Axes3D
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        if numd == 2:
            ax = fig.gca()
        if numd == 3:
            ax = fig.gca(projection='3d')
            ax.set_zlabel('Z')
        plt.xlabel('X')
        plt.ylabel('Y')
    if center is None:
        center = np.zeros(numd)
    box = plane.box(scale, center)
    if numd == 2:
        ax.plot(box[:, 0], box[:, 1], color, alpha=alpha)
    if numd == 3:
        tri = Poly3DCollection([box])
        tri.set_color(color)
        tri.set_edgecolor('k')
        tri.set_alpha(alpha)
        ax.add_collection3d(tri)
    plt.show()
    return ax


def main(args):
	# Declare Test Variables
	# Camera Intrinsics
	fx = 529.29
	fy = 531.28
	px = 466.96
	py = 273.26
	I = np.array([fx, 0 , px, 0, fy, py, 0, 0, 1]).reshape(3,3)
	
	x_start = 522
	x_end = 536
	y_start = 114
	y_end = 130
	rgb_image = cv2.imread("../data/rgb_frame.png")
	depth_image = cv2.imread("../data/depth_frame.png", cv2.IMREAD_ANYDEPTH)
	april_tag_rgb = rgb_image[y_start:y_end, x_start:x_end]
	april_tag_depth = depth_image[y_start:y_end, x_start:x_end]
	# cv2.imshow('april_tag', april_tag_depth)
	# cv2.waitKey(0)
	all_pts = []
	for i in range(x_start, x_end):
		for j in range(y_start, y_end):
			depth = depth_image[j,i] / 1000.0
			if(depth != 0):
				x = (i - px) * depth / fx
				y = (j - py) * depth / fy
				all_pts.append([x,y,depth])
	sample_cov = 0.05
	samples = np.array(all_pts)
	print "Sample points from the depth sensor"
	print samples[0:5, :]
	cov = np.asarray([sample_cov] * samples.shape[0])
	depth_plane_est = bayesplane.fit_plane_bayes(samples, cov)

	# For now hard code the test data x y values
	# Generate homogenous matrix for pose 
	x_r = 0.628904642725
	y_r = 0.714860496858
	z_r = 0.078189643076
	w_r = -0.295533077855
	M = tf.quaternion_matrix([w_r,x_r,y_r,z_r]) 
	x_t = 0.190438637432
	y_t = -0.450768199945
	z_t = 1.59547154445
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
	cv2.circle(rgb_image, (int(x_coord), int(y_coord)), 3, (255, 0,0))
	cv2.imshow('april_tag', rgb_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

	x_samples = np.linspace(-0.01, 0.01, num = 10)
	y_samples = np.linspace(-0.01, 0.01, num = 10)
	sample_points = []
	for i in x_samples:
		for j in y_samples:
			sample_points.append([i,j,0,1])
	sample_points = np.transpose(np.array(sample_points))
	sample_points_viz = np.dot(C, sample_points)
	sample_points = np.transpose(np.dot(M, sample_points))
	for i in range(0, 50):
		x_coord = sample_points_viz[0, i] / sample_points_viz[2, i]
		y_coord = sample_points_viz[1, i] / sample_points_viz[2, i]
		cv2.circle(rgb_image, (int(x_coord), int(y_coord)), 3, (255, 0,0))
	cv2.imshow('april_tag', rgb_image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	print "Sample points from the RGB sensor"
	print  sample_points[0:5, :]
	rgb_plane_est = bayesplane.fit_plane_bayes(sample_points, cov)
	plot(rgb_plane_est, 10)

if __name__ == '__main__':
	main(sys.argv)