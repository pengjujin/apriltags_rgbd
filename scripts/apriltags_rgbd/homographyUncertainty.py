import numpy as np 
import cv2 
from tf.transformations import quaternion_matrix

A = np.array([ ])

def printMatrixE(a):
   rows = a.shape[0]
   cols = a.shape[1]
   for i in range(0,rows):
      for j in range(0,cols):
         print(("%6.5f" %a[i,j]), end=' ')
      print()
   print()     

def printPoints(toprint, point):

	print(toprint)
	if len(point) == 3 :
		print(" ( " + str(point[0]) + " , " + str(point[1]) + " , " + str(point[2]) + " ) ")
	else : 
		print(" ( " + str(point[0]) + " , " + str(point[1])  + " ) ")

def computeAmatrix( wld, cam ):

	A_matrix = np.zeros((2,9)) 
	r = len(cam)
	
	for i in range(0,r):
		uv_cam = cam[i,:]
		xyz_world = wld[i,:]

		u_cam = uv_cam[0] 
		v_cam = uv_cam[1] 

		# Aentry = np.matrix([[u_cam, v_cam, 1, 0, 0, 0, -u_cam*xyz_world[0], -v_cam*xyz_world[0], -xyz_world[0]],
		# [0, 0, 0, u_cam, v_cam, 1, -u_cam*xyz_world[1], -v_cam*xyz_world[1], -xyz_world[1]]])

		xw = xyz_world[0]
		yw = xyz_world[1]

		Aentry = np.matrix([[xw, yw, 1, 0, 0, 0, -u_cam*xw, -u_cam*yw, -u_cam ],
		[0, 0, 0, xw, yw, 1, -v_cam*xw, -v_cam*yw, -v_cam]])

		if (i == 0): 
			A_matrix = Aentry
		else:
			A_matrix = np.vstack([A_matrix, Aentry])

	return A_matrix 

def computeHomographySVD(Amatrix, A):

	U, s, V = np.linalg.svd(Amatrix, full_matrices=True)

	print(Amatrix)
	
	c = V.shape[1]

	h = V[c-1, :]
	# h = np.asarray(h)
	# h = np.append(h,1)

	h = h.reshape(3,3)

	h1 = np.array([[h[0,0]], [h[1,0]],  [h[2,0]]])
	h2 = np.array([[h[0,1]], [h[1,1]],  [h[2,1]]])
	h3 = np.array([[h[0,2]], [h[1,2]],  [h[2,2]]])

	print(' *********** h **************')
	print(h)

	return h

def testHomographies(M, solvepnp_H, apriltags_H, svd_H):
	Mh = M[:, np.newaxis]
	Mh[2] = 1

	test_solvepnpH = np.dot(solvepnp_H, Mh)
	test_solvepnpH = np.array([test_solvepnpH[0]/test_solvepnpH[2], test_solvepnpH[1]/test_solvepnpH[2], 1])

	test_apriltagsH = np.dot(apriltags_H, Mh)
	test_apriltagsH = np.array([test_apriltagsH[0]/test_apriltagsH[2], test_apriltagsH[1]/test_apriltagsH[2], 1])

	test_svdH = np.dot(svd_H, Mh)
	test_svdH = np.array([test_svdH[0]/test_svdH[2], test_svdH[1]/test_svdH[2], 1])

	printPoints('tests 1 ( from svd_H)', test_solvepnpH)

	printPoints('tests 2 ( apriltags_H)', test_apriltagsH)

	printPoints('tests 3 ( normalized_svd_H)', test_svdH)


def testHomographies_wld(c, solvepnp_H, apriltags_H, svd_H):

	ch = c[:, np.newaxis]
	ch = np.append(ch, 1)

	test_solvepnpH = np.dot(np.linalg.inv(solvepnp_H), ch)

	test_apriltagsH = np.dot(np.linalg.inv(apriltags_H), ch)

	test_svdH = np.dot(np.linalg.inv(svd_H), ch)

	printPoints('tests 1 ( from svd_H)', test_solvepnpH)

	printPoints('tests 2 ( apriltags_H)', test_apriltagsH)

	printPoints('tests 3 ( normalized_svd_H)', test_svdH)

	return test_solvepnpH, test_apriltagsH, test_svdH

def cost_comp(x1, x2):
	sum_ = 0;
	for i in range(0, len(x1)-1):
		dif_ = x1[i]-x2[i]
		sum_ = sum_ + np.power(dif_,2)
	return sum_

def residual(vars, xdata, ydata, t): 

	M = xdata[3, :]
	M1 = M[:, np.newaxis]
	M1[2] = 1
	M = xdata[2, :]
	M2 = M[:, np.newaxis]
	M2[2] = 1
	M = xdata[1, :]
	M3 = M[:, np.newaxis]
	M3[2] = 1
	M = xdata[0, :]
	M4 = M[:, np.newaxis]
	M4[2] = 1

	r = cv2.Rodrigues(np.array([vars[0], vars[1], vars[2]]))[0]
	T = np.zeros((3,3))
	T[:,0] = r[:, 0]
	T[:,1] = r[:, 1]
	T[:,2] = t

	H = np.dot(A, T)

	c1 = np.dot(H, M1)
	c1 = np.array([c1[0]/c1[2], c1[1]/c1[2], 1.])

	c2 = np.dot(H, M2)
	c2 = np.array([c2[0]/c2[2], c2[1]/c2[2], 1.])

	c3 = np.dot(H, M3)
	c3 = np.array([c3[0]/c3[2], c3[1]/c3[2], 1.])

	c4 = np.dot(H, M4)
	c4 = np.array([c4[0]/c4[2], c4[1]/c4[2], 1.])

	# cost function 
	cost = cost_comp(ydata[0, :], c1) + cost_comp(ydata[1, :], c2) + cost_comp(ydata[2, :], c3) + cost_comp(ydata[3, :], c4)
	return (cost)

def ZhangCalculus(h, A):

	h1 = h[:, 0]
	h2 = h[:, 1]
	h3 = h[:, 2]

	# retrieving r and t 
	l = 1 / np.linalg.norm(np.dot(np.linalg.inv(A), h1)) # translation fine, rotation not
	r1 = -l* np.dot(np.linalg.inv(A), h1)
	r2 = -l* np.dot(np.linalg.inv(A), h2)
	r3 = np.cross(r1, r2)
	t = -l * np.dot(np.linalg.inv(A), h3)

	r = np.zeros((3,3))
	r[:, 0] = r1
	r[:, 1] = r2
	r[:, 2] = r3

	print('r')
	print(r)
	print('t')
	print(t)


	U_, s_, V_ = np.linalg.svd(r, full_matrices=True)

	new_r = np.dot(V_.transpose(), U_.transpose()) 

	T =  np.zeros((3,3))
	T[:, 0:2] = new_r[:, 0:2]
	T[:, 2] = t

	print('normalized T')
	print(T)

	H = np.dot(A, T)

	return H , r, new_r, t

def covarianceEstimationHomography(h, wld_data, corners):

	print(' ------------------------- ')
	print('    Uncertainty Homog. ')
	print(' ------------------------- ')

	u, s, v = np.linalg.svd(Amatrix, full_matrices=True)

	# J computation
	eigenvalues = s
	eigenvectors_ata = v.transpose() #eigenvectors are now the columns of eigenvectors_ata
	J= np.array([])

	for k in range(0, 7):
		print(k)
		eigVect =  eigenvectors_ata[:, k]
		print(eigVect)
		eigVal = eigenvalues[k]
		print(eigVal)
		J_entry = np.divide(np.dot(eigVect, eigVect.transpose()), eigVal)
		print(J_entry)

		if J.shape[0] == 0:
			J = J_entry
		else:
			J = J + J_entry

		print(J)

	J = - J 

	print('J')
	print(J)

	h1 = h[0,0]
	h2 = h[0,1]
	h3 = h[0,2]
	h4 = h[1,0]
	h5 = h[1,1]
	h6 = h[1,2]
	h7 = h[2,0]
	h8 = h[2,1]
	h9 = h[2,2]


	# S computation 

	S = np.array([])
	for i in range(1, 4):

		print('Iteration nr : ')
		print(i) 

		a_e = Amatrix[2*i, :] # a _{2i} 
		a_o = Amatrix[2*i-1, :]# a _{2i-1}

		print('a_e') 
		print(a_e)

		print('a_o')
		print(a_o)

		xi = corners[i, 0]
		yi = corners[i, 1]				

		Xi = wld_data[i, 0]
		Yi = wld_data[i, 1]

		sigma = 0.5

		Sigma_sqr = 1 #np.dot(Sigma, Sigma.transpose())
		
		f_oi = sigma*sigma*(h1*h1+h2*h2-2*Xi*(h1*h7+h2*h8)) + 2*Sigma_sqr*(xi*h7*h9+xi*yi*h7*h8+yi*h8*h9) + (sigma*sigma*Xi*Xi + xi*xi*Sigma_sqr)*h7*h7 + (sigma*sigma*Xi*Xi + yi*yi*Sigma_sqr)*h8*h8 + Sigma_sqr*h9*h9

		f_ei = sigma*sigma*(h4*h4+h5*h5-2*Yi*(h4*h7+h5*h8)) + 2*Sigma_sqr*(xi*h7*h9+xi*yi*h7*h8+yi*h8*h9) + (sigma*sigma*Yi*Yi + xi*xi*Sigma_sqr)*h7*h7 + (sigma*sigma*Yi*Yi + yi*yi*Sigma_sqr)*h8*h8 + Sigma_sqr*h9*h9

		f_eoi = sigma*sigma*((h1-Xi*h7)*(h4*Yi*h7) + (h2- Xi*h8)*(h5-Yi*h8))

		print('f_oi')
		print(f_oi)

		print('f_ei')
		print(f_ei) 

		print('f_eoi')
		print(f_eoi)

		S_entry = np.dot(a_o.transpose(), a_o) * f_oi + np.dot(a_e.transpose(), a_e) * f_ei + np.dot(a_o.transpose(), a_e)*f_eoi + np.dot(a_e.transpose(), a_o)*f_eoi

		print('S_entry')
		print(S_entry)

		if S.shape[0] == 0:
			S = S_entry
		else:
			S = S + S_entry

	print('S')
	print(S)

	print(S.shape)
	print(J.shape)

	Cov_H = np.dot(np.dot(J.transpose(), S), J)
	print('Cov_H')
	print(Cov_H)



## main program ...

## Camera parameters
f_x = 529.2945040622658
f_y = 531.2834529497384
u0 = 466.96044871160075
v0 = 273.2593671723483

## Intrinsics Matrix
A = np.array([  [f_x, 0.0, u0],
		[0.0, f_y, v0], 
		[0.0, 0.0, 1.0]])

## Distortation Matrix
D = np.array([[0.0676550948466241, -0.058556753440857666, 0.007350271107666055, -0.00817256648923586]])

tag_size = 0.048
tag_radius = tag_size/2

## Tag Corners
c1 = np.array([454.5, 331.5])
c2 = np.array([485.5, 333.5])
c3 = np.array([487.5, 305.5])
c4 = np.array([454.5, 303.5])
corners = np.array([c4, c3, c2, c1])

M1 = np.array([-tag_radius, -tag_radius, 0])
M2 = np.array([ tag_radius, -tag_radius, 0])
M3 = np.array([ tag_radius,  tag_radius, 0])
M4 = np.array([-tag_radius,  tag_radius, 0])

wld = np.array([M1, M2, M3, M4])

print(' *********************************************** ')

# Solve for tag homography
retval, rvec, tvec = cv2.solvePnP(wld, corners, A, D, np.array([]), np.array([]), False, 0) 
rot = cv2.Rodrigues(rvec)[0]

print('rvec: ')
print(rvec)

# Pose rotation
solvepnp_T = np.zeros((3,4)) 
solvepnp_T[0,:] = np.array([ rot[0,0], rot[0,1], rot[0,2], tvec[0] ])
solvepnp_T[1,:] = np.array([ rot[1,0], rot[1,1], rot[1,2], tvec[1] ])
solvepnp_T[2,:] = np.array([ rot[2,0], rot[2,1], rot[2,2], tvec[2] ])

# print 'solvepnp_T: '
# print solvepnp_T

## solve pnp
# Pose Translation
solvepnp_m = np.zeros((3,3))
solvepnp_m[:, 0] = solvepnp_T[:,0]
solvepnp_m[:, 1] = solvepnp_T[:,1]
solvepnp_m[:, 2] = solvepnp_T[:,3]

solvepnp_H = np.dot(A, solvepnp_m)

# print 'solvepnp_m'
# print solvepnp_m

# print 'solvepnp_H: '
# print(solvepnp_H)

print(' *********************************************** ')

apriltags_T = np.zeros((3,4)) 

r_apriltags = quaternion_matrix([0.0218586, -0.0351169, -0.22179, 0.974217])
rod_apriltags = cv2.Rodrigues(r_apriltags[0:3, 0:3])[0]
apriltags_R = cv2.Rodrigues(np.array([-rod_apriltags[2], -rod_apriltags[1], rod_apriltags[0]]))[0]
	 # problems in quaternion transformation maybe?

t1 = 0.00530124
t2 = 0.06750890
t3 = 0.79176000
apriltags_T[0,:] = np.array([apriltags_R[0,0], apriltags_R[0,1], apriltags_R[0,2], t1])
apriltags_T[1,:] = np.array([apriltags_R[1,0], apriltags_R[1,1], apriltags_R[1,2], t2])
apriltags_T[2,:] = np.array([apriltags_R[2,0], apriltags_R[2,1], apriltags_R[2,2], t3])

print('apriltags_T: ')
print(apriltags_T)

apriltags_m = np.zeros((3,3))
apriltags_m[:, 0] = apriltags_T[:,0]
apriltags_m[:, 1] = apriltags_T[:,1]
apriltags_m[:, 2] = apriltags_T[:,3]

apriltags_H = np.dot(A, apriltags_m)

print('apriltags_m')
print(apriltags_m)

print('apriltags_H: ')
print(apriltags_H)

print(' *********************************************** ')

## svd

Amatrix = computeAmatrix( wld, corners)
svd_H = computeHomographySVD(Amatrix, A)
svd_H = np.asarray(svd_H)

print('Doing RR = inv(A)*H')

svd_m = np.dot(np.linalg.inv(A),svd_H)
print('svd_m')
print(svd_m)

print(' *********************************************** ')

# normalizing 

normalized_svd_H, r_old, r_new, t = ZhangCalculus(svd_H, A)

print('normalized_svd_H')
print(normalized_svd_H) 

# non linear sqr fitting
r = cv2.Rodrigues(r_new)[0]

from scipy.optimize import minimize
vars = (r[0], r[1], r[2])
print(vars)
# args = xdata, ydata

out = minimize(residual, vars, args=(wld, corners, t), options={'xtol': 1e-8, 'disp': True})
print('out')
print(out.x)
svd_opt_ = np.zeros((3,3))
opt_r = cv2.Rodrigues(np.array([out.x[0], out.x[1], out.x[2]]))[0]

print('opt_r')
print(opt_r)

svd_opt_[:,0:2] = opt_r[:, 0:2]
svd_opt_[:, 2] = t

svd_opt_H = np.dot(A, svd_opt_)


print(' ------------------------- ')
print('  Homography Estimations ')
print(' ------------------------- ')


print('From svd (Criminisi approach)')
print(svd_opt_H) 


print('From solvepnp (apriltags approach): ')
print(apriltags_H)



# tests
print('********** Tests **********')
print(' test_svd_H : from criminisi approach (first), normalized (last)')
print(' test_apriltags_H : from apriltags approach')

printPoints('c1 : ', c1)
testHomographies(M4, svd_H, apriltags_H, normalized_svd_H)


printPoints('c2 : ', c2)
testHomographies(M3, svd_H, apriltags_H, normalized_svd_H)


printPoints('c3 : ', c3)
testHomographies(M2, svd_H, apriltags_H, normalized_svd_H)


printPoints('c4 : ', c4)
testHomographies(M1, svd_H, apriltags_H, normalized_svd_H)

print('********** Tests (WLD points) **********')
print(' test_svd_H : from criminisi approach (first), normalized (last)')
print(' test_apriltags_H : from apriltags approach')
wld[3, 2] = 1
wld[2, 2] = 1
wld[1, 2] = 1
wld[0, 2] = 1

printPoints('M4 : ', wld[3, :])
test_solvepnpH, test_apriltagsH, test_svdH = testHomographies_wld(corners[3,:], svd_H, apriltags_H, normalized_svd_H)
svd_p = test_svdH
apriltags_p = test_apriltagsH

printPoints('M3 : ', wld[2, :])
test_solvepnpH, test_apriltagsH, test_svdH = testHomographies_wld(corners[2,:], svd_H, apriltags_H, normalized_svd_H)
svd_p = np.vstack((svd_p, test_svdH))
apriltags_p = np.vstack((apriltags_p, test_apriltagsH))

printPoints('M2 : ', wld[1, :])
test_solvepnpH, test_apriltagsH, test_svdH = testHomographies_wld(corners[1,:], svd_H, apriltags_H, normalized_svd_H)
svd_p = np.vstack((svd_p, test_svdH))
apriltags_p = np.vstack((apriltags_p, test_apriltagsH))

printPoints('M1 : ', wld[0, :])
test_solvepnpH, test_apriltagsH, test_svdH = testHomographies_wld(corners[0,:], svd_H, apriltags_H, normalized_svd_H)
svd_p = np.vstack((svd_p, test_svdH))
apriltags_p = np.vstack((apriltags_p, test_apriltagsH))

## plotting world points 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection= '3d')


i = 0
ax.scatter(wld[3-i, 0], wld[3-i, 1], wld[3-i, 2], c='r',marker='s')
ax.scatter(svd_p[i, 0], svd_p[i, 1], svd_p[i, 2], c='r',marker='o')
ax.scatter(apriltags_p[i, 0], apriltags_p[i, 1], apriltags_p[i, 2], c='r',marker='^')
i = 1
ax.scatter(wld[3-i, 0], wld[3-i, 1], wld[3-i, 2], c='g',marker='s')
ax.scatter(svd_p[i, 0], svd_p[i, 1], svd_p[i, 2], c='g',marker='o')
ax.scatter(apriltags_p[i, 0], apriltags_p[i, 1], apriltags_p[i, 2], c='g' ,marker='^')
i = 2
ax.scatter(wld[3-i, 0], wld[3-i, 1], wld[3-i, 2], c='m',marker='s')
ax.scatter(svd_p[i, 0], svd_p[i, 1], svd_p[i, 2], c='m',marker='o')
ax.scatter(apriltags_p[i, 0], apriltags_p[i, 1], apriltags_p[i, 2], c='m',marker='^')
i = 3
ax.scatter(wld[3-i, 0], wld[3-i, 1], wld[3-i, 2], c='b',marker='s')
ax.scatter(svd_p[i, 0], svd_p[i, 1], svd_p[i, 2], c='b',marker='o')
ax.scatter(apriltags_p[i, 0], apriltags_p[i, 1], apriltags_p[i, 2], c='b',marker='^')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

covarianceEstimationHomography(normalized_svd_H, wld, corners)


#from IPython import embed
#embed()
