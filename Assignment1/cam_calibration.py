from scipy import linalg as LA
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Image
img = Image.open('data_img.jpg').convert('L')
np_img = np.asarray(img)

# Real-world coordinates in 3D
D = [[4,0,4],
	[4,0,6],
	[3,0,7],
	[7,0,9],
	[2,0,4],
	[8,0,2],
	[0,1,9],
	[0,3,8],
	[0,4,4],
	[0,7,7],
	[0,2,2],
	[0,7,3],
	[0,2,6],
	[6,5,10],
	[4,3,10],
	[2,7,10]]
D = np.asarray(D)

# Image coordinates in 2D (3516x3516)
d = [[1220,2221],
	[1237,1875],
	[1386,1716],
	[775,1251],
	[1488,2220],
	[551,2627],
	[1897,1443],
	[2142,1588],
	[2234,2282],
	[2758,1754],
	[1978,2570],
	[2711,2531],
	[1999,1924],
	[1606,895],
	[1626,1089],
	[2516,1089]]
d = np.asarray(d)

n = 16

# Normalize coordinates

## Real-world
D_centre = [sum(x)/len(x) for x in zip(*D)]
D_centre = np.asarray(D_centre)
D0 = D - D_centre
avg_dist_D0 = 0
for i in range(n):
	avg_dist_D0 = avg_dist_D0 + np.linalg.norm(D0[i,:])
avg_dist_D0 = avg_dist_D0/float(n)
scaler1 = np.sqrt(3)/float(avg_dist_D0)
D_norm = scaler1*D0
D_norm = np.append(D_norm, np.ones((n,1)), axis=1)
D_scale = np.diag([scaler1,scaler1,scaler1,1.0])
D_shift = np.eye(3)
D_shift = np.append(D_shift, [[0,0,0]], axis=0)
D_aux1 = -D_centre
D_aux2 = np.append(D_aux1, [1])
D_aux3 = D_aux2[..., None]
D_shift = np.append(D_shift, D_aux3, axis=1)
D_trans = np.matmul(D_scale, D_shift) 

## Image Plane
d_centre = [sum(x)/len(x) for x in zip(*d)]
d_centre = np.asarray(d_centre)
d0 = d - d_centre
avg_dist_d0 = 0
for i in range(n):
	avg_dist_d0 = avg_dist_d0 + np.linalg.norm(d0[i,:])
avg_dist_d0 = avg_dist_d0/float(n)
scaler2 = np.sqrt(2)/float(avg_dist_d0)
d_norm = scaler2*d0
d_norm = np.append(d_norm, np.ones((n,1)), axis=1)
d_scale = np.diag([scaler2,scaler2,1.0])
d_shift = np.eye(2)
d_shift = np.append(d_shift, [[0,0]], axis=0)
d_aux1 = -d_centre
d_aux2 = np.append(d_aux1, [1])
d_aux3 = d_aux2[..., None]
d_shift = np.append(d_shift, d_aux3, axis=1)
d_trans = np.matmul(d_scale, d_shift)

d = np.append(d, np.ones((n,1)), axis=1)
D = np.append(D, np.ones((n,1)), axis=1)

# DLT Method
O = np.array([0,0,0,0])
Q = np.zeros((2*n,12))
for i in range(n):
	temp1 = np.array([])
	temp1 = np.append(temp1, D_norm[i,:])
	temp1 = np.append(temp1, O)
	temp1 = np.append(temp1, -d_norm[i,0]*D_norm[i,:])
	Q[2*i, :] = temp1
	temp2 = np.array([])
	temp2 = np.append(temp2, O)
	temp2 = np.append(temp2, D_norm[i,:])
	temp2 = np.append(temp2, -d_norm[i,1]*D_norm[i,:])
	Q[2*i+1, :] = temp2
A = np.matmul(Q.T, Q)
w, v = LA.eig(A, left=False, right=True)
idx = np.argmin(w)
vec = v[:,idx]
M_norm = np.reshape(vec, (3,4))
M = np.matmul(np.matmul(np.linalg.inv(d_trans), M_norm), D_trans)
projected_pts = (np.matmul(M, D.T)).T
projected_pts[:,0] = projected_pts[:,0]/(projected_pts[:,2])
projected_pts[:,1] = projected_pts[:,1]/(projected_pts[:,2])
projected_pts[:,2] = projected_pts[:,2]/(projected_pts[:,2])

# Error in projections and actual 2D points
# Validate projection matrix
err = projected_pts - d
sqe = err**2
mse = np.mean(sqe)
rmse = np.sqrt(mse)
print("RMSE =",rmse)

# See predicted projections against image projections
plt.figure(figsize=(40,40))
plt.scatter(d[:,0], d[:,1], c='blue', marker='+')
plt.scatter(projected_pts[:,0], projected_pts[:,1], c='red', marker='x')
plt.xlabel("X-axis")
plt.ylabel("Y-Axis")
plt.legend(['Original', 'Predicted'], loc = 'upper left')
plt.savefig('Predictions.png')
plt.show()

# Intrinsic and Extrinsic params
A = M[:,0:3]
R = np.zeros((3,3))
K = np.zeros((3,3))
rho = -1/np.linalg.norm(A[2,:])
R[2,:] = rho*A[2,:]
X0 = rho*rho*(np.matmul(A[0,:], A[2,:].T))
Y0 = rho*rho*(np.matmul(A[1,:], A[2,:].T))
cross1 = np.cross(A[0,:],A[2,:])
cross2 = np.cross(A[1,:],A[2,:])
n_cross1 = np.linalg.norm(cross1)
n_cross2 = np.linalg.norm(cross2)
c1 = -cross1/n_cross1
c2 = cross2[...,None]/n_cross2
theta = np.arccos(np.matmul(c1, c2))
alpha = rho*rho*n_cross1*np.sin(theta)
beta = rho*rho*n_cross2*np.sin(theta)

R[0,:] = cross2/n_cross2
R[1,:] = np.cross(R[2,:],R[0,:])
print("R =\n", R)

K[0,:] = np.array([alpha, -alpha/np.tan(theta), X0])
K[1,:] = np.array([0, beta/np.sin(theta), Y0])
K[2,:] = np.array([0, 0, 1])
print("K =\n", K)

u0 = K[0,2]
v0 = K[1,2]
print("u0 =", u0)
print("v0 =", v0)
print("theta =", theta)
print("alpha =", alpha)
print("beta =", beta)

t = (np.matmul(rho*np.linalg.inv(K), M[:,3].T))[...,None]
print("t=\n", t)

x0 = -np.matmul(np.linalg.inv(R), t)
print("x0=\n", x0)