import scipy as sp
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Image
img = Image.open('data_img.jpg').convert('L')
np_img = np.asarray(img)

# Real-world coordinates in 3D
D = [[4,6,0],
	[4,4,0],
	[3,3,0],
	[7,1,0],
	[2,6,0],
	[8,8,0],
	[0,1,1],
	[0,2,3],
	[0,6,4],
	[0,3,7],
	[0,8,2],
	[0,7,7],
	[0,4,2],
	[6,0,5],
	[4,0,3],
	[2,0,7]]
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
# print(D_trans)

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
# print(d_trans)

d = np.append(d, np.ones((n,1)), axis=1)
D = np.append(d, np.ones((n,1)), axis=1)

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
w, v = np.linalg.eig(A)
idx = np.argmin(w)
vec = v[:,idx]
M_norm = np.reshape(vec, (3,4))
M = np.matmul(np.matmul(np.linalg.inv(d_trans), M_norm), D_trans)
projected_pts = (np.matmul(M, D.T)).T
projected_pts[:,0] = projected_pts[:,0]/(projected_pts[:,2])
projected_pts[:,1] = projected_pts[:,1]/(projected_pts[:,2])
projected_pts[:,2] = projected_pts[:,2]/(projected_pts[:,2])

# Error in projections and actual 2D points
err = projected_pts - d
sqe = err**2
mse = np.mean(sqe)
rmse = np.sqrt(mse)
print(rmse)