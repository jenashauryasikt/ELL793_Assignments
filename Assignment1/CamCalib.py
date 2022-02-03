import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# TODO : Remove this Function
def read_image(filename):
    # Make Image Grayscale
    img = Image.open(filename).convert('L')
    np_img = np.asarray(img)
    return np_img


# -----------------------------------------------------
# Utils

def normalise(D):
    dim = D.shape[1]
    n = D.shape[0]
    
    D_centre = [sum(x)/len(x) for x in zip(*D)]
    D_centre = np.asarray(D_centre)
    D0 = D - D_centre

    avg_dist_D0 = 0
    for i in range(n):
        avg_dist_D0 = avg_dist_D0 + np.linalg.norm(D0[i,:])
    avg_dist_D0 = avg_dist_D0/float(n)
    scaler1 = np.sqrt(dim)/float(avg_dist_D0)
    D_norm = scaler1*D0
    D_norm = np.append(D_norm, np.ones((n,1)), axis=1)
    D_scale = np.diag([*[scaler1 for i in range(0,dim)],1.0])
    D_shift = np.eye(dim)
    D_shift = np.append(D_shift, [[0]*dim], axis=0)
    D_aux1 = -D_centre
    D_aux2 = np.append(D_aux1, [1])
    D_aux3 = D_aux2[..., None]
    D_shift = np.append(D_shift, D_aux3, axis=1)
    D_trans = np.matmul(D_scale, D_shift) 

    return D_norm,D_trans
# ---------------------------------------------

def DLT(D_norm,D_trans,d_norm,d_trans):
    n = D_norm.shape[0]
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
    return M

def test(M,D,d,path=None):
    projected_pts = (np.matmul(M, D.T)).T
    projected_pts[:,0] = projected_pts[:,0]/(projected_pts[:,2])
    projected_pts[:,1] = projected_pts[:,1]/(projected_pts[:,2])
    projected_pts[:,2] = projected_pts[:,2]/(projected_pts[:,2])

    err = projected_pts - d
    sqe = err**2
    mse = np.mean(sqe[:,:2])
    rmse = np.sqrt(mse)
    print("Error is " ,rmse)

    if path is not None:
        plt.figure(figsize=(40,40))
        plt.scatter(d[:,0],d[:,1],facecolors='none',edgecolor='k',marker='o',linewidth=10,s=2000)

        plt.scatter(projected_pts[:,0],projected_pts[:,1],color='r',marker='^',s=2000)
        plt.xlabel('X axis',fontsize=100)
        plt.ylabel("Y Axis",fontsize=100)
        plt.legend(['Camera Points','Predicted Points'],loc='lower left',prop={'size':100},fontsize=100)
        plt.grid(linewidth=10)
        plt.savefig(path)

    return

# -----------------------------------------------------------

if __name__ == '__main__':
# --------------- Dataset ----------------

    X = [[4,6,0],
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

    D = np.asarray(X)

    # Image coordinates in 2D (3516x3516)
    Y = [[1220,2221],
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
    d = np.asarray(Y)
    # ----------------------------------------------------

    # Train Test Split
    n = 16
    D_test = D[n:,:]
    d_test = d[n:,:]
    D = D[:n,:]
    d = d[:n,:]

    D_norm,D_trans = normalise(D)
    d_norm,d_trans = normalise(d)
    # Convert to Homogenous Coordinates
    d = np.append(d, np.ones((n,1)), axis=1)
    D = np.append(D, np.ones((n,1)), axis=1)
    d_test = np.append(d_test, np.ones((len(d_test),1)), axis=1)
    D_test = np.append(D_test, np.ones((len(D_test),1)), axis=1)

    M = DLT(D_norm,D_trans,d_norm,d_trans)

    test(M,D,d,'Results/Result.png')
    