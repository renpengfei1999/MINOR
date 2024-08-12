import numpy as np
import pandas as pd
import numpy.linalg as LA
import scipy
def solve_l1l2(W,lambda1):
    nv=W.shape[1]
    F=W.copy()
    for p in range(nv):
        nw=LA.norm(W[:,p],"fro")
        if nw>lambda1:
            F[:,p]=(nw-lambda1)*W[:,p]/nw
        else:F[:,p]=np.zeros((W[:,p].shape[0],1))
    return F
a=np.mat(np.zeros((541,831)))#Define a 541*831 matrix of zeros
AA=a.astype(int)#Convert matrix AA to integer
#Read the known SM-miRNA associations
b=np.loadtxt('known SM-miRNA interaction.txt')
B=b.astype(int)
#Change the corresponding position element of matrix AA to 1 to obtain the adjacency matrix AA
for x in B:
    AA[x[0]-1,x[1]-1]=1
#Matrix decomposition
A = AA.copy()
alpha = 0.1
J = np.mat(np.zeros((831, 831)))
X = np.mat(np.zeros((831, 831)))
E = np.mat(np.zeros((541, 831)))
Y1 = np.mat(np.zeros((541, 831)))
Y2 = np.mat(np.zeros((831, 831)))
mu = 10 ** -4
max_mu = 10 ** 10
rho = 1.1
epsilon = 10 ** 10
while True:
            [U, sigma1, V] = scipy.linalg.svd(X + Y2 / mu, lapack_driver='gesvd',full_matrices=True)
            G = [sigma1[k] for k in range(len(sigma1)) if sigma1[k] > 1 / mu]
            svp = len(G)
            if svp >= 1:
                sigma1 = sigma1[0:svp] - 1 / mu
            else:
                sigma1 = [0]
                svp = 1
            J = np.mat(U[:, 0:svp]) * np.mat(np.diag(sigma1)) * np.mat(V[0:svp, :])
            ATA = A.T * A
            X = (ATA + np.eye(831)).I * (ATA - A.T * E + J + (A.T * Y1 - Y2) / mu)
            temp1 = A - A * X
            E = solve_l1l2(temp1 + Y1 / mu, alpha / mu)
            Y1 = Y1 + mu * (temp1 - E)
            Y2 = Y2 + mu * (X - J)
            mu = min(rho * mu, max_mu)
            if LA.norm(temp1 - E, np.inf) < epsilon and LA.norm(X - J, np.inf) < epsilon: break
P = A * X
dt=pd.DataFrame(P.T)
