
import numpy as np
import pandas as pd
import numpy.linalg as LA
import scipy.linalg as LG
import os
def solve_l1l2(W,lambda1):
    nv=W.shape[1]
    F=W.copy()
    for p in range(nv):
        nw=LA.norm(W[:,p],"fro")
        if nw>lambda1:
            F[:,p]=(nw-lambda1)*W[:,p]/nw
        else:F[:,p]=np.zeros((W[:,p].shape[0],1))
    return F
def matrix_rowSum(mat):
    mat_new = np.zeros(shape=(mat.shape[0],mat.shape[1]))
    for i in range(mat.shape[0]):
        mat_new[i,i] = np.sum(mat[i,])
    return mat_new

def matrix_AS(mat_A,mat_S):
    mat_n = np.zeros(shape=(mat_S.shape[0],mat_S.shape[1]))
    
    for i in range(mat_A.shape[0]):
        for j in range(mat_S.shape[1]):
            mat_n[i,j] = np.sum(mat_A[i,])
    return mat_n

def max_min(matr):
    matr_max = np.max(matr)
    matr_min = np.min(matr)
    matr = (matr-matr_min)/(matr_max-matr_min)
    return matr


def read_data(path):

    data = []
    for line in open(path, 'r'):
        ele = line.strip().split(" ")
        tmp = []
        for e in ele:
            if e != '':
                tmp.append(float(e))
        data.append(tmp)
    return data
if __name__ == '__main__':
    
    data_path = os.path.join(os.path.dirname(os.getcwd()),"data")
a=np.mat(np.zeros((541,831)))#Define a 541*831 matrix of zeros
AA=a.astype(int)#Convert matrix AA to integer
#Read the known SM-miRNA associations
b=np.loadtxt('')
F=b.astype(int)
#Change the corresponding position element of matrix AA to 1 to obtain the adjacency matrix AA
for x in F:
    AA[x[0]-1,x[1]-1]=1
    #Matrix decomposition
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
    [U, sigma1, V] = LG.svd(X + Y2 / mu, lapack_driver='gesvd')
    G = [sigma1[k] for k in range(len(sigma1)) if sigma1[k] > 1 / mu]
    svp = len(G)
    if svp >= 1:
        sigma1 = sigma1[0:svp] - 1 / mu
    else:
        sigma1 = [0]
        svp = 1
    J = np.mat(U[:, 0:svp]) * np.mat(np.diag(sigma1)) * np.mat(V[0:svp, :])
    ATA = AA.T * AA
    X = (ATA + np.eye(831)).I * (ATA - AA.T * E + J + (AA.T * Y1 - Y2) / mu)
    temp1 = AA - AA * X
    E = solve_l1l2(temp1 + Y1 / mu, alpha / mu)
    Y1 = Y1 + mu * (temp1 - E)
    Y2 = Y2 + mu * (X - J)
    mu = min(rho * mu, max_mu)
    if LA.norm(temp1 - E, np.inf) < epsilon and LA.norm(X - J, np.inf) < epsilon: break
P= AA * X

    # Fac
data_dm = P.T
data_Dsim = read_data(data_path+'\\ssmdataset.txt')
data_Msim = read_data(data_path+'\\smidataset.txt') 
data_dm = np.array(data_dm)
data_Dsim = np.array(data_Dsim)
data_Msim = np.array(data_Msim)
Y =  np.array(data_dm)
A = np.random.random(size=(Y.shape[0],120))
B = np.random.random(size=(Y.shape[1],120))
alpha = 0.1
beta = 0.01
delta = 0.1   
for i in range(50):
    A = A*((Y.dot(B)+alpha*data_Dsim.dot(A))/(A.dot(np.transpose(B)).dot(B)+alpha*matrix_rowSum(data_Dsim).dot(A)+delta*A))
    B = B*(((np.transpose(Y)).dot(A)+beta*data_Msim.dot(B))/(B.dot(np.transpose(A)).dot(A)+beta*matrix_rowSum(data_Msim).dot(B)+delta*B))
        #Snew = Snew*((alpha*A.dot(np.transpose(A)))/(alpha*matrix_AS(A,Ss)+2*delta*Snew))
A = max_min(A)
B = max_min(B)
C = np.dot(A,B.T)
dt=pd.DataFrame(C)
dt.to_excel('case.xlsx')
