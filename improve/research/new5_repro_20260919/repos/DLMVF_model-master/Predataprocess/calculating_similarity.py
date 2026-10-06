"Gaussiankernel similarity"
import numpy as np
import pandas as pd
def calculate_kernel_bandwidth(A):
    IP_0 = 0
    for i in range(A.shape[0]):
        IP = np.square(np.linalg.norm(A[i]))
        # print(IP)
        IP_0 += IP
    lambd = 1/((1/A.shape[0]) * IP_0)
    return lambd

def calculate_GaussianKernel_sim(A):
    kernel_bandwidth = calculate_kernel_bandwidth(A)
    gauss_kernel_sim = np.zeros((A.shape[0],A.shape[0]))
    for i in range(A.shape[0]):
        for j in range(A.shape[0]):
            gaussianKernel = np.exp(-kernel_bandwidth * np.square(np.linalg.norm(A[i] - A[j])))
            gauss_kernel_sim[i][j] = gaussianKernel
            # print("gau",gauss_kernel_sim)

    return gauss_kernel_sim

df = pd.read_excel('data/association_matrix.xlsx', index_col=0)
A=df.to_numpy()
A_T = A.T

# 计算高斯核相似性矩阵
mm = calculate_GaussianKernel_sim(A)
dd= calculate_GaussianKernel_sim(A_T)
print(mm.shape)

np.savetxt("../data/mg-mg.txt", mm, fmt='%.4f')
