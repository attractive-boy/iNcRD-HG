"""Reconstruct the missing data/MD_adjaceny.txt.

The official repo references `data/MD_adjaceny.txt` (integrated heterogeneous
adjacency) but does not ship it, and `Predataprocess/miRTarbase.py::MD_Matrix`
expects missing `data/m-m.txt` (miRNA similarity) and `data/d-d.txt`
(drug similarity).  This script substitutes those two similarity matrices with
Gaussian-kernel similarities of the miRNA-drug association matrix, exactly as
formulated in `Predataprocess/calculating_similarity.py`:

    lambda = 1 / ((1/n) * sum_i ||A_i||^2)
    sim_ij = exp(-lambda * ||A_i - A_j||^2)

M_MM = Gaussian(A)   with A    = M_MD (n=1578)
M_DD = Gaussian(A.T) with A.T  = (n=156)

The heterogeneous matrix is laid out exactly as miRTarbase.py::MD_Matrix:
    [[M_MM, M_MD],
     [M_MD.T, M_DD]]
and written space-delimited with np.savetxt, matching the reader in
dataprocess.py (`pd.read_csv(..., delimiter=' ')`).
"""
import os
import numpy as np
import pandas as pd


def calculate_kernel_bandwidth(A):
    IP_0 = 0.0
    for i in range(A.shape[0]):
        IP_0 += np.square(np.linalg.norm(A[i]))
    lambd = 1.0 / ((1.0 / A.shape[0]) * IP_0)
    return lambd


def calculate_GaussianKernel_sim(A):
    kernel_bandwidth = calculate_kernel_bandwidth(A)
    n = A.shape[0]
    gauss_kernel_sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            gauss_kernel_sim[i][j] = np.exp(
                -kernel_bandwidth * np.square(np.linalg.norm(A[i] - A[j]))
            )
    return gauss_kernel_sim


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)

    df = pd.read_excel(os.path.join(root, 'data', 'miRNA_drug_matrix.xlsx'),
                       index_col=0)
    M_MD = df.values.astype(float)
    M_MM = calculate_GaussianKernel_sim(M_MD)
    M_DD = calculate_GaussianKernel_sim(M_MD.T)
    print('M_MM', M_MM.shape, 'M_MD', M_MD.shape, 'M_DD', M_DD.shape)

    new_matrix = np.zeros((M_MM.shape[0] + M_DD.shape[0],
                           M_MM.shape[1] + M_MD.shape[1]))
    new_matrix[:M_MM.shape[0], :M_MM.shape[1]] = M_MM
    new_matrix[:M_MM.shape[0], M_MM.shape[1]:] = M_MD
    new_matrix[M_MM.shape[0]:, :M_MM.shape[1]] = M_MD.T
    new_matrix[M_MM.shape[0]:, M_MM.shape[1]:] = M_DD

    np.savetxt(os.path.join(root, 'data', 'MD_adjaceny.txt'),
               new_matrix, fmt='%f')
    print('wrote', os.path.join(root, 'data', 'MD_adjaceny.txt'),
          new_matrix.shape)


if __name__ == '__main__':
    main()
