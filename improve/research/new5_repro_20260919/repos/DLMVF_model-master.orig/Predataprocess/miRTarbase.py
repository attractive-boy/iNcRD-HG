import pandas as pd
import numpy as np

def association_matrix():
    df = pd.read_excel('data/miRTarbase.xlsx')
    df1=pd.read_excel('data/miRNA_gen.xlsx')
    # 提取两列数据
    miRNA_list = df1['miRNA_name'].unique()  # miRNA 列，唯一值
    df['Target Gene'] = df['Target Gene'].str.upper()
    Gene = df['Target Gene'].unique()    # drug 列，唯一值

    # 创建一个空的关联矩阵，行是 miRNA，列是 drug
    association_matrix = pd.DataFrame(0, index=miRNA_list, columns=Gene)

    # 填充关联矩阵，将关联的位置设置为1
    for i, row in df.iterrows():
        miRNA = row['miRNA']
        TemGene = row['Target Gene']
        association_matrix.loc[miRNA,TemGene] = 1
    association_matrix.to_excel('data/association_matrix.xlsx', index=True)



def MD_Matrix():
    M_MM = np.loadtxt(r"../data/m-m.txt", dtype=float)
    df=pd.read_excel('data/miRNA_drug_matrix.xlsx', index_col=0) # n x d 维度
    M_MD=df.values
    M_DD = np.loadtxt(r"../data/d-d.txt", dtype=float)  # n x d 维度
     # d x d 维度

    # 创建新矩阵，初始化为零
    New_Matrix = np.zeros((M_MM.shape[0] + M_DD.shape[0], M_MM.shape[1] + M_MD.shape[1]))

    # 填充新矩阵
    New_Matrix[:M_MM.shape[0], :M_MM.shape[1]] = M_MM
    New_Matrix[:M_MM.shape[0], M_MM.shape[1]:] = M_MD
    New_Matrix[M_MM.shape[0]:, :M_MM.shape[1]] = M_MD.T
    New_Matrix[M_MM.shape[0]:, M_MM.shape[1]:] = M_DD
    np.savetxt("../data/MD_adjaceny.txt", New_Matrix, fmt='%f')

if __name__ == '__main__':
    MD_Matrix()
    #association_matrix()
    # with open('data/m-m.txt') as f:
    #     MD_Matrix = np.loadtxt(f)
    #     print(MD_Matrix.shape)