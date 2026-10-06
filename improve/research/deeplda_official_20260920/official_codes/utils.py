import numpy as np
import scipy.sparse as sp
import torch
import math
import argparse
from sklearn.metrics import precision_recall_curve,roc_curve, accuracy_score,f1_score,auc,precision_score,recall_score,matthews_corrcoef


def parameter_parser():
    parser = argparse.ArgumentParser(description="Run GAT-LGA.")
    parser.add_argument("--dataset_path",nargs="?",default="../Datasets/Dataset2/",help="Training datasets.")#Dataset1,Dataset2
    parser.add_argument("--result_path",nargs="?",default="../Results/2/",help="Training datasets.")#1,2
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
    parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=1050, help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--nhid', type=int, default=50, help='Number of hidden units.')
    parser.add_argument('--nclass', type=int, default=200, help='Number of hidden units.')
    parser.add_argument('--nheads', type=int, default=8, help='Number of head attentions.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    parser.add_argument('--patience', type=int, default=1000, help='Patience')
    return parser.parse_args()
    
def get_edge_index(matrix):
    edge_index = [[], []]
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] > 0.5:
                edge_index[0].append(i)
                edge_index[1].append(j)
    return torch.tensor(np.array(edge_index))
    
def get_adj(matrix):
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i][j] > 0.5:
                matrix[i][j] = 1
            else:
                matrix[i][j] = 0
    return torch.tensor(matrix)

def gaussian_sim(data_list):
    '''
    calculate the gaussian similarity
    '''
    print("Similarity calculation!")
    nl=data_list.shape[0]
    nd=data_list.shape[1]
    sl=[0]*nl
    sd=[0]*nd
    pkl=np.zeros((nl, nl))
    pkd=np.zeros((nd, nd))
    
    X = np.array(data_list)
    Y = np.array(data_list)
    D = np.sum(X*X, axis=1, keepdims=True) \
            + np.sum(Y*Y, axis=1, keepdims=True).T \
            - 2 * np.dot(X, Y.T)
    for i in range(nl):
        sl[i]=pow(np.linalg.norm(data_list[i,:]),2)
    gamal=sum(sl)/nl
    pkl=np.exp(-gamal*D)
    
    X = np.array(data_list.T)
    Y = np.array(data_list.T)
    D2 = np.sum(X*X, axis=1, keepdims=True) \
            + np.sum(Y*Y, axis=1, keepdims=True).T \
            - 2 * np.dot(X, Y.T)
    for i in range(nd):
        sd[i]=pow(np.linalg.norm(data_list[:,i]),2)
    gamad=sum(sd)/nd 
    pkd=np.exp(-gamad*D2)
    print("Finish similarity calculation!")
    #np.savetxt('../Datasets/Dataset1/lnc_feat.txt',pkl,fmt="%.18f",delimiter="\t")
    #np.savetxt('../Datasets/Dataset1/drug_feat.txt',pkd,fmt="%.18f",delimiter="\t")
    return pkl, pkd

def load_data(args):
    print('Loading {}...'.format(args.dataset_path))
    
    LG_matrix = np.loadtxt(open(args.dataset_path + 'lnc_drug_net.csv',"rb"),delimiter=",",skiprows=0)   
    reorder = np.arange(LG_matrix.shape[0])
    np.random.shuffle(reorder)
    LG_matrix =  LG_matrix[reorder,:]
    
    #labels=LG_matrix    
    #l_feat=np.hstack((l_m_feat,l_s_feat))
    #g_feat=np.hstack((g_m_feat,g_s_feat))
    #F1=l_s_feat
    #F2=g_s_feat 
    l_f_feat,g_f_feat=gaussian_sim(LG_matrix)
    
    
    '''print(l_m_feat.shape)
    print(g_m_feat.shape)
    print(l_f_feat.shape)
    print(g_f_feat.shape)'''
    #print(l_feat.shape)
    #print(g_feat.shape)
    
    Lnc_f_edge_index = get_edge_index(l_f_feat)
    Gene_f_edge_index = get_edge_index(g_f_feat)    
    Lnc_f_adj=get_adj(l_f_feat)
    Gene_f_adj=get_adj(g_f_feat)     
    
    
    #adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32) #论文间的邻接矩阵
    
    cv=10
    num=int(LG_matrix.shape[0]/cv)
    
    idx_train = range((cv-1)*num)
    #idx_val = range((cv-4)*num, (cv-2)*num)
    idx_val = range((cv-1)*num, LG_matrix.shape[0])
    idx_test = range((cv-1)*num, LG_matrix.shape[0])


    Lnc_f_features = torch.FloatTensor(l_f_feat)
    Gene_f_features = torch.FloatTensor(g_f_feat)
    
    LG_matrix = torch.FloatTensor(LG_matrix)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    dataset = dict()
    #Lnc_adj=Lnc_f_adj+Lnc_m_adj
    #Gene_adj=Gene_f_adj+Gene_m_adj
    
    
    dataset['Lnc_f_edge_index']=Lnc_f_edge_index
    dataset['Lnc_f_adj']=Lnc_f_adj
    dataset['Lnc_f_features']=Lnc_f_features
    dataset['Gene_f_edge_index']=Gene_f_edge_index
    dataset['Gene_f_adj']=Gene_f_adj
    dataset['Gene_f_features']=Gene_f_features
    
    dataset['labels']=LG_matrix 
    dataset['idx_train']=idx_train 
    dataset['idx_val']=idx_val 
    dataset['idx_test']=idx_test
    return dataset
  
def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def metrics(y_score, y_true):
    y_score=y_score.view(1,-1).detach().cpu().numpy().flatten().tolist()
    y_true=y_true.view(1,-1).detach().cpu().numpy().flatten().tolist()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auroc=auc(fpr, tpr)
    aupr=auc(recall, precision)
    #acc=accuracy_score(y_true, np.rint(y_score))
    F1=f1_score(y_true, np.rint(y_score), average='macro')
    #Pre=precision_score(y_true, np.rint(y_score), average='macro')
    #Rec=recall_score(y_true, np.rint(y_score), average='macro')
    Mcc=matthews_corrcoef(y_true,np.rint(y_score))
    return auroc,aupr,F1,Mcc
    
def Auc(y_score, y_true):
    y_score=y_score.view(1,-1).detach().cpu().numpy().flatten().tolist()
    y_true=y_true.view(1,-1).detach().cpu().numpy().flatten().tolist()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auroc=auc(fpr, tpr)
    return auroc

