# -*- coding: utf-8 -*-
"""HAWKEYE-style diagnostic on PLMF-MDA: is 1-hop/degree structure discriminative,
or is the deeper (length-3/5) cohesive bridge the real signal? Training-free discAUC."""
import sys, os, numpy as np
from sklearn.metrics import roc_auc_score
BASE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(BASE)
sys.path.insert(0, os.path.join(ROOT,'mphgnn_plmf_20260921'))
from plmf_splits import load_dataset, fold_5cv
from cold_probe import cold_split

def build(pos, nr, nd):
    A=np.zeros((nr,nd),np.float64); A[pos[:,0],pos[:,1]]=1; return A

def feats(A, q):
    r,d=q[:,0],q[:,1]
    AA=A@A.T                      # miRNA-miRNA shared drugs
    Ad=A.T@A                      # drug-drug shared miRNAs
    L3=(AA@A)[r,d]                # length-3 paths r-d'-r'-d  (2-hop cohesive bridge)
    L5=(AA@A@A.T@A)[r,d]           # length-5 paths
    degr=A.sum(1)[r]; degd=A.sum(0)[d]
    return dict(deg_prod=degr*degd, deg_r=degr, deg_d=degd, L3=L3,
                L3_norm=L3/np.maximum(degr*degd,1), L5=L5, L5_norm=L5/np.maximum(degr*degd,1))

for ds in ('MDR','MDS'):
    mir,dru,pos,neg=load_dataset(ds); nr,nd=len(mir),len(dru)
    # warm 5cv fold0
    tr,ytr,te,yte=fold_5cv(ds,0,seed=42)
    A=build(tr[ytr==1],nr,nd); f=feats(A,te)
    print(f'== {ds} warm 5cv fold0 (n={len(yte)}) ==')
    for k,v in f.items(): print(f'   {k:9s} discAUC {roc_auc_score(yte,v):.4f}')
    # cold-drug (deg 11-20)
    ddeg=build(pos,nr,nd).sum(0); test_drugs=np.where((ddeg>=11)&(ddeg<=20))[0]
    tr,ytr,te,yte=cold_split(dict(nr=nr,nd=nd,pos=pos),test_drugs,0)
    A=build(tr[ytr==1],nr,nd); f=feats(A,te)
    print(f'== {ds} cold-drug deg11-20 (n={len(yte)}) ==')
    for k,v in f.items(): print(f'   {k:9s} discAUC {roc_auc_score(yte,v):.4f}')
