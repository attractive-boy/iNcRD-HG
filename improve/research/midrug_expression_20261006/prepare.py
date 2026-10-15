"""Provenance-checked transformations of original RNA expression only."""
import sys,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
HERE=Path(__file__).resolve().parent
RELEASE=HERE.parents[2]/'release/iNcRD-HG-M1M2M3'
sys.path.insert(0,str(RELEASE))
from incrd_hg_m123.data import load_graph,sha256
ORIGINAL=HERE.parents[3]/'incrd-orig'


def main():
    g=load_graph('MiDrug');arrays={};proof={};cells={}
    for kind,file in [('miRNA','MiExp.csv'),('lncRNA','LncExp.csv')]:
        rawpath=ORIGINAL/'2.construct_network'/file
        infopath=ORIGINAL/'3.heter_data'/f'{kind}_info.pth'
        raw=pd.read_csv(rawpath,index_col=0).T
        info=torch.load(infopath,map_location='cpu')[kind+'_feature']
        # Reorder by explicit entity and cell identifiers; no association labels.
        aligned=raw.loc[info.index,info.columns]
        np.testing.assert_array_equal(aligned.values,info.values)
        X=aligned.values.astype(np.float32)
        S=(X-X.min(1,keepdims=True))/(np.ptp(X,axis=1,keepdims=True)+1e-8)
        np.testing.assert_array_equal(S,g[kind].x.numpy())
        cells[kind]=list(info.columns)
        arrays[kind+'_original']=S
        arrays[kind+'_global']=(X-X.min())/(np.ptp(X)+1e-8)
        arrays[kind+'_column']=(X-X.min(0,keepdims=True))/(np.ptp(X,axis=0,keepdims=True)+1e-8)
        arrays[kind+'_rowz']=(X-X.mean(1,keepdims=True))/(X.std(1,keepdims=True)+1e-4)
        arrays[kind+'_columnz']=(X-X.mean(0,keepdims=True))/(X.std(0,keepdims=True)+1e-4)
        stats=np.stack([X.mean(1),X.std(1),X.min(1),X.max(1)],1)
        stats=(stats-stats.mean(0))/(stats.std(0)+1e-6)
        arrays[kind+'_stats']=np.concatenate([S,stats],1)
        assert all(np.isfinite(v).all() for v in arrays.values())
        proof[kind]=dict(raw_sha256=sha256(rawpath),source_info_sha256=sha256(infopath),
            node_count=len(X),sample_count=X.shape[1],frozen_feature_exact_reconstruction=True,
            raw_column_order_matches_frozen=list(raw.columns)==list(info.columns))
    assert cells['miRNA']==cells['lncRNA']
    np.savez_compressed(HERE/'expression.npz',**arrays)
    result=dict(status='passed',records=proof,shared_sample_ids=cells['miRNA'],
        data_sources='Original iNcRD-HG RNA expression only; no added datasets or labels',
        expression_sha256=sha256(HERE/'expression.npz'),source_sha256=sha256(__file__))
    (HERE/'provenance.json').write_text(json.dumps(result,indent=2))
    print('PASS exact reconstruction of frozen RNA features; shared 938-cell alignment; 6 label-free representations per RNA type',flush=True)

if __name__=='__main__':main()
