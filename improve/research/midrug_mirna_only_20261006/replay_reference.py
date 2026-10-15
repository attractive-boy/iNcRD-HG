"""Replay already-evaluated pure-miRNA reference checkpoints on the clean graph."""
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from data import HERE,sha256,make_graph,assert_clean
from incrd_hg_m123.data import load_fold
from incrd_hg_m123.adapter import Adapter
from incrd_hg_m123.engine import seed_all


def main():
    data=load_fold('MiDrug',0);old=HERE.parent/'midrug_ranking_20261006/results';rows=[]
    for cfg in ('bce','bpr010_mixed'):
        for seed in (42,43,44):
            tag=f'{cfg}_s{seed}';reference=json.loads((old/'test'/f'{tag}.json').read_text())
            cp=old/'validation'/f'{tag}.pt';assert sha256(cp)==reference['checkpoint_sha256']
            seed_all(seed);a=Adapter(data,make_graph(data),'cuda:0');assert_clean(a.graph)
            a.model.load_state_dict(torch.load(cp,map_location='cuda:0',weights_only=True));a.model.eval()
            assert all('lnc' not in key.lower() for key in a.model.state_dict())
            with torch.no_grad():p=a.predict('test')
            metrics=dict(auc=roc_auc_score(data['test_y'],p),ap=average_precision_score(data['test_y'],p))
            assert abs(metrics['auc']-reference['metrics']['auc'])<1e-6
            assert abs(metrics['ap']-reference['metrics']['aupr'])<1e-6
            rows.append(dict(config=cfg,seed=seed,checkpoint_sha256=sha256(cp),**metrics))
            print('PASS reference',tag,metrics,flush=True);del a
    means={cfg:{key:float(np.mean([r[key] for r in rows if r['config']==cfg])) for key in ('auc','ap')} for cfg in ('bce','bpr010_mixed')}
    result=dict(status='passed',scope='Replay of six already-evaluated pure-miRNA reference checkpoints; no new candidate test selection',
        records=rows,means=means,new_candidate_test_evaluations=0,lncrna_used=False,clean_graph_sha256=sha256(HERE/'graph.pt'))
    (HERE/'reference_replay.json').write_text(json.dumps(result,indent=2));print(json.dumps(means),flush=True)

if __name__=='__main__':main()
