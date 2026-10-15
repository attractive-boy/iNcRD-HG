"""Select a config by paired training-internal gains, before outer validation."""
import json,csv
import numpy as np
from model import HERE,CONFIGS,sha256


def main():
    plan=json.loads((HERE/'results/plan.json').read_text());results={};flat=[]
    for name in CONFIGS:
        results[name]=[]
        for fold in range(3):
            r=json.loads((HERE/'results'/f'{name}_f{fold}_s42.json').read_text());results[name].append(r)
            flat.append({k:r[k] for k in ('config','fold','inner_auc','inner_ap','best_epoch','lncrna_used','outer_validation_evaluated','test_evaluated')})
    rows=[];base=results['control']
    for name,runs in results.items():
        da=[r['inner_auc']-b['inner_auc'] for r,b in zip(runs,base)]
        dp=[r['inner_ap']-b['inner_ap'] for r,b in zip(runs,base)]
        rows.append(dict(config=name,mean_inner_auc=float(np.mean([r['inner_auc'] for r in runs])),mean_inner_ap=float(np.mean([r['inner_ap'] for r in runs])),
            mean_auc_gain=float(np.mean(da)),mean_ap_gain=float(np.mean(dp)),fold_auc_gains=da,fold_ap_gains=dp,selection_score=min(float(np.mean(da)),float(np.mean(dp)))))
    best=max(rows,key=lambda r:r['selection_score']);promote=best['config']!='control' and best['selection_score']>=plan['outer_promotion_minimum_both_gains']
    result=dict(status='frozen_inner_selection',rows=rows,best=best,promote_to_outer_validation=promote,lncrna_used=False,outer_validation_evaluated=False,test_evaluated=False,
        plan_sha256=sha256(HERE/'results/plan.json'),result_hashes={str(p.relative_to(HERE)):sha256(p) for p in sorted((HERE/'results').glob('*_s42.json'))})
    (HERE/'selection.json').write_text(json.dumps(result,indent=2))
    with (HERE/'inner_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
