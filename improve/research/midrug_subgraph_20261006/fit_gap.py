"""Measure the pure-M123 fitting gap using only original-training inner folds."""
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss
from model import HERE,CLEAN,load_inner,make_adapter,seed_all,sha256


def main():
    rows=[]
    for fold in range(3):
        data=load_inner(fold);path=CLEAN/'results'/f'control_f{fold}_s42.json';meta=json.loads(path.read_text())
        assert sha256(path.with_suffix('.pt'))==meta['checkpoint_sha256']
        seed_all(42);a=make_adapter(data,'control',fold,'cuda:0')
        a.model.load_state_dict(torch.load(path.with_suffix('.pt'),map_location='cuda:0',weights_only=True));a.model.eval()
        with torch.no_grad():p=a.predict('train')
        saved=np.load(path.with_name(path.stem+'_inner.npz'))
        row=dict(fold=fold,train_auc=roc_auc_score(data['train_y'],p),train_ap=average_precision_score(data['train_y'],p),
            inner_auc=meta['inner_auc'],inner_ap=meta['inner_ap'],train_logloss=log_loss(data['train_y'],p),inner_logloss=log_loss(saved['y'],saved['probability']))
        rows.append(row);print(json.dumps(row),flush=True);del a
    result=dict(status='measured',records=rows,scope='Training-internal fitting gap; descriptive, not a causal proof of residual training interference',
        mean={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='fold'},no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    (HERE/'fit_gap.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
