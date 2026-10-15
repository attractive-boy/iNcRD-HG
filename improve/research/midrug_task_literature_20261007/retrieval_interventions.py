"""Post-selection fixed-checkpoint diagnosis, never a new candidate selection."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score,average_precision_score
HERE=Path(__file__).resolve().parent
ROUND=HERE.parent/'midrug_label_retrieval_20261007'
sys.path.insert(0,str(ROUND))
from common import load_inner,seed_all,sha256
from model import make_adapter

@torch.no_grad()
def main():
    sp=ROUND/'selection.json';selected=json.loads(sp.read_text());assert selected['status']=='frozen_inner_selection'
    records=[]
    for fold in range(3):
        data=load_inner(fold);seed_all(42);a=make_adapter(data,'label','cuda:0');path=ROUND/'results'/f'label_f{fold}_s42.pt'
        state=torch.load(path,map_location='cuda:0',weights_only=True);a.model.load_state_dict(state);a.model.eval()
        before={k:v.clone() for k,v in a.model.state_dict().items()};m=a.model.case_memory;q=a.pairs['val']
        full=a.logits('val');delta=m(q);base=full-delta
        config=m.config;m.config={**config,'labels':False};without_label=base+m(q);m.config=config
        h,_,ix,weights=m.context(q);nodes=m.nodes();keys={n:F.linear(v,m.K.weight,None) for n,v in nodes.items()}
        qk=keys['miRNA'][q[0]]+keys['drug'][q[1]]+m.K.bias;spairs=m.support_pairs
        sk=keys['miRNA'][spairs[:,0]]+keys['drug'][spairs[:,1]]+m.K.bias
        values=m.T(qk[:,None]-sk[ix])+m.label(m.support_y[ix])
        uniform=base+m.head(h+values.mean(1)).squeeze(-1)
        logits={'full':full,'without_label_value':without_label,'without_branch':base,'uniform_selected_weights':uniform}
        r=dict(fold=fold,checkpoint_sha256=sha256(path),metrics={},mean_abs_probability_changes={})
        probabilities={n:torch.sigmoid(x).cpu().numpy() for n,x in logits.items()}
        for n,p in probabilities.items():
            r['metrics'][n]=dict(auc=roc_auc_score(data['val_y'],p),ap=average_precision_score(data['val_y'],p))
            r['mean_abs_probability_changes'][n]=float(abs(p-probabilities['full']).mean())
        with np.load(ROUND/'results'/f'label_f{fold}_s42_inner.npz') as z:np.testing.assert_allclose(probabilities['full'],z['probability'],atol=2e-5,rtol=2e-5)
        for k,v in a.model.state_dict().items():torch.testing.assert_close(v,before[k],atol=0,rtol=0)
        assert m.config==config;records.append(r)
    means={n:{metric:float(np.mean([r['metrics'][n][metric] for r in records])) for metric in ['auc','ap']} for n in logits}
    result=dict(status='passed',selection_sha256=sha256(ROUND/'selection.json'),source_sha256=sha256(HERE/'retrieval_interventions.py'),records=records,means=means,
        all_parameters_and_buffers_unchanged=True,new_fits=0,no_candidate_selection=True,outer_validation_evaluated=False,test_evaluated=False)
    (HERE/'retrieval_interventions.json').write_text(json.dumps(result,indent=2));print(json.dumps(means,indent=2))

if __name__=='__main__':main()
