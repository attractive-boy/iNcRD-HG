"""Read-only gradient diagnostic at nine fixed reconstruction checkpoints."""
import json,sys
from pathlib import Path
import torch
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent/'midrug_training_size_20261007'
sys.path.insert(0,str(STUDY))
from common import load_subset,seed_all,sha256
from model import make_adapter,mask_generator


def examine(row):
    data=load_subset(row['fold'],row['size']);seed_all(42);a=make_adapter(data,'unmasked10','cuda:0')
    path=Path(row['metadata_path']).with_suffix('.pt');assert sha256(path)==row['checkpoint_sha256']
    a.model.load_state_dict(torch.load(path,map_location='cuda:0',weights_only=True));m=a.model
    before={k:v.detach().clone() for k,v in m.state_dict().items()};m.train()
    generator=mask_generator('cuda:0',20261006+row['fold'])
    # Reconstruct the private auxiliary sampler's next step after best_epoch.
    for _ in range(row['best_epoch']+1):indices=m.reconstruction.masks(a.graph,.5,generator)
    bce=F.binary_cross_entropy_with_logits(a.logits('train'),a.y)
    auxiliary,_=m.reconstruction.loss(m.encoder,a.graph,indices,False)
    shared=[p for p in m.encoder.parameters() if p.requires_grad]
    primary=torch.autograd.grad(bce,shared,allow_unused=True)
    extra=torch.autograd.grad(10*auxiliary,shared,allow_unused=True)
    g=torch.cat([(torch.zeros_like(p) if v is None else v).detach().flatten().double() for p,v in zip(shared,primary)])
    h=torch.cat([(torch.zeros_like(p) if v is None else v).detach().flatten().double() for p,v in zip(shared,extra)])
    assert torch.isfinite(g).all() and torch.isfinite(h).all()
    gn,hn=g.norm(),h.norm();dot=g@h
    coefficient=(dot/gn.square()).clamp(max=0) if gn>0 else dot.new_zeros(())
    projected=h-coefficient*g
    result=dict(config='unmasked10',size=row['size'],fold=row['fold'],train_count=row['train_count'],checkpoint_sha256=sha256(path),
        bce=float(bce),unweighted_reconstruction=float(auxiliary),primary_norm=float(gn),weighted_auxiliary_norm=float(hn),
        auxiliary_to_primary_norm_ratio=float(hn/gn) if gn>0 else None,cosine=float(dot/(gn*hn)) if gn*hn>0 else None,
        conflicting=bool(dot<0),shared_parameter_count=g.numel(),
        projected_auxiliary_norm_fraction=float(projected.norm()/hn) if hn>0 else 1.,
        projected_primary_dot=float(g@projected),selected_epoch=row['best_epoch'],mask_step=row['best_epoch']+1)
    m.load_state_dict(before);m.eval()
    for key,value in before.items():torch.testing.assert_close(value,m.state_dict()[key],atol=0,rtol=0)
    assert sha256(path)==row['checkpoint_sha256']
    del a,m,g,h,projected,primary,extra,before;torch.cuda.empty_cache()
    return result


def main():
    archive=STUDY/'archive_recheck.json';assert json.loads(archive.read_text())['status']=='passed'
    verification=STUDY/'verification.json';rows=json.loads(verification.read_text())['records']
    records=[]
    for row in rows:
        if row['config']=='unmasked10':
            result=examine(row);records.append(result);print('PASS',row['size'],row['fold'],result['cosine'],flush=True)
    assert len(records)==9
    source=[Path(__file__),STUDY/'common.py',STUDY/'model.py',STUDY/'assessment.json',archive,verification,
            HERE.parent.parent/'conflict_free_graph_training.py',HERE.parent/'new_mechanisms_20260915/READING_NOTES.md']
    result=dict(status='passed',records=records,source_hashes={str(p):sha256(p) for p in source},
        optimizer_updates=0,no_heldout_scoring=True,no_checkpoint_or_buffer_updates=True,no_lncrna=True,
        limitation='This measures raw shared-encoder gradients at selected checkpoints in training mode with the next deterministic auxiliary mask. It does not identify a cause of generalization error, measure the actual RMSprop update, or establish a benefit from projection.')
    (HERE/'gradient_precheck.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
