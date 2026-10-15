"""Frozen-checkpoint diagnostics; no fitting, selection, or original test access."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score,average_precision_score
HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'midrug_walk_kernel_20261007'
sys.path.insert(0,str(PREVIOUS))
from common import load_inner,seed_all,sha256,base
from model import initialize,sym_normalize

@torch.no_grad()
def decomposition(m,batch,xr,xd):
    k=m.subgraph;rid,did=batch['rid'],batch['did'];mask=torch.cat([rid<605,did<216],1)
    r,d=k.rproj(xr),k.dproj(xd);r=torch.cat([r,torch.zeros_like(r[:1])]);d=torch.cat([d,torch.zeros_like(d[:1])])
    x=k.norm(k.role(batch['features'])+torch.cat([r[rid],d[did]],1))*mask[:,:,None]
    s=torch.sigmoid(torch.einsum('bnf,ghf->bghn',x,k.prototypes)/4)
    types=torch.zeros((4,32),dtype=torch.bool,device=x.device);types[:2,:16]=True;types[2:,16:]=True
    s=s*mask[:,None,None,:]*types[None,None]
    pos=batch['pos'];zero=torch.zeros_like(pos)
    a=sym_normalize(torch.cat([torch.cat([zero,pos],-1),torch.cat([pos.transpose(-1,-2),zero],-1)],-2));h=k.hidden_graph()
    full=s.clone();root=s.clone();root[:,:,:,1:]=0
    reverse=s.clone();reverse[:,:,:,:16]=0;reverse[:,:,:,17:]=0
    forward_values=[];reverse_values=[];full_values=[]
    for step in range(3):
        def update(v):return s*torch.einsum('ghi,bgin->bghn',h,torch.matmul(v,a[:,None]))
        full,root,reverse=update(full),update(root),update(reverse)
        forward_values.append(root[:,:,:,16].sum(-1));reverse_values.append(reverse[:,:,:,0].sum(-1));full_values.append(full.sum((-1,-2)))
    forward=torch.stack(forward_values,-1);reverse=torch.stack(reverse_values,-1);total=torch.stack(full_values,-1)
    torch.testing.assert_close(forward,reverse,atol=1e-6,rtol=1e-5)
    assert forward[:,:,:2].abs().max()==0
    fraction=(forward[:,:,2]+reverse[:,:,2])/total[:,:,2].clamp_min(1e-30)
    assert (fraction>=0).all() and (fraction<=1+1e-5).all()
    # An explicit enumeration of all simple r0-dj-ri-d0 paths for sampled queries.
    errors=[]
    for ix in [0,1,len(rid)-1]:
        g=0;manual=s.new_zeros(())
        for di in range(17,32):
            for ri in range(1,16):
                path_weight=a[ix,0,di]*a[ix,di,ri]*a[ix,ri,16]
                if path_weight==0:continue
                for u in [0,1]:
                    for v in [2,3]:
                        for w in [0,1]:
                            for t in [2,3]:
                                manual+=path_weight*h[g,u,v]*h[g,v,w]*h[g,w,t]*s[ix,g,u,0]*s[ix,g,v,di]*s[ix,g,w,ri]*s[ix,g,t,16]
        torch.testing.assert_close(manual,forward[ix,0,2],atol=1e-6,rtol=1e-5);errors.append(float(abs(manual-forward[ix,0,2])))
    nonzero=forward[:,:,2].sum(1)>0
    return dict(two_root_fraction_mean=float(fraction.mean()),two_root_fraction_median=float(fraction.median()),
        two_root_fraction_mean_if_path=float(fraction[nonzero].mean()) if nonzero.any() else 0.,
        queries_with_connecting_path_fraction=float(nonzero.float().mean()),direct_enumeration_errors=errors,
        first_two_steps_connecting_mass_zero=True,forward_reverse_equal=True)

def metric(y,p):return dict(auc=roc_auc_score(y,p),ap=average_precision_score(y,p))

def main():
    manifest=json.loads((PREVIOUS/'archive_manifest.json').read_text());checks=0
    for f,h in manifest['artifact_hashes'].items():assert sha256(PREVIOUS/f)==h;checks+=1
    for f,h in manifest['frozen_dependencies'].items():assert sha256(f)==h;checks+=1
    results=[]
    for name in ['pathwise','endpoints','fixed','no_edges','mpnn']:
        for fold in range(3):
            seed_all(42);data=load_inner(fold);a,_,_=initialize(data,name,fold,'cuda:0');m=a.model
            p=PREVIOUS/'results'/f'{name}_f{fold}_s42';m.load_state_dict(torch.load(p.with_suffix('.pt'),map_location='cuda:0',weights_only=True));m.eval()
            tensors={}
            with torch.no_grad():
                for split in ['train','val']:
                    batch=m.cache.select(*a.pairs[split])
                    if name=='mpnn':batch={**batch,'neg':torch.zeros_like(batch['pos'])}
                    branch=m.subgraph(batch,*a.dense[:2]);plain=base.Model.forward(m,a.pairs[split],a.ps,a.qis[split],a.ease[split],a.dense)
                    tensors[split]=(branch,plain,batch)
            tr=tensors['train'][0];va,plain,batch=tensors['val'];bias=float(tr.mean())
            actual=(plain+va).sigmoid().cpu().numpy();bias_only=(plain+bias).sigmoid().cpu().numpy()
            with np.load(p.with_name(p.name+'_inner.npz')) as z:np.testing.assert_allclose(actual,z['probability'],atol=2e-5,rtol=2e-5)
            raw_rms=float(va.square().mean().sqrt());centered=float(va.std(unbiased=False))
            r=dict(config=name,fold=fold,checkpoint_sha256=sha256(p.with_suffix('.pt')),train_branch_mean=bias,
                val_branch_mean=float(va.mean()),val_branch_rms=raw_rms,val_branch_std=centered,
                centered_fraction_of_second_moment=centered**2/max(raw_rms**2,1e-30),actual=metric(data['val_y'],actual),
                training_mean_bias_intervention=metric(data['val_y'],bias_only),
                interpretation='A fixed additive bias is monotone and cannot improve exact ranking; float32 sigmoid ties may cause tiny numerical metric differences.')
            if name=='pathwise':r['path_mass']=decomposition(m,batch,*a.dense[:2])
            results.append(r);print('PASS',name,fold,flush=True);del a,m,tensors;torch.cuda.empty_cache()
    out=dict(status='passed',records=results,source_sha256=sha256(__file__),source_archive_sha256=sha256(PREVIOUS/'archive_manifest.json'),
        source_selection_sha256=sha256(PREVIOUS/'selection.json'),prior_archive_hash_checks=checks,no_new_fits=True,
        original_validation_evaluated=False,test_evaluated=False,no_lncrna=True)
    (HERE/'diagnostics.json').write_text(json.dumps(out,indent=2))

if __name__=='__main__':main()
