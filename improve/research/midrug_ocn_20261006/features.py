"""OCN-inspired 3/5-walk coefficients, exactly excluding the query cell.

This is a bipartite adaptation, not a reproduction of the published OCN model.
No new data, pseudo-edges, external weights or held-out labels are loaded.
"""
import argparse,json
import numpy as np
import torch
from common import HERE,CLEAN,load_inner,sha256

CONFIGS={
    'raw':dict(normalize=False,orthogonalize=False),
    'norm':dict(normalize=True,orthogonalize=False),
    'orth':dict(normalize=False,orthogonalize=True),
    'orth_norm':dict(normalize=True,orthogonalize=True),
}

def adjacency(data,device='cpu'):
    A=torch.zeros(605,216,dtype=torch.float64,device=device)
    p=torch.as_tensor(data['train_pairs'][data['train_y']==1],device=device)
    A[p[:,0],p[:,1]]=1
    return A

@torch.no_grad()
def coefficients(A,pairs):
    """Exact recurrence v B_q = v B - q*v[d]*e_r - q*v[r]*e_d."""
    nr,nd=A.shape;n=nr+nd
    B=torch.zeros(n,n,dtype=A.dtype,device=A.device);B[:nr,nr:]=A;B[nr:,:nr]=A.T
    p=torch.as_tensor(pairs,device=A.device);r,d=p[:,0],p[:,1]+nr
    row=torch.arange(len(p),device=A.device);q=A[p[:,0],p[:,1]]
    def step(v):
        out=v@B
        out[row,r]-=q*v[row,d];out[row,d]-=q*v[row,r]
        return out
    ur=torch.nn.functional.one_hot(r,n).to(A.dtype)
    ud=torch.nn.functional.one_hot(d,n).to(A.dtype)
    mass=torch.ones_like(ur);rs=[ur];ds=[ud];ms=[mass]
    for k in range(3):rs.append(step(rs[-1]));ds.append(step(ds[-1]));ms.append(step(ms[-1]))
    out=[]
    for k in (2,3):
        w=rs[k-1]*ds[k]+rs[k]*ds[k-1]+rs[k]*ds[k]
        w[row,r]=0;w[row,d]=0
        out.append(w)
    return torch.stack(out,1),torch.stack(ms[2:],1)

def transform(raw,mass,cfg):
    w=raw/mass.clamp(min=1) if cfg['normalize'] else raw.clone()
    if cfg['orthogonalize']:
        first,second=w[:,0],w[:,1]
        alpha=(first*second).sum(1)/(first.square().sum(1).clamp(min=1e-20))
        w=torch.stack([first,second-alpha[:,None]*first],1)
    # Pool scale and path magnitude are kept separately for numerical stability.
    scale=w.abs().sum(2,keepdim=True).clamp(min=1)
    summary=torch.cat([torch.log1p(w[:,:,:605].abs().sum(2)),torch.log1p(w[:,:,605:].abs().sum(2))],1)
    return (w/scale).float(),summary.float()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    out=HERE/'cache';out.mkdir(exist_ok=True)
    for fold in range(3):
        data=load_inner(fold);A=adjacency(data,args.device)
        pairs=np.concatenate([data['train_pairs'],data['val_pairs']]);raws=[];masses=[]
        for p in np.array_split(pairs,20):
            raw,mass=coefficients(A,p);raws.append(raw.cpu());masses.append(mass.cpu())
        raw,mass=torch.cat(raws),torch.cat(masses)
        for name,cfg in CONFIGS.items():
            w,s=transform(raw,mass,cfg);path=out/f'{name}_f{fold}.npz'
            if path.exists():raise FileExistsError(path)
            np.savez_compressed(path,pairs=pairs,weights=w.numpy(),summary=s.numpy())
            meta=dict(config=cfg,fold=fold,cache_sha256=sha256(path),features_sha256=sha256(__file__),data_sha256=sha256(CLEAN/'inner_data.npz'))
            path.with_suffix('.json').write_text(json.dumps(meta,indent=2))
        print('prepared',fold,flush=True)

if __name__=='__main__':main()

