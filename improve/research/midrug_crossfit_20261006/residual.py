"""Learn residuals only from nested OOF base logits and nested graph contexts."""
import argparse,json,time
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,CLEAN,SG,SubgraphGNN,load_inner,seed_all,sha256

CONFIGS={
    'offset_l001':dict(offset=True,penalty=.01),
    'offset_l01':dict(offset=True,penalty=.1),
    'independent25':dict(offset=False,penalty=0.,inference_weight=.25),
}
KEYS=('rid','did','features','pos','neg')

def load_context(fold,device):
    outer=load_inner(fold);n=len(outer['train_y']);filled=np.zeros(n,dtype=int)
    rows={};offset=np.zeros(n,dtype=np.float32)
    hashes={}
    for k in range(5):
        path=HERE/'bases'/f'base_f{fold}_k{k}.npz';meta=json.loads(path.with_suffix('.json').read_text())
        assert meta['epochs']==300 and not meta['prediction_fold_labels_used_for_selection']
        assert sha256(path)==meta['cache_sha256'];hashes[path.name]=sha256(path)
        with np.load(path) as z:
            i=z['prediction_rows'];filled[i]+=1
            np.testing.assert_array_equal(z['prediction_indices'],outer['train_indices'][i])
            np.testing.assert_array_equal(z['pairs'],outer['train_pairs'][i])
            assert not np.intersect1d(z['train_indices'],z['prediction_indices']).size
            assert not np.intersect1d(z['train_indices'],outer['val_indices']).size
            offset[i]=z['logit']
            for key in KEYS:
                if key not in rows:rows[key]=np.zeros((n,*z[key].shape[1:]),dtype=z[key].dtype)
                rows[key][i]=z[key]
    assert (filled==1).all()
    path=HERE/'bases'/f'base_f{fold}_k-1.npz';meta=json.loads(path.with_suffix('.json').read_text())
    assert sha256(path)==meta['cache_sha256'];hashes[path.name]=sha256(path)
    with np.load(path) as z:
        np.testing.assert_array_equal(z['prediction_indices'],outer['val_indices'])
        np.testing.assert_array_equal(z['train_indices'],outer['train_indices'])
        np.testing.assert_array_equal(z['pairs'],outer['val_pairs'])
        val={key:z[key].copy() for key in KEYS};vo=z['logit'].copy()
    tensor=lambda x:torch.as_tensor(x,device=device,dtype=torch.long if x.dtype.kind in 'iu' and x.ndim==2 and x.shape[1]==SG['k'] else torch.float32)
    tr={key:tensor(x) for key,x in rows.items()};va={key:tensor(x) for key,x in val.items()}
    return outer,tr,va,torch.as_tensor(offset,device=device),torch.as_tensor(vo,device=device),hashes

def predict(model,batch,offset,cfg):
    delta=model(batch,None,None)
    return offset+delta if cfg['offset'] else (1-cfg['inference_weight'])*offset+cfg['inference_weight']*delta

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',choices=CONFIGS,required=True);p.add_argument('--fold',type=int,required=True);p.add_argument('--device',default='cuda:0')
    args=p.parse_args();torch.cuda.set_device(args.device);seed_all(42);cfg=CONFIGS[args.config]
    data,tr,va,offset,vo,hashes=load_context(args.fold,args.device)
    model=SubgraphGNN(SG,0,0).to(args.device);opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-3)
    y=torch.as_tensor(data['train_y'],device=args.device,dtype=torch.float32)
    out=HERE/'results';out.mkdir(exist_ok=True);tag=f'{args.config}_f{args.fold}_s42'
    if (out/f'{tag}.json').exists():raise FileExistsError(tag)
    best,best_ep,state,best_p=-1,0,None,None;curves=[];start=time.time()
    for ep in range(1,801):
        model.train();opt.zero_grad();delta=model(tr,None,None)
        score=offset+delta if cfg['offset'] else delta
        loss=F.binary_cross_entropy_with_logits(score,y)+cfg['penalty']*delta.square().mean()
        assert torch.isfinite(loss);loss.backward();opt.step();model.eval()
        with torch.no_grad():prob=predict(model,va,vo,cfg).sigmoid().cpu().numpy()
        auc,ap=roc_auc_score(data['val_y'],prob),average_precision_score(data['val_y'],prob);s=.5*(auc+ap)
        curves.append(dict(epoch=ep,loss=float(loss),inner_auc=auc,inner_ap=ap,selection_score=s))
        if s>best:
            best,best_ep,best_p=s,ep,prob.copy();state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if ep%100==0:print(tag,ep,best,best_ep,time.time()-start,flush=True)
        if ep>=300 and ep-best_ep>=200:break
    torch.save(state,out/f'{tag}.pt');np.savez_compressed(out/f'{tag}_inner.npz',pairs=data['val_pairs'],y=data['val_y'],probability=best_p)
    (out/f'{tag}_curves.json').write_text(json.dumps(curves,indent=2))
    base_prob=vo.sigmoid().cpu().numpy()
    record=dict(tag=tag,config=args.config,fold=args.fold,seed=42,inner_auc=roc_auc_score(data['val_y'],best_p),inner_ap=average_precision_score(data['val_y'],best_p),
        control_auc=roc_auc_score(data['val_y'],base_prob),control_ap=average_precision_score(data['val_y'],base_prob),best_epoch=best_ep,epochs_run=ep,
        checkpoint_sha256=sha256(out/f'{tag}.pt'),cache_hashes=hashes,source_sha256=sha256(__file__),no_lncrna=True,
        outer_validation_evaluated=False,test_evaluated=False,elapsed_seconds=time.time()-start)
    (out/f'{tag}.json').write_text(json.dumps(record,indent=2));print('DONE',json.dumps(record),flush=True)

if __name__=='__main__':main()

