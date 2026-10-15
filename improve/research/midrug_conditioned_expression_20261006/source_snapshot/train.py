"""Original-training three-fold development of drug-conditioned raw-expression propagation."""
import argparse,json,time
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,CLEAN,load_inner,seed_all,sha256
from model import CONFIGS,make_adapter,mask_generator,make_optimizer

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',choices=CONFIGS,required=True)
    p.add_argument('--fold',type=int,choices=range(3),required=True);p.add_argument('--device',default='cuda:0');args=p.parse_args()
    torch.cuda.set_device(args.device);seed_all(42);plan=json.loads((HERE/'results/plan.json').read_text())
    for path,h in plan['source_hashes'].items():assert sha256(path)==h
    cfg=CONFIGS[args.config];data=load_inner(args.fold);a=make_adapter(data,args.config,args.device);m=a.model;opt=make_optimizer(m,cfg)
    generator=mask_generator(args.device,20261006+args.fold)
    out=HERE/'results';tag=f'{args.config}_f{args.fold}_s42'
    if (out/f'{tag}.json').exists():raise FileExistsError(tag)
    best,best_ep,state,best_p=-1,0,None,None;curves=[];start=time.time();torch.cuda.reset_peak_memory_stats()
    for ep in range(1,801):
        m.train();opt.zero_grad();logits=a.logits('train');bce=F.binary_cross_entropy_with_logits(logits,a.y)
        weight=cfg['reconstruction_weight']*min(1.,ep/50.);rec=bce.new_zeros(())
        if weight:
            indices=m.reconstruction.masks(a.graph,.5,generator);rec,_=m.reconstruction.loss(m.encoder,a.graph,indices,False)
        loss=bce+weight*rec;assert torch.isfinite(loss);loss.backward();opt.step();m.eval()
        with torch.no_grad():prob=a.predict('val')
        auc,ap=roc_auc_score(data['val_y'],prob),average_precision_score(data['val_y'],prob);score=.5*(auc+ap)
        curves.append(dict(epoch=ep,bce=float(bce.detach()),reconstruction=float(rec.detach()),weight=weight,loss=float(loss.detach()),inner_auc=auc,inner_ap=ap,selection_score=score))
        if score>best:
            best,best_ep,best_p=score,ep,prob.copy();state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        if ep%100==0:print(tag,ep,best,best_ep,time.time()-start,flush=True)
        if ep>=300 and ep-best_ep>=200:break
    torch.save(state,out/f'{tag}.pt');np.savez_compressed(out/f'{tag}_inner.npz',pairs=data['val_pairs'],y=data['val_y'],indices=data['val_indices'],probability=best_p)
    (out/f'{tag}_curves.json').write_text(json.dumps(curves,indent=2))
    r=dict(tag=tag,config=args.config,fold=args.fold,seed=42,settings=cfg,inner_auc=roc_auc_score(data['val_y'],best_p),inner_ap=average_precision_score(data['val_y'],best_p),
        best_epoch=best_ep,epochs_run=ep,selection_score=best,checkpoint_sha256=sha256(out/f'{tag}.pt'),source_hashes=plan['source_hashes'],
        graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False,elapsed_seconds=time.time()-start)
    (out/f'{tag}.json').write_text(json.dumps(r,indent=2));print('DONE',tag,flush=True)

if __name__=='__main__':main()
