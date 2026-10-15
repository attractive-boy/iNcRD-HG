"""One fixed variance/covariance training objective on an original-training inner fold."""
import argparse
import json
import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,CLEAN,load_inner,seed_all,sha256
from model import CONFIGS,make_adapter,make_optimizer,mask_generator,step


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',choices=CONFIGS,required=True)
    p.add_argument('--fold',type=int,choices=range(3),required=True);p.add_argument('--device',default='cuda:0');args=p.parse_args()
    torch.cuda.set_device(args.device);seed_all(42);plan=json.loads((HERE/'results/plan.json').read_text())
    for path,digest in plan['source_hashes'].items():assert sha256(path)==digest
    data=load_inner(args.fold);a=make_adapter(data,args.config,args.device);m=a.model;opt=make_optimizer(m)
    generator=mask_generator(args.device,20261006+args.fold);curves=[];best=-1.;best_ep=0;state=None;best_p=None
    tag=f'{args.config}_f{args.fold}_s42';out=HERE/'results';assert not (out/(tag+'.json')).exists();started=time.time()
    for ep in range(1,801):
        losses=step(a,opt,ep,generator)
        with torch.no_grad():pred=a.predict('val')
        auc,ap=roc_auc_score(data['val_y'],pred),average_precision_score(data['val_y'],pred);score=.5*(auc+ap)
        curves.append(dict(epoch=ep,inner_auc=auc,inner_ap=ap,selection_score=score,**losses))
        if score>best:
            best,best_ep,best_p=score,ep,pred.copy();state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        if ep%100==0:print(tag,ep,best,best_ep,time.time()-started,flush=True)
        if ep>=300 and ep-best_ep>=200:break
    torch.save(state,out/(tag+'.pt'))
    np.savez_compressed(out/(tag+'_inner.npz'),pairs=data['val_pairs'],y=data['val_y'],indices=data['val_indices'],probability=best_p)
    (out/(tag+'_curves.json')).write_text(json.dumps(curves,indent=2))
    result=dict(tag=tag,config=args.config,fold=args.fold,seed=42,best_epoch=best_ep,epochs_run=ep,
        inner_auc=roc_auc_score(data['val_y'],best_p),inner_ap=average_precision_score(data['val_y'],best_p),selection_score=best,
        checkpoint_sha256=sha256(out/(tag+'.pt')),source_hashes=plan['source_hashes'],graph_sha256=sha256(CLEAN/'graph.pt'),
        data_sha256=sha256(CLEAN/'inner_data.npz'),no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False,elapsed_seconds=time.time()-started)
    (out/(tag+'.json')).write_text(json.dumps(result,indent=2));print('DONE',tag,flush=True)


if __name__=='__main__':main()
