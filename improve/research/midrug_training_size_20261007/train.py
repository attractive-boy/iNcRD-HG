"""Fit one fixed model on one frozen nested internal training subset."""
import argparse,json,time
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,CLEAN,load_subset,seed_all,sha256
from model import CONFIGS,SIZES,make_adapter,make_optimizer,mask_generator


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',choices=CONFIGS,required=True)
    parser.add_argument('--size',choices=SIZES,required=True);parser.add_argument('--fold',type=int,choices=range(3),required=True)
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    torch.cuda.set_device(args.device);seed_all(42);plan=json.loads((HERE/'results/plan.json').read_text())
    for path,digest in plan['source_hashes'].items():assert sha256(path)==digest
    assert sha256(HERE/'subsets.npz')==plan['subsets_sha256']
    data=load_subset(args.fold,args.size);cfg=CONFIGS[args.config];a=make_adapter(data,args.config,args.device);m=a.model;opt=make_optimizer(m)
    generator=mask_generator(args.device,20261006+args.fold);out=HERE/'results';tag=f'{args.config}_{args.size}_f{args.fold}_s42'
    if (out/f'{tag}.json').exists():raise FileExistsError(tag)
    best,best_ep,state,best_p=-1,0,None,None;curves=[];started=time.time()
    for ep in range(1,801):
        m.train();opt.zero_grad();bce=F.binary_cross_entropy_with_logits(a.logits('train'),a.y)
        weight=cfg['reconstruction_weight']*min(1.,ep/50.);rec=bce.new_zeros(())
        if weight:
            ix=m.reconstruction.masks(a.graph,.5,generator);rec,_=m.reconstruction.loss(m.encoder,a.graph,ix,False)
        loss=bce+weight*rec;assert torch.isfinite(loss);loss.backward();opt.step();m.eval()
        with torch.no_grad():pred=a.predict('val')
        auc,ap=roc_auc_score(data['val_y'],pred),average_precision_score(data['val_y'],pred);score=.5*(auc+ap)
        curves.append(dict(epoch=ep,bce=float(bce),reconstruction=float(rec),weight=weight,loss=float(loss),inner_auc=auc,inner_ap=ap,selection_score=score))
        if score>best:
            best,best_ep,best_p=score,ep,pred.copy();state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        if ep%100==0:print(tag,ep,best,best_ep,time.time()-started,flush=True)
        if ep>=300 and ep-best_ep>=200:break
    torch.save(state,out/f'{tag}.pt');np.savez_compressed(out/f'{tag}_inner.npz',pairs=data['val_pairs'],y=data['val_y'],indices=data['val_indices'],probability=best_p)
    (out/f'{tag}_curves.json').write_text(json.dumps(curves,indent=2))
    result=dict(tag=tag,config=args.config,size=args.size,fraction=SIZES[args.size],fold=args.fold,seed=42,
        train_count=len(data['train_y']),positive_count=int(data['train_y'].sum()),
        inner_auc=roc_auc_score(data['val_y'],best_p),inner_ap=average_precision_score(data['val_y'],best_p),best_epoch=best_ep,epochs_run=ep,
        selection_score=best,checkpoint_sha256=sha256(out/f'{tag}.pt'),source_hashes=plan['source_hashes'],subsets_sha256=plan['subsets_sha256'],
        graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False,elapsed_seconds=time.time()-started)
    (out/f'{tag}.json').write_text(json.dumps(result,indent=2));print('DONE',tag,flush=True)

if __name__=='__main__':main()
