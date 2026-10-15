"""Train masked/unmasked reconstruction regularization on internal folds only."""
import argparse,json,time
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,CLEAN,load_inner,seed_all,sha256
from model import CONFIGS,make_adapter,mask_generator

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',choices=CONFIGS,required=True);p.add_argument('--fold',type=int,choices=range(3),required=True)
    p.add_argument('--device',default='cuda:0');args=p.parse_args();torch.cuda.set_device(args.device);seed_all(42)
    cfg=CONFIGS[args.config];data=load_inner(args.fold);a=make_adapter(data,args.config,args.device);m=a.model;rec=m.reconstruction
    generator=mask_generator(args.device,20261006+args.fold)
    base=[p for n,p in m.named_parameters() if not n.startswith('reconstruction.')]
    aux=[p for n,p in m.named_parameters() if n.startswith('reconstruction.')]
    opt=torch.optim.RMSprop([dict(params=base,lr=1e-4),dict(params=aux,lr=1e-4,weight_decay=1e-4)],alpha=.99)
    out=HERE/'results';out.mkdir(exist_ok=True);tag=f'{args.config}_f{args.fold}_s42'
    if (out/f'{tag}.json').exists():raise FileExistsError(tag)
    start=time.time();best,best_ep,state,best_p=-1,0,None,None;curves=[]
    for ep in range(1,801):
        m.train();opt.zero_grad();bce=F.binary_cross_entropy_with_logits(a.logits('train'),a.y)
        indices=rec.masks(a.graph,cfg['ratio'],generator);auxloss,losses=rec.loss(m.encoder,a.graph,indices,cfg['mask'])
        weight=cfg['weight']*min(1.,ep/50.);loss=bce+weight*auxloss
        assert torch.isfinite(loss);loss.backward();opt.step();m.eval()
        with torch.no_grad():pred=a.predict('val')
        auc,ap=roc_auc_score(data['val_y'],pred),average_precision_score(data['val_y'],pred);score=.5*(auc+ap)
        curves.append(dict(epoch=ep,loss=float(loss),bce=float(bce),reconstruction=float(auxloss),reconstruction_rna=float(losses['miRNA']),reconstruction_drug=float(losses['drug']),weight=weight,inner_auc=auc,inner_ap=ap,selection_score=score))
        if score>best:
            best,best_ep,best_p=score,ep,pred.copy();state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        if ep%100==0:print(tag,ep,best,best_ep,float(auxloss),time.time()-start,flush=True)
        if ep>=300 and ep-best_ep>=200:break
    torch.save(state,out/f'{tag}.pt');np.savez_compressed(out/f'{tag}_inner.npz',pairs=data['val_pairs'],y=data['val_y'],indices=data['val_indices'],probability=best_p)
    (out/f'{tag}_curves.json').write_text(json.dumps(curves,indent=2))
    r=dict(tag=tag,config=args.config,fold=args.fold,seed=42,inner_auc=roc_auc_score(data['val_y'],best_p),inner_ap=average_precision_score(data['val_y'],best_p),best_epoch=best_ep,
        epochs_run=ep,selection_score=best,checkpoint_sha256=sha256(out/f'{tag}.pt'),hashes={name:sha256(HERE/name) for name in ['common.py','model.py','train.py']},
        graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False,elapsed_seconds=time.time()-start)
    (out/f'{tag}.json').write_text(json.dumps(r,indent=2));print('DONE',json.dumps(r),flush=True)

if __name__=='__main__':main()

