"""Predeclared 2x2 OCN-inspired ablation on original-training inner folds."""
import argparse,json,time
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,CLEAN,load_inner,seed_all,sha256
from model import CONFIGS,make_adapter

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',choices=CONFIGS,required=True);p.add_argument('--fold',type=int,choices=range(3),required=True)
    p.add_argument('--device',default='cuda:0');args=p.parse_args()
    torch.cuda.set_device(args.device);seed_all(42);data=load_inner(args.fold)
    out=HERE/'results';out.mkdir(exist_ok=True);tag=f'{args.config}_f{args.fold}_s42'
    if (out/f'{tag}.json').exists():raise FileExistsError(tag)
    a=make_adapter(data,args.config,args.fold,args.device);m=a.model
    base=[p for n,p in m.named_parameters() if not n.startswith('ocn.')]
    extra=[p for n,p in m.named_parameters() if n.startswith('ocn.')]
    opt=torch.optim.RMSprop([dict(params=base,lr=1e-4),dict(params=extra,lr=1e-4,weight_decay=1e-4)],alpha=.99)
    best,best_ep,state,best_p=-1,0,None,None;curves=[];start=time.time()
    for ep in range(1,801):
        m.train();opt.zero_grad();loss=F.binary_cross_entropy_with_logits(a.logits('train'),a.y)
        assert torch.isfinite(loss);loss.backward();opt.step();m.eval()
        with torch.no_grad():pred=a.predict('val')
        auc,ap=roc_auc_score(data['val_y'],pred),average_precision_score(data['val_y'],pred);score=.5*(auc+ap)
        curves.append(dict(epoch=ep,loss=float(loss),inner_auc=auc,inner_ap=ap,selection_score=score))
        if score>best:
            best,best_ep,best_p=score,ep,pred.copy();state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        if ep%100==0:print(tag,ep,best,best_ep,time.time()-start,flush=True)
        if ep>=300 and ep-best_ep>=200:break
    torch.save(state,out/f'{tag}.pt');np.savez_compressed(out/f'{tag}_inner.npz',pairs=data['val_pairs'],y=data['val_y'],indices=data['val_indices'],probability=best_p)
    (out/f'{tag}_curves.json').write_text(json.dumps(curves,indent=2))
    record=dict(tag=tag,config=args.config,fold=args.fold,seed=42,inner_auc=roc_auc_score(data['val_y'],best_p),inner_ap=average_precision_score(data['val_y'],best_p),
        best_epoch=best_ep,epochs_run=ep,selection_score=best,elapsed_seconds=time.time()-start,
        hashes={name:sha256(HERE/name) for name in ('common.py','features.py','model.py','train.py')},
        graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),checkpoint_sha256=sha256(out/f'{tag}.pt'),
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    (out/f'{tag}.json').write_text(json.dumps(record,indent=2));print('DONE',json.dumps(record),flush=True)

if __name__=='__main__':main()

