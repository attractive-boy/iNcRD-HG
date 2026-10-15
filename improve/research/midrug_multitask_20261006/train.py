"""Fit only on inner training; checkpoint by mean inner AUC/AP."""
import argparse,json,time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,average_precision_score
from model import HERE,CONFIGS,load_inner,make_adapter,seed_all,sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',choices=CONFIGS,required=True)
    p.add_argument('--fold',type=int,choices=range(3),required=True);p.add_argument('--device',default='cuda:0')
    args=p.parse_args();seed_all(42);data=load_inner(args.fold);exp=CONFIGS[args.config]
    out=HERE/'results';out.mkdir(exist_ok=True);tag=f'{args.config}_f{args.fold}_s42'
    if (out/f'{tag}.json').exists():raise FileExistsError(tag)
    a=make_adapter(data,exp,args.device);m=a.model
    aux_pairs=torch.as_tensor(data['aux_pairs'].T,device=args.device);aux_y=torch.as_tensor(data['aux_y'],dtype=torch.float32,device=args.device)
    base=[v for k,v in m.named_parameters() if not k.startswith('branch.')]
    extra=[v for k,v in m.named_parameters() if k.startswith('branch.')]
    opt=torch.optim.RMSprop([dict(params=base,lr=1e-4),dict(params=extra,lr=1e-4,weight_decay=1e-4)],alpha=.99)
    best,best_ep,state,best_p=-1,0,None,None;curves=[];started=time.time()
    for ep in range(1,801):
        m.train();opt.zero_grad();loss=F.binary_cross_entropy_with_logits(a.logits('train'),a.y)
        penalty=m.branch_penalty
        if exp['aux_weight']:
            aux,reg=m.auxiliary(a.graph,aux_pairs);aux_loss=F.binary_cross_entropy_with_logits(aux,aux_y)
            loss=loss+exp['aux_weight']*aux_loss;penalty=penalty+reg
        else:aux_loss=loss.new_zeros(())
        loss=loss+.001*penalty
        assert torch.isfinite(loss)
        loss.backward();opt.step();m.eval()
        with torch.no_grad():pred=a.predict('val')
        auc,ap=roc_auc_score(data['val_y'],pred),average_precision_score(data['val_y'],pred);score=.5*(auc+ap)
        curves.append(dict(epoch=ep,loss=float(loss),aux_loss=float(aux_loss),inner_auc=auc,inner_ap=ap,selection_score=score))
        if score>best:
            best,best_ep,best_p=score,ep,pred.copy();state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        if ep%100==0:print(tag,ep,best,best_ep,time.time()-started,flush=True)
        if ep>=300 and ep-best_ep>=200:break
    torch.save(state,out/f'{tag}.pt');np.savez_compressed(out/f'{tag}_inner.npz',pairs=data['val_pairs'],y=data['val_y'],indices=data['val_indices'],probability=best_p)
    (out/f'{tag}_curves.json').write_text(json.dumps(curves,indent=2))
    r=dict(tag=tag,config=args.config,experiment=exp,fold=args.fold,seed=42,inner_auc=roc_auc_score(data['val_y'],best_p),inner_ap=average_precision_score(data['val_y'],best_p),
        best_epoch=best_ep,selection_score=best,epochs_run=ep,source_sha256=sha256(__file__),model_sha256=sha256(HERE/'model.py'),
        data_sha256=sha256(HERE/'inner_data.npz'),checkpoint_sha256=sha256(out/f'{tag}.pt'),outer_validation_evaluated=False,test_evaluated=False,elapsed_seconds=time.time()-started)
    (out/f'{tag}.json').write_text(json.dumps(r,indent=2));print('DONE',json.dumps(r),flush=True)

if __name__=='__main__':main()
