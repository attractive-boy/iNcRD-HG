"""Nine proper-loss fits; unchanged graph and inference."""
import argparse,copy,json,time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,CLEAN,load_inner,seed_all,sha256
from model import CONFIGS,make_adapter,make_optimizer,mask_generator,association_loss

def cpu(x):
    if torch.is_tensor(x):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu(v) for v in x]
    return copy.deepcopy(x)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',choices=CONFIGS,required=True);p.add_argument('--fold',type=int,choices=range(3),required=True);p.add_argument('--device',default='cuda:0');args=p.parse_args()
    plan=json.loads((HERE/'results/plan.json').read_text())
    for f,h in {**plan['source_hashes'],**plan['evidence_hashes']}.items():assert sha256(f)==h
    seed_all(42);torch.cuda.set_device(args.device);cfg=CONFIGS[args.config];data=load_inner(args.fold)
    a=make_adapter(data,args.config,args.device);m=a.model;opt=make_optimizer(m);masks=mask_generator(args.device,20261006+args.fold)
    out=HERE/'results';tag=f'{args.config}_f{args.fold}_s42';assert not (out/f'{tag}.json').exists()
    best=-1.;best_epoch=0;curves=[];start=time.time()
    for epoch in range(1,801):
        m.train();opt.zero_grad(set_to_none=True);main,stats,_=association_loss(a,cfg,epoch,args.fold)
        indices=m.reconstruction.masks(a.graph,.5,masks);rec,_=m.reconstruction.loss(m.encoder,a.graph,indices,False)
        weight=10.*min(1.,epoch/50.);loss=main+weight*rec;assert torch.isfinite(loss)
        loss.backward();opt.step();m.eval()
        with torch.no_grad():pred=a.predict('val')
        auc,ap=roc_auc_score(data['val_y'],pred),average_precision_score(data['val_y'],pred);score=.5*(auc+ap)
        curves.append(dict(epoch=epoch,**stats,reconstruction=float(rec.detach()),reconstruction_weight=weight,loss=float(loss.detach()),inner_auc=auc,inner_ap=ap,selection_score=score))
        if score>best:
            best,best_epoch,best_p=score,epoch,pred.copy();state=cpu(m.state_dict());optimizer=cpu(opt.state_dict());best_rng=cpu(masks.get_state())
        if epoch%100==0:print(tag,epoch,best,best_epoch,time.time()-start,flush=True)
        if epoch>=300 and epoch-best_epoch>=200:break
    torch.save(state,out/f'{tag}.pt');torch.save(cpu(m.state_dict()),out/f'{tag}_final.pt')
    torch.save(dict(selected=optimizer,final=cpu(opt.state_dict()),selected_mask_rng=best_rng,final_mask_rng=cpu(masks.get_state())),out/f'{tag}_optimizers.pt')
    np.savez_compressed(out/f'{tag}_inner.npz',pairs=data['val_pairs'],y=data['val_y'],indices=data['val_indices'],probability=best_p)
    (out/f'{tag}_curves.json').write_text(json.dumps(curves,indent=2))
    r=dict(tag=tag,config=args.config,fold=args.fold,seed=42,best_epoch=best_epoch,epochs_run=epoch,inner_auc=roc_auc_score(data['val_y'],best_p),inner_ap=average_precision_score(data['val_y'],best_p),selection_score=best,
        hashes={f'{tag}{s}':sha256(out/f'{tag}{s}') for s in ['.pt','_final.pt','_optimizers.pt','_inner.npz','_curves.json']},plan_sha256=sha256(out/'plan.json'),
        elapsed_seconds=time.time()-start,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    (out/f'{tag}.json').write_text(json.dumps(r,indent=2));print('DONE',tag,flush=True)

if __name__=='__main__':main()
