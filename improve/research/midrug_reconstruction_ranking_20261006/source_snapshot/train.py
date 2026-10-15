"""Fixed training-internal ranking/reconstruction comparison, no original test access."""
import argparse,json,time
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,CLEAN,load_inner,seed_all,sha256,SameDrugRanking,rank_scale
from model import CONFIGS,make_adapter,mask_generator

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',choices=CONFIGS,required=True);parser.add_argument('--fold',type=int,choices=range(3),required=True)
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args();torch.cuda.set_device(args.device);seed_all(42)
    data=load_inner(args.fold);cfg=CONFIGS[args.config];a=make_adapter(data,args.config,args.device);m=a.model
    rank=SameDrugRanking(data['train_pairs'],data['train_y'],args.device,42,'mixed_middle');gen=mask_generator(args.device,args.fold)
    base=[p for n,p in m.named_parameters() if not n.startswith('reconstruction.')];extra=[p for n,p in m.named_parameters() if n.startswith('reconstruction.')]
    groups=[dict(params=base,lr=1e-4)]
    if extra:groups.append(dict(params=extra,lr=1e-4,weight_decay=1e-4))
    opt=torch.optim.RMSprop(groups,alpha=.99)
    out=HERE/'results';out.mkdir(exist_ok=True);tag=f'{args.config}_f{args.fold}_s42'
    if (out/f'{tag}.json').exists():raise FileExistsError(tag)
    best,best_ep,state,best_p=-1,0,None,None;curves=[];started=time.time()
    for ep in range(1,801):
        m.train();opt.zero_grad();logits=a.logits('train');bce=F.binary_cross_entropy_with_logits(logits,a.y);bpr=rank.loss(logits)
        rw=rank_scale(ep,cfg['rank_weight']);aw=cfg['reconstruction_weight']*min(1.,ep/50.)
        aux=logits.new_zeros(())
        if cfg['reconstruction_weight']:
            rec=m.reconstruction;indices=rec.masks(a.graph,.5,gen);aux,_=rec.loss(m.encoder,a.graph,indices,False)
        loss=bce+rw*bpr+aw*aux;assert torch.isfinite(loss);loss.backward();opt.step();m.eval()
        with torch.no_grad():p=a.predict('val')
        auc,ap=roc_auc_score(data['val_y'],p),average_precision_score(data['val_y'],p);score=.5*(auc+ap)
        curves.append(dict(epoch=ep,bce=float(bce),bpr=float(bpr),reconstruction=float(aux),rank_weight=rw,reconstruction_weight=aw,loss=float(loss),inner_auc=auc,inner_ap=ap,selection_score=score))
        if score>best:
            best,best_ep,best_p=score,ep,p.copy();state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        if ep%100==0:print(tag,ep,best,best_ep,time.time()-started,flush=True)
        if ep>=300 and ep-best_ep>=200:break
    torch.save(state,out/f'{tag}.pt');np.savez_compressed(out/f'{tag}_inner.npz',pairs=data['val_pairs'],y=data['val_y'],indices=data['val_indices'],probability=best_p)
    (out/f'{tag}_curves.json').write_text(json.dumps(curves,indent=2))
    r=dict(tag=tag,config=args.config,fold=args.fold,seed=42,inner_auc=roc_auc_score(data['val_y'],best_p),inner_ap=average_precision_score(data['val_y'],best_p),
        best_epoch=best_ep,epochs_run=ep,selection_score=best,ranking_coverage=rank.coverage,checkpoint_sha256=sha256(out/f'{tag}.pt'),
        hashes={name:sha256(HERE/name) for name in ['common.py','model.py','train.py']},graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False,elapsed_seconds=time.time()-started)
    (out/f'{tag}.json').write_text(json.dumps(r,indent=2));print('DONE',json.dumps(r),flush=True)

if __name__=='__main__':main()

