"""Train with exact query-deleted attention; only original-training inner validation."""
import argparse,json,time
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,CLEAN,load_inner,seed_all,sha256
from model import CONFIGS,make_adapter,mask_generator


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',choices=CONFIGS,required=True)
    p.add_argument('--fold',type=int,choices=range(3),required=True);p.add_argument('--device',default='cuda:0')
    args=p.parse_args();torch.cuda.set_device(args.device);seed_all(42)
    plan=json.loads((HERE/'results/plan.json').read_text())
    for path,h in plan['source_hashes'].items():assert sha256(path)==h
    data=load_inner(args.fold);cfg=CONFIGS[args.config];a=make_adapter(data,args.config,args.device);m=a.model
    gates=[p for n,p in m.named_parameters() if n.startswith(('encoder.gates1.','encoder.gates2.')) and p.requires_grad]
    base=[p for n,p in m.named_parameters() if not n.startswith(('encoder.gates1.','encoder.gates2.','reconstruction.'))]
    groups=[dict(params=base,lr=1e-4),dict(params=gates,lr=1e-4,weight_decay=1e-4)]
    if cfg['reconstruction_weight']:
        groups.append(dict(params=list(m.reconstruction.parameters()),lr=1e-4,weight_decay=1e-4))
    opt=torch.optim.RMSprop(groups,alpha=.99);gen=mask_generator(args.device,args.fold)
    out=HERE/'results';tag=f'{args.config}_f{args.fold}_s42'
    if (out/f'{tag}.json').exists():raise FileExistsError(tag)
    best,best_ep,state,best_p=-1,0,None,None;curves=[];started=time.time();torch.cuda.reset_peak_memory_stats()
    for ep in range(1,801):
        m.train();opt.zero_grad();logits=a.logits('train');bce=F.binary_cross_entropy_with_logits(logits,a.y)
        aux=logits.new_zeros(());weight=cfg['reconstruction_weight']*min(1.,ep/50.)
        if cfg['reconstruction_weight']:
            rec=m.reconstruction;idx=rec.masks(a.graph,.5,gen);aux,_=rec.loss(m.encoder,a.graph,idx,False)
        loss=bce+weight*aux;assert torch.isfinite(loss);loss.backward();opt.step();m.eval()
        with torch.no_grad():prob=a.predict('val')
        auc,ap=roc_auc_score(data['val_y'],prob),average_precision_score(data['val_y'],prob);score=.5*(auc+ap)
        curves.append(dict(epoch=ep,bce=float(bce),reconstruction=float(aux),weight=weight,loss=float(loss),
            inner_auc=auc,inner_ap=ap,selection_score=score))
        if score>best:
            best,best_ep,best_p=score,ep,prob.copy();state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        if ep%100==0:print(tag,ep,best,best_ep,time.time()-started,flush=True)
        if ep>=300 and ep-best_ep>=200:break
    torch.save(state,out/f'{tag}.pt')
    np.savez_compressed(out/f'{tag}_inner.npz',pairs=data['val_pairs'],y=data['val_y'],indices=data['val_indices'],probability=best_p)
    (out/f'{tag}_curves.json').write_text(json.dumps(curves,indent=2))
    r=dict(tag=tag,config=args.config,fold=args.fold,seed=42,settings=cfg,
        inner_auc=roc_auc_score(data['val_y'],best_p),inner_ap=average_precision_score(data['val_y'],best_p),
        best_epoch=best_ep,epochs_run=ep,selection_score=best,peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,
        checkpoint_sha256=sha256(out/f'{tag}.pt'),source_hashes=plan['source_hashes'],
        graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False,elapsed_seconds=time.time()-started)
    (out/f'{tag}.json').write_text(json.dumps(r,indent=2));print('DONE',json.dumps(r),flush=True)


if __name__=='__main__':main()
