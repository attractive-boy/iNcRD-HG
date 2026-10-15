"""Compare drug input modes under an identical internal validation protocol."""
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
    groups=[dict(params=[v for n,v in m.named_parameters() if not n.startswith(('reconstruction.','molecular.'))],lr=1e-4),
            dict(params=list(m.reconstruction.parameters()),lr=1e-4,weight_decay=1e-4),
            dict(params=list(m.molecular.parameters()),lr=3e-4,weight_decay=cfg['weight_decay'])]
    opt=torch.optim.RMSprop(groups,alpha=.99);gen=mask_generator(args.device,args.fold)
    out=HERE/'results';tag=f'{args.config}_f{args.fold}_s42'
    if (out/f'{tag}.json').exists():raise FileExistsError(tag)
    best,best_ep,state,best_p=-1,0,None,None;curves=[];started=time.time()
    for ep in range(1,801):
        m.train();opt.zero_grad();logits=a.logits('train')
        bce=F.binary_cross_entropy_with_logits(logits,a.y)
        branch_bce=F.binary_cross_entropy_with_logits(m.branch_logits,a.y)
        rec=m.reconstruction;indices=rec.masks(a.graph,.5,gen)
        aux,_=rec.loss(m.encoder,a.graph,indices,False);aw=cfg['reconstruction_weight']*min(1.,ep/50.)
        loss=bce+cfg['branch_aux_weight']*branch_bce+aw*aux
        assert torch.isfinite(loss);loss.backward();opt.step();m.eval()
        with torch.no_grad():
            probs=dict(probability=a.predict('val'),branch_probability=m.branch_logits.sigmoid().cpu().numpy(),
                       base_probability=m.base_logits.sigmoid().cpu().numpy())
        auc,ap=roc_auc_score(data['val_y'],probs['probability']),average_precision_score(data['val_y'],probs['probability'])
        score=.5*(auc+ap)
        curves.append(dict(epoch=ep,bce=float(bce),branch_bce=float(branch_bce),reconstruction=float(aux),
            branch_aux_weight=cfg['branch_aux_weight'],reconstruction_weight=aw,loss=float(loss),
            inner_auc=auc,inner_ap=ap,selection_score=score))
        if score>best:
            best,best_ep,best_p=score,ep,{k:v.copy() for k,v in probs.items()}
            state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
        if ep%100==0:print(tag,ep,best,best_ep,time.time()-started,flush=True)
        if ep>=300 and ep-best_ep>=200:break
    torch.save(state,out/f'{tag}.pt')
    np.savez_compressed(out/f'{tag}_inner.npz',pairs=data['val_pairs'],y=data['val_y'],indices=data['val_indices'],**best_p)
    (out/f'{tag}_curves.json').write_text(json.dumps(curves,indent=2))
    diagnostic={k:dict(auc=roc_auc_score(data['val_y'],v),ap=average_precision_score(data['val_y'],v))
                for k,v in best_p.items() if k!='probability'}
    r=dict(tag=tag,config=args.config,fold=args.fold,seed=42,settings=cfg,
        inner_auc=roc_auc_score(data['val_y'],best_p['probability']),inner_ap=average_precision_score(data['val_y'],best_p['probability']),
        branch_diagnostics=diagnostic,best_epoch=best_ep,epochs_run=ep,selection_score=best,
        branch_parameters=sum(p.numel() for p in m.molecular.parameters()),
        checkpoint_sha256=sha256(out/f'{tag}.pt'),source_hashes=plan['source_hashes'],
        graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),
        no_lncrna=True,external_data_used=False,outer_validation_evaluated=False,test_evaluated=False,
        elapsed_seconds=time.time()-started)
    (out/f'{tag}.json').write_text(json.dumps(r,indent=2));print('DONE',json.dumps(r),flush=True)


if __name__=='__main__':main()
