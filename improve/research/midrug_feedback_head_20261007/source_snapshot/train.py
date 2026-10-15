"""Train the fixed head-adaptation ablation, with every encoder/buffer frozen."""
import argparse,copy,json,time
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,CLEAN,load_inner,seed_all,sha256
from model import CONFIGS,initialize,make_optimizer

def cpu_copy(x):
    if torch.is_tensor(x):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu_copy(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu_copy(v) for v in x]
    return copy.deepcopy(x)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',choices=CONFIGS,required=True);p.add_argument('--fold',type=int,choices=range(3),required=True)
    p.add_argument('--device',default='cuda:0');args=p.parse_args();plan=json.loads((HERE/'results/plan.json').read_text())
    for path,h in {**plan['source_hashes'],**plan['evidence_hashes']}.items():assert sha256(path)==h
    torch.cuda.set_device(args.device);seed_all(42);data=load_inner(args.fold);a,source,origin=initialize(data,args.config,args.fold,args.device)
    m=a.model;m.eval();opt=make_optimizer(m);assert not opt.state
    with torch.no_grad():pred=a.predict('val')
    with np.load(source.with_name(source.stem+'_inner.npz')) as z:best_p=z['probability'].copy()
    np.testing.assert_allclose(pred,best_p,atol=2e-5,rtol=2e-5)
    auc,ap=roc_auc_score(data['val_y'],best_p),average_precision_score(data['val_y'],best_p);best=.5*(auc+ap);best_ep=0
    best_state=cpu_copy(m.state_dict());best_opt=cpu_copy(opt.state_dict());curves=[dict(stage_epoch=0,optimizer_update=False,inner_auc=auc,inner_ap=ap,selection_score=best)]
    out=HERE/'results';tag=f'{args.config}_f{args.fold}_s42';assert not (out/f'{tag}.json').exists();start=time.time();torch.cuda.reset_peak_memory_stats()
    for ep in range(1,301):
        m.eval()
        if hasattr(m,'feedback'):m.feedback.train()
        opt.zero_grad(set_to_none=True)
        loss=F.binary_cross_entropy_with_logits(a.logits('train'),a.y);assert torch.isfinite(loss);loss.backward();opt.step();m.eval()
        with torch.no_grad():pred=a.predict('val')
        auc,ap=roc_auc_score(data['val_y'],pred),average_precision_score(data['val_y'],pred);score=.5*(auc+ap)
        curves.append(dict(stage_epoch=ep,optimizer_update=True,loss=float(loss),inner_auc=auc,inner_ap=ap,selection_score=score))
        if score>best:best,best_ep,best_p,best_state,best_opt=score,ep,pred.copy(),cpu_copy(m.state_dict()),cpu_copy(opt.state_dict())
        if ep%25==0:print(tag,ep,best,best_ep,time.time()-start,flush=True)
        if ep>=100 and ep-best_ep>=100:break
    torch.save(best_state,out/f'{tag}.pt');torch.save(cpu_copy(m.state_dict()),out/f'{tag}_final.pt')
    torch.save(dict(selected=best_opt,final=cpu_copy(opt.state_dict())),out/f'{tag}_optimizers.pt')
    np.savez_compressed(out/f'{tag}_inner.npz',pairs=data['val_pairs'],y=data['val_y'],indices=data['val_indices'],probability=best_p)
    (out/f'{tag}_curves.json').write_text(json.dumps(curves,indent=2))
    r=dict(tag=tag,config=args.config,fold=args.fold,seed=42,source_metadata=str(source),source_metadata_sha256=sha256(source),source_checkpoint_sha256=origin['checkpoint_sha256'],source_best_epoch=origin['best_epoch'],
        best_epoch=best_ep,epochs_run=ep,selection_score=best,inner_auc=roc_auc_score(data['val_y'],best_p),inner_ap=average_precision_score(data['val_y'],best_p),
        checkpoint_sha256=sha256(out/f'{tag}.pt'),final_checkpoint_sha256=sha256(out/f'{tag}_final.pt'),optimizers_sha256=sha256(out/f'{tag}_optimizers.pt'),
        source_hashes=plan['source_hashes'],plan_sha256=sha256(out/'plan.json'),graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False,elapsed_seconds=time.time()-start,peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30)
    (out/f'{tag}.json').write_text(json.dumps(r,indent=2));print('DONE',tag,flush=True)

if __name__=='__main__':main()
