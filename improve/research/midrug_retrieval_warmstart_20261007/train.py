"""Frozen or jointly updated backbone with an identical zero-residual retrieval start."""
import argparse,copy,json,time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,CLEAN,load_inner,seed_all,sha256
from model import CONFIGS,initialize,make_optimizer,resumed_sampler,step


def cpu_copy(obj):
    if torch.is_tensor(obj):return obj.detach().cpu().clone()
    if isinstance(obj,dict):return {k:cpu_copy(v) for k,v in obj.items()}
    if isinstance(obj,list):return [cpu_copy(v) for v in obj]
    if isinstance(obj,tuple):return tuple(cpu_copy(v) for v in obj)
    return copy.deepcopy(obj)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',choices=CONFIGS,required=True)
    parser.add_argument('--fold',type=int,choices=range(3),required=True);parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    torch.cuda.set_device(args.device);seed_all(42);plan=json.loads((HERE/'results/plan.json').read_text())
    for path,digest in {**plan['source_hashes'],**plan['evidence_hashes']}.items():assert sha256(path)==digest
    data=load_inner(args.fold);a,source,baseline=initialize(data,args.config,args.fold,args.device);m=a.model;opt=make_optimizer(m)
    assert not opt.state;generator=resumed_sampler(a,args.fold,baseline['best_epoch'])
    with np.load(source.with_name(source.stem+'_inner.npz')) as z:
        for key,dk in [('pairs','val_pairs'),('y','val_y'),('indices','val_indices')]:np.testing.assert_array_equal(z[key],data[dk])
        best_p=z['probability'].copy()
    m.eval()
    with torch.no_grad():initial_p=a.predict('val')
    np.testing.assert_allclose(initial_p,best_p,atol=2e-5,rtol=2e-5)
    best=.5*(baseline['inner_auc']+baseline['inner_ap']);best_ep=0;best_state=cpu_copy(m.state_dict());best_opt=cpu_copy(opt.state_dict())
    curves=[dict(stage_epoch=0,optimizer_update=False,inner_auc=baseline['inner_auc'],inner_ap=baseline['inner_ap'],selection_score=best)]
    out=HERE/'results';tag=f'{args.config}_f{args.fold}_s42'
    if (out/f'{tag}.json').exists():raise FileExistsError(tag)
    started=time.time()
    for ep in range(1,301):
        losses=step(a,opt,args.config,ep,generator);m.eval()
        with torch.no_grad():pred=a.predict('val')
        auc,ap=roc_auc_score(data['val_y'],pred),average_precision_score(data['val_y'],pred);score=.5*(auc+ap)
        curves.append(dict(stage_epoch=ep,optimizer_update=True,inner_auc=auc,inner_ap=ap,selection_score=score,**losses))
        if score>best:best,best_ep,best_p,best_state,best_opt=score,ep,pred.copy(),cpu_copy(m.state_dict()),cpu_copy(opt.state_dict())
        if ep%50==0:print(tag,ep,best,best_ep,time.time()-started,flush=True)
        if ep>=100 and ep-best_ep>=100:break
    torch.save(cpu_copy(m.state_dict()),out/f'{tag}_final.pt');torch.save(best_state,out/f'{tag}.pt');torch.save(dict(selected=best_opt,final=cpu_copy(opt.state_dict())),out/f'{tag}_optimizers.pt')
    np.savez_compressed(out/f'{tag}_inner.npz',pairs=data['val_pairs'],y=data['val_y'],indices=data['val_indices'],probability=best_p)
    (out/f'{tag}_curves.json').write_text(json.dumps(curves,indent=2))
    result=dict(tag=tag,config=args.config,fold=args.fold,seed=42,source_metadata=str(source),source_metadata_sha256=sha256(source),source_checkpoint_sha256=baseline['checkpoint_sha256'],
        source_best_epoch=baseline['best_epoch'],best_epoch=best_ep,epochs_run=ep,selection_score=best,
        inner_auc=roc_auc_score(data['val_y'],best_p),inner_ap=average_precision_score(data['val_y'],best_p),
        checkpoint_sha256=sha256(out/f'{tag}.pt'),final_checkpoint_sha256=sha256(out/f'{tag}_final.pt'),plan_sha256=sha256(out/'plan.json'),optimizers_sha256=sha256(out/f'{tag}_optimizers.pt'),source_hashes=plan['source_hashes'],
        graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False,elapsed_seconds=time.time()-started)
    (out/f'{tag}.json').write_text(json.dumps(result,indent=2));print('DONE',tag,flush=True)

if __name__=='__main__':main()
