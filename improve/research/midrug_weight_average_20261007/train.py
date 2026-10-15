"""Fit one trajectory; record matched raw and fixed-mean selected models."""
import argparse
import json
import time
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from common import HERE, CLEAN, load_inner, seed_all, sha256
from model import make_adapter, make_optimizer, ParameterMean, clone_state, predict_mean, step, START_EPOCH


def scores(y, p):
    auc, ap = roc_auc_score(y, p), average_precision_score(y, p)
    return dict(inner_auc=auc, inner_ap=ap, selection_score=.5*(auc+ap))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--fold',type=int,choices=range(3),required=True)
    parser.add_argument('--device',default='cuda:0'); args=parser.parse_args()
    torch.cuda.set_device(args.device); seed_all(42)
    plan=json.loads((HERE/'results/plan.json').read_text())
    for path,digest in plan['source_hashes'].items(): assert sha256(path)==digest
    data=load_inner(args.fold);a=make_adapter(data,'raw',args.device);opt=make_optimizer(a.model);average=ParameterMean()
    generator=torch.Generator(device=args.device).manual_seed(20261006+args.fold)
    best={n:dict(selection_score=-1.,epoch=0) for n in ['raw','mean']};curves={n:[] for n in best}
    out=HERE/'results'; started=time.time()
    for name in best: assert not (out/f'{name}_f{args.fold}_s42.json').exists()
    for ep in range(1,801):
        losses=step(a,opt,ep,generator); average.update(a.model,ep)
        with torch.no_grad():raw=a.predict('val')
        predictions={'raw':raw}
        if average.count:predictions['mean']=predict_mean(a,average)
        for name,pred in predictions.items():
            row=dict(epoch=ep,**scores(data['val_y'],pred))
            if name=='raw':row.update(losses)
            else:row['average_count']=average.count
            curves[name].append(row)
            if row['selection_score']>best[name]['selection_score']:
                state=clone_state(a.model,cpu=True) if name=='raw' else {k:v.cpu().clone() for k,v in average.state(a.model).items()}
                best[name]=dict(**row,probability=pred.copy(),state=state)
                if name=='mean': best[name]['sums']={n:p.cpu().clone() for n,p in average.sums.items()}
        if ep%100==0:print('EPOCH',args.fold,ep,{n:(b['epoch'],b['selection_score']) for n,b in best.items()},flush=True)
        if ep>=300 and ep-best['raw']['epoch']>=200:break
    for name,b in best.items():
        tag=f'{name}_f{args.fold}_s42'; checkpoint=out/(tag+'.pt')
        torch.save(b['state'],checkpoint)
        np.savez_compressed(out/(tag+'_inner.npz'),pairs=data['val_pairs'],y=data['val_y'],indices=data['val_indices'],probability=b['probability'])
        (out/(tag+'_curves.json')).write_text(json.dumps(curves[name],indent=2))
        metadata=dict(tag=tag,config=name,fold=args.fold,seed=42,inner_auc=b['inner_auc'],inner_ap=b['inner_ap'],
            selection_score=b['selection_score'],best_epoch=b['epoch'],epochs_run=ep,checkpoint_sha256=sha256(checkpoint),
            source_hashes=plan['source_hashes'],graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),
            no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False,trajectory_elapsed_seconds=time.time()-started)
        if name=='mean':
            sums=out/(tag+'_sums.pt'); torch.save(dict(selected=b['sums'],selected_count=b['average_count'],
                final={n:p.cpu().clone() for n,p in average.sums.items()},final_count=average.count),sums)
            metadata.update(average_start_epoch=START_EPOCH,average_count=b['average_count'],sums_sha256=sha256(sums))
        (out/(tag+'.json')).write_text(json.dumps(metadata,indent=2))
    print('DONE',args.fold,flush=True)


if __name__=='__main__':main()
