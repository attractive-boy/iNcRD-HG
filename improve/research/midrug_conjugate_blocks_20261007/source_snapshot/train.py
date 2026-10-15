"""Fit all states and freeze training-ELBO choices before any heldout prediction."""
import datetime,json,time
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score,log_loss,brier_score_loss
from common import HERE,load_inner,groups,control,verify_plan,sha256,write
from numerics import observations,initialization,globals_update,fit,elbo,predict,global_mean,fuse

def metrics(y,p):return dict(auc=float(roc_auc_score(y,p)),ap=float(average_precision_score(y,p)),
    bce=float(log_loss(y,p)),brier=float(brier_score_loss(y,p)))

def main():
    plan=verify_plan();out=HERE/'results';assert not (out/'fits_frozen.json').exists();g=groups();records=[]
    for fold in plan['folds']:
        data=load_inner(fold);P,N=observations(data['train_pairs'],data['train_y'],(605,216));starts={};fixed={}
        for start in plan['starts']:
            prefix='' if start=='feature' else 'shuffled_';q0,q1=initialization(g[prefix+'miRNA'],g[prefix+'drug'])
            t=time.monotonic();s,history,converged=fit(P,N,q0,q1,max_sweeps=plan['max_sweeps']);elapsed=time.monotonic()-t
            tag=f'f{fold}_{start}';statefile=out/f'{tag}.npz';np.savez_compressed(statefile,**s)
            rec=dict(fold=fold,start=start,converged=converged,history=history,seconds=elapsed,state_sha256=sha256(statefile),
                     final_elbo=history[-1]['elbo'],plan_sha256=sha256(out/'plan.json'))
            write(out/f'{tag}.json',rec);starts[start]=rec
            qr,qd=initialization(g[prefix+'miRNA'],g[prefix+'drug'],soft=False);fs=globals_update(qr,qd,P,N)
            ff=out/f'f{fold}_fixed_{start}.npz';np.savez_compressed(ff,**fs);fixed[start]=dict(state_sha256=sha256(ff),elbo=elbo(fs,P,N))
            print(f'FIT fold={fold} start={start} sweeps={len(history)-1} converged={converged} training_ELBO={history[-1]["elbo"]:.9f}',flush=True)
        chosen=max(plan['starts'],key=lambda x:starts[x]['final_elbo'])
        records.append(dict(fold=fold,starts=starts,fixed=fixed,chosen=chosen,prior=global_mean(data['train_y'])))
    frozen=dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),records=records,
        all_converged=all(s['converged'] for r in records for s in r['starts'].values()),plan_sha256=sha256(out/'plan.json'),
        heldout_predictions_evaluated=False,heldout_metrics_evaluated=False)
    write(out/'fits_frozen.json',frozen)
    if not frozen['all_converged']:
        print('Nonconvergence: no heldout prediction/evaluation; protocol is ineligible.',flush=True);return
    # All states and choices are now written, immutable, and keyed by training evidence.
    for r in records:
        fold=r['fold'];data=load_inner(fold);base=control(fold,'unmasked10',data);outputs={};scores={};standalone={}
        for name in plan['readouts']:
            tag=f'f{fold}_{r["chosen"]}' if name=='learned' else f'f{fold}_{name}'
            s=dict(np.load(out/f'{tag}.npz'));bp=predict(s,data['val_pairs']);pp=fuse(base,bp,r['prior'],plan['fusion_weight'])
            outputs['block_'+name]=bp;outputs['prob_'+name]=pp;scores[name]=metrics(data['val_y'],pp);standalone[name]=metrics(data['val_y'],bp)
        np.savez_compressed(out/f'f{fold}_predictions.npz',**outputs,base_probability=base,
            pairs=data['val_pairs'],y=data['val_y'],indices=data['val_indices'])
        write(out/f'f{fold}_evaluation.json',dict(fold=fold,chosen=r['chosen'],inner_metrics=scores,block_standalone=standalone,
            fits_frozen_sha256=sha256(out/'fits_frozen.json'),predictions_sha256=sha256(out/f'f{fold}_predictions.npz')))
        print('EVALUATED',fold,json.dumps(scores),flush=True)

if __name__=='__main__':main()
