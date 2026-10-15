"""Replay all fitted parameters; independent sufficient statistics and ELBO."""
import json
import numpy as np
from scipy.special import gammaln,betaln,xlogy
from scipy.stats import rankdata
from common import HERE,load_inner,groups,control,verify_plan,sha256,write
from numerics import KEYS,observations,initialization,fit,sweep,predict,global_mean,fuse
from train import metrics

def independent_counts(s,pairs,y):
    counts=[]
    for label in [1,0]:
        pp=pairs[y==label]
        counts.append(np.einsum('ik,il->kl',s['qr'][pp[:,0]],s['qd'][pp[:,1]],optimize=False))
    return counts

def profiled_elbo(s,cp,cn):
    value=np.sum(betaln(cp+.5,cn+.5)-betaln(.5,.5))
    for q in [s['qr'],s['qd']]:
        g=1+q.sum(0)
        value+=gammaln(g).sum()-gammaln(g.sum())+gammaln(len(g))-np.sum(xlogy(q,q))
    return float(value)

def rank_metrics(y,p):
    n1=y.sum();n0=len(y)-n1;ranks=rankdata(p)
    auc=(ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
    order=np.argsort(-p,kind='stable');ys=y[order];ps=p[order]
    ends=np.r_[np.flatnonzero(ps[:-1]!=ps[1:]),len(ps)-1];tp=np.cumsum(ys)[ends]
    ap=np.sum(np.diff(np.r_[0,tp])/n1*tp/(ends+1))
    return dict(auc=float(auc),ap=float(ap))

def main():
    plan=verify_plan();out=HERE/'results';frozen=json.loads((out/'fits_frozen.json').read_text());g=groups()
    assert frozen['plan_sha256']==sha256(out/'plan.json');assert not frozen['heldout_predictions_evaluated']
    records=[];readouts=[];controls=[]
    for r in frozen['records']:
        fold=r['fold'];data=load_inner(fold);P,N=observations(data['train_pairs'],data['train_y'],(605,216))
        assert r['chosen']==max(plan['starts'],key=lambda key:r['starts'][key]['final_elbo'])
        assert r['prior']==global_mean(data['train_y'])
        for start in plan['starts']:
            prefix='' if start=='feature' else 'shuffled_';qr,qd=initialization(g[prefix+'miRNA'],g[prefix+'drug'])
            ss,hh,cc=fit(P,N,qr,qd,max_sweeps=plan['max_sweeps']);old=r['starts'][start]
            assert cc==old['converged'] and len(hh)==len(old['history'])
            np.testing.assert_allclose([v['elbo'] for v in hh],[v['elbo'] for v in old['history']],atol=1e-9,rtol=0)
            for mode in [start,'fixed_'+start]:
                f=out/f'f{fold}_{mode}.npz';s=dict(np.load(f))
                expectedhash=old['state_sha256'] if mode==start else r['fixed'][start]['state_sha256']
                assert sha256(f)==expectedhash
                if mode==start:
                    for k in KEYS:np.testing.assert_allclose(ss[k],s[k],atol=1e-12,rtol=0)
                    nxt=sweep(s,P,N);residual=max(float(np.max(abs(nxt[k]-s[k]))) for k in ['qr','qd'])
                    if cc:assert residual<=2e-8
                else:
                    a,b=initialization(g[prefix+'miRNA'],g[prefix+'drug'],soft=False)
                    np.testing.assert_array_equal(s['qr'],a);np.testing.assert_array_equal(s['qd'],b);residual=None
                cp,cn=independent_counts(s,data['train_pairs'],data['train_y'])
                np.testing.assert_allclose(s['a'],cp+.5,atol=1e-10,rtol=1e-12)
                np.testing.assert_allclose(s['b'],cn+.5,atol=1e-10,rtol=1e-12)
                for key,q in [('gamma_r',s['qr']),('gamma_d',s['qd'])]:
                    np.testing.assert_allclose(s[key],1+q.sum(0),atol=1e-12,rtol=0)
                    np.testing.assert_allclose(q.sum(1),1,atol=1e-14,rtol=0);assert (q>=0).all()
                assert abs(cp.sum()-data['train_y'].sum())<1e-8
                assert abs(cn.sum()-(len(data['train_y'])-data['train_y'].sum()))<1e-8
                independent=profiled_elbo(s,cp,cn);saved=old['final_elbo'] if mode==start else r['fixed'][start]['elbo']
                assert abs(independent-saved)<1e-7
                records.append(dict(fold=fold,mode=mode,state_hash_passed=True,state_replay=mode==start,
                    independent_count_and_profiled_elbo_passed=True,elbo_error=abs(independent-saved),coordinate_residual=residual))
        if not frozen['all_converged']:continue
        predfile=out/f'f{fold}_predictions.npz';z=dict(np.load(predfile));ev=json.loads((out/f'f{fold}_evaluation.json').read_text())
        assert sha256(predfile)==ev['predictions_sha256']
        assert sha256(out/'fits_frozen.json')==ev['fits_frozen_sha256']
        for key in ['pairs','y','indices']:np.testing.assert_array_equal(z[key],data['val_'+key])
        base=control(fold,'unmasked10',data);np.testing.assert_array_equal(base,z['base_probability'])
        for name in plan['readouts']:
            mode=r['chosen'] if name=='learned' else name;s=dict(np.load(out/f'f{fold}_{mode}.npz'))
            bp=predict(s,data['val_pairs']);pp=fuse(base,bp,r['prior'],plan['fusion_weight'])
            np.testing.assert_allclose(bp,z['block_'+name],atol=1e-14,rtol=0)
            np.testing.assert_allclose(pp,z['prob_'+name],atol=1e-14,rtol=0)
            np.testing.assert_allclose(predict(s,data['val_pairs'][::-1])[::-1],bp,atol=1e-14,rtol=0)
            batched=np.concatenate([predict(s,part) for part in np.array_split(data['val_pairs'],7)])
            np.testing.assert_allclose(batched,bp,atol=1e-14,rtol=0)
            # Latent-category permutation is only an index relabeling.
            perm={**s,'qr':s['qr'][:,::-1],'qd':s['qd'][:,::-1],'a':s['a'][::-1,::-1],'b':s['b'][::-1,::-1]}
            np.testing.assert_allclose(predict(perm,data['val_pairs']),bp,atol=1e-14,rtol=0)
            direct=[]
            for i,j in data['val_pairs'][:17]:
                value=sum(s['qr'][i,k]*s['qd'][j,l]*s['a'][k,l]/(s['a'][k,l]+s['b'][k,l]) for k in range(16) for l in range(8))
                direct.append(value)
            np.testing.assert_allclose(direct,bp[:17],atol=1e-14,rtol=0)
            for label,pr in [('inner_metrics',pp),('block_standalone',bp)]:
                actual=metrics(data['val_y'],pr);ind=rank_metrics(data['val_y'],pr)
                for k,v in actual.items():assert abs(v-ev[label][name][k])<1e-12
                for k,v in ind.items():assert abs(v-actual[k])<1e-12
            readouts.append(dict(fold=fold,config=name,probability_replay_error=float(np.max(abs(pp-z['prob_'+name]))),
                order_batch_category_invariant=True,independent_scalar_prediction=True,independent_rank_metrics=True))
        for name in plan['controls']:
            p=control(fold,name,data);controls.append(dict(fold=fold,name=name,metrics=metrics(data['val_y'],p),reused=True))
        print('AUDIT',fold,'passed',flush=True)
    write(HERE/'verification.json',dict(status='passed',all_converged=frozen['all_converged'],parameter_records=records,
        prediction_records=readouts,reused_controls=controls,new_iterative_fit_replays=6,new_closed_form_state_checks=6,
        new_neural_gpu_replays=0,new_plmf_gpu_replays=0,plan_sha256=sha256(out/'plan.json'),fits_frozen_sha256=sha256(out/'fits_frozen.json'),
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False))

if __name__=='__main__':main()

