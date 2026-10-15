"""AUC ceiling for fixed nonnegative mixtures of audited INNER predictions.

For every positive/negative comparison that every model gets strictly wrong,
any convex mixture remains strictly wrong. If every model ties it, any mixture
ties it. Ignoring all other constraints gives a valid, generally loose upper
bound. This is NOT a ceiling for newly trained models or original-test AUC.
"""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,CLEAN,RECON,sha256,load_inner


def ceiling(predictions,y):
    p=np.flatnonzero(y==1);n=np.flatnonzero(y==0)
    wrong=np.ones((len(p),len(n)),dtype=bool);tied=wrong.copy()
    for score in predictions:
        difference=score[p,None]-score[None,n]
        wrong &= difference<0;tied &= difference==0
    mass=(int(wrong.sum())+.5*int(tied.sum()))/wrong.size
    return dict(shared_strict_errors=int(wrong.sum()),all_model_ties=int(tied.sum()),comparisons=int(wrong.size),
        auc_upper_bound=float(1-mass),common_error_mass=float(mass))


def main():
    assert json.loads((HERE/'verification.json').read_text())['status']=='passed'
    families={}
    for name,directory in [('attention','midrug_exact_attention_20261006'),('global_features','midrug_global_features_20261006'),
                           ('observed_context','midrug_observed_context_20261006'),('degree','midrug_degree_encoder_20261006'),
                           ('molecular','midrug_molecular_encoder_20261006'),
                           ('path_moments','midrug_path_moments_20261006'),('class_prior','midrug_class_prior_20261006'),
                           ('feature_completion','midrug_soft_paths_20261006'),('graph_completion',HERE.name)]:
        root=HERE.parent/directory
        assert json.loads((root/'verification.json').read_text())['status']=='passed'
        plan=json.loads((root/'results/plan.json').read_text())
        configs=plan['configs']
        families[name]=[(name+':'+config,root/'results'/f'{config}_f{{fold}}_s42_inner.npz') for config in configs]
    families['references']=[('control',CLEAN/'results/control_f{fold}_s42_inner.npz'),
                            ('unmasked10',RECON/'phase2/results/unmasked10_f{fold}_s42_inner.npz')]
    rows=[];hashes={}
    for fold in range(3):
        data=load_inner(fold);pred=[];names=[];family_rows=[]
        for family,paths in families.items():
            current=[]
            for name,template in paths:
                path=Path(str(template).format(fold=fold));hashes[str(path)]=sha256(path)
                with np.load(path) as z:
                    np.testing.assert_array_equal(z['pairs'],data['val_pairs']);np.testing.assert_array_equal(z['y'],data['val_y'])
                    np.testing.assert_array_equal(z['indices'],data['val_indices'])
                    value=z['probability'].copy()
                meta=json.loads(path.with_name(path.name.replace('_inner.npz','.json')).read_text())
                assert abs(roc_auc_score(data['val_y'],value)-meta['inner_auc'])<1e-12
                assert abs(average_precision_score(data['val_y'],value)-meta['inner_ap'])<1e-12
                current.append(value);pred.append(value);names.append(name)
            family_rows.append(dict(family=family,models=len(current),**ceiling(current,data['val_y'])))
        result=ceiling(pred,data['val_y']);base=pred[names.index('control')]
        # A fixed uniform mixture is only a sanity check for the upper bound.
        uniform=np.mean(np.stack(pred).astype(np.float64),axis=0)
        uniform_auc=float(roc_auc_score(data['val_y'],uniform));assert uniform_auc<=result['auc_upper_bound']+1e-12
        base_auc=float(roc_auc_score(data['val_y'],base))
        rows.append(dict(fold=fold,models=len(pred),model_names=names,families=family_rows,control_auc=base_auc,
            uniform_mixture_auc=uniform_auc,max_possible_gain_over_control=result['auc_upper_bound']-base_auc,**result))
    result=dict(scope='Original-training internal heldouts only; bound for fixed nonnegative weighted mixtures with positive total weight of these saved predictions, not a new-model or original-test limit',
        records=rows,mean_auc_upper_bound=float(np.mean([r['auc_upper_bound'] for r in rows])),
        mean_max_possible_gain_over_control=float(np.mean([r['max_possible_gain_over_control'] for r in rows])),
        prediction_hashes=hashes,source_sha256=sha256(HERE/'shared_errors.py'),
        used_for_checkpoint_or_config_selection=False,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    (HERE/'shared_error_bound.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['records','prediction_hashes']},indent=2))


if __name__=='__main__':main()
