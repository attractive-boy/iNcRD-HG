"""Replay checkpoint outputs and verify trained molecular/association invariances."""
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,load_inner,seed_all,sha256,load_molecules
from model import CONFIGS,make_adapter
from check import deletion_check,permutation_check,QUERY_CHECK_SOURCE


def main():
    plan=json.loads((HERE/'results/plan.json').read_text())
    for p,h in plan['source_hashes'].items():assert sha256(p)==h
    assert sha256(HERE/'molecular_inputs.npz')==plan['molecule_sha256']
    assert sha256(HERE/'input_manifest.json')==plan['molecule_manifest_sha256']
    molecules,_=load_molecules('cuda:0');records=[]
    for fold in range(3):
        data=load_inner(fold);assert not set(data['train_indices'])&set(data['val_indices'])
        assert not len(data['test_pairs'])
        paths=[(name,Path(c[str(fold)]['path']),True) for name,c in plan['controls'].items()]
        paths += [(name,HERE/'results'/f'{name}_f{fold}_s42.json',False) for name in CONFIGS]
        for name,path,reused in paths:
            r=json.loads(path.read_text());seed_all(42);a=make_adapter(data,name,'cuda:0')
            state=torch.load(path.with_suffix('.pt'),map_location='cuda:0',weights_only=True)
            assert sha256(path.with_suffix('.pt'))==r['checkpoint_sha256']
            a.model.load_state_dict(state);a.model.eval()
            assert set(a.graph.node_types)=={'miRNA','drug'} and all('lnc' not in k.lower() for k in state)
            with torch.no_grad():
                predictions=dict(probability=a.predict('val'))
                if not reused:
                    predictions.update(branch_probability=a.model.branch_logits.sigmoid().cpu().numpy(),
                                       base_probability=a.model.base_logits.sigmoid().cpu().numpy())
            errors={}
            with np.load(path.with_name(path.stem+'_inner.npz')) as z:
                for k,dkey in [('pairs','val_pairs'),('y','val_y'),('indices','val_indices')]:
                    np.testing.assert_array_equal(z[k],data[dkey])
                for k,v in predictions.items():
                    np.testing.assert_allclose(v,z[k],atol=2e-5,rtol=2e-5);errors[k]=float(abs(v-z[k]).max())
                    score=dict(auc=roc_auc_score(z['y'],z[k]),ap=average_precision_score(z['y'],z[k]))
                    expected=dict(auc=r['inner_auc'],ap=r['inner_ap']) if k=='probability' else r['branch_diagnostics'][k]
                    for metric in score:assert abs(score[metric]-expected[metric])<1e-12
            record=dict(tag=path.stem,reused_control=reused,status='passed',max_prediction_errors=errors)
            if not reused:
                assert r['settings']==CONFIGS[name] and r['source_hashes']==plan['source_hashes']
                assert r['no_lncrna'] and not r['external_data_used'] and not r['outer_validation_evaluated'] and not r['test_evaluated']
                for k,v in molecules.items():torch.testing.assert_close(state['molecular.drug.'+k],v,atol=0,rtol=0)
                curves=json.loads(path.with_name(path.stem+'_curves.json').read_text())
                assert max(curves,key=lambda x:x['selection_score'])['epoch']==r['best_epoch']
                for x in curves:
                    assert abs(x['selection_score']-.5*(x['inner_auc']+x['inner_ap']))<1e-12
                    assert x['branch_aux_weight']==CONFIGS[name]['branch_aux_weight']
                    assert x['reconstruction_weight']==CONFIGS[name]['reconstruction_weight']*min(1.,x['epoch']/50.)
                    expected=float((np.float32(x['bce'])+np.float32(x['branch_aux_weight'])*np.float32(x['branch_bce']))
                                   +np.float32(x['reconstruction_weight'])*np.float32(x['reconstruction']))
                    assert expected==x['loss']
                record['molecular_invariances']=permutation_check(a,require_nonzero=False)
                record['explicit_deletions']=deletion_check(a,data,name)
                saved_rec=a.model.reconstruction;del a.model.reconstruction
                with torch.no_grad():bare=a.predict('val')
                a.model.add_module('reconstruction',saved_rec)
                np.testing.assert_allclose(predictions['probability'],bare,atol=2e-5,rtol=2e-5)
                record['without_reconstruction_decoder_error']=float(abs(predictions['probability']-bare).max())
            records.append(record);print('PASS',path.stem,flush=True);torch.cuda.empty_cache()
    (HERE/'verification.json').write_text(json.dumps(dict(status='passed',records=records,
        audit_sources={str(p):sha256(p) for p in [HERE/'audit.py',HERE/'check.py',QUERY_CHECK_SOURCE]},
        no_lncrna=True,external_data_used=False,outer_validation_evaluated=False,test_evaluated=False),indent=2))


if __name__=='__main__':main()
