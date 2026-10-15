"""Replay selected models, independently recompute selection, physically remove queries."""
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from common import HERE, load_inner, seed_all, sha256
from model import CONFIGS, make_adapter
from check import deletion_check

def main():
    assert not (HERE/'verification.json').exists()
    plan = json.loads((HERE/'results/plan.json').read_text())
    for p,h in {**plan['source_hashes'], **plan['input_hashes']}.items(): assert sha256(p)==h, p
    records=[]
    for fold in range(3):
        data=load_inner(fold)
        assert not set(data['train_indices']) & set(data['val_indices'])
        assert len(data['test_pairs']) == 0
        for name in CONFIGS:
            tag=f'{name}_f{fold}_s42'; root=HERE/'results'
            r=json.loads((root/f'{tag}.json').read_text())
            assert r['source_hashes']==plan['source_hashes'] and r['settings']==CONFIGS[name]
            assert r['plan_sha256']==sha256(root/'plan.json')
            assert sha256(root/f'{tag}.pt')==r['checkpoint_sha256']
            seed_all(42); a=make_adapter(data,name,'cuda:0')
            state=torch.load(root/f'{tag}.pt', map_location='cuda:0', weights_only=True)
            a.model.load_state_dict(state); a.model.eval()
            assert all('lnc' not in key.lower() for key in state)
            with torch.no_grad(): p=a.predict('val')
            with np.load(root/f'{tag}_inner.npz') as z:
                for key, source in [('pairs','val_pairs'),('y','val_y'),('indices','val_indices')]:
                    np.testing.assert_array_equal(z[key],data[source])
                np.testing.assert_allclose(p,z['probability'],atol=2e-5,rtol=2e-5)
                error=float(abs(p-z['probability']).max())
                assert abs(roc_auc_score(z['y'],z['probability'])-r['inner_auc'])<1e-12
                assert abs(average_precision_score(z['y'],z['probability'])-r['inner_ap'])<1e-12
            curves=json.loads((root/f'{tag}_curves.json').read_text())
            assert [x['epoch'] for x in curves]==list(range(1,len(curves)+1))
            selected=max(curves,key=lambda x:.5*(x['inner_auc']+x['inner_ap']))
            assert selected['epoch']==r['best_epoch']
            assert selected['inner_auc']==r['inner_auc'] and selected['inner_ap']==r['inner_ap']
            for x in curves:
                assert x['selection_score']==.5*(x['inner_auc']+x['inner_ap'])
                assert x['weight']==10.*min(1.,x['epoch']/50.)
                assert x['loss']==float(np.float32(x['bce'])+np.float32(x['weight'])*np.float32(x['reconstruction']))
            with torch.no_grad():
                refiner=a.model.encoder.refiner
                saved_features=refiner.features.clone()
                shift=refiner.projection(saved_features)
                logits=a.logits('val')
                refiner.features.zero_()
                zero_logits=a.logits('val')
                refiner.features.copy_(saved_features)
                text_diagnostics=dict(drug_input_shift_rms=float(shift.square().mean().sqrt()),
                    zero_text_logit_delta_rms=float((logits-zero_logits).square().mean().sqrt()),
                    zero_text_mean_abs_probability_delta=float((logits.sigmoid()-zero_logits.sigmoid()).abs().mean()))
            deletion=deletion_check(a,data,name)
            del a.model.reconstruction
            with torch.no_grad(): bare=a.predict('val')
            np.testing.assert_allclose(p,bare,atol=2e-5,rtol=2e-5)
            records.append(dict(tag=tag,status='passed',max_prediction_error=error,
                                max_decoder_removal_error=float(abs(p-bare).max()),explicit_deletions=deletion,
                                fixed_model_text_diagnostics=text_diagnostics))
            print('PASS',tag,flush=True)
    (HERE/'verification.json').write_text(json.dumps(dict(status='passed',records=records,
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))
if __name__=='__main__': main()
