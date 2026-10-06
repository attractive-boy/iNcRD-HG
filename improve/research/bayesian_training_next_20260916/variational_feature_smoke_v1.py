"""Exercise only the new feature-gradient ablation; inherit graph-boundary evidence."""
from datetime import datetime, timezone
from pathlib import Path
import gc

import numpy as np
import torch

from improve import anchor_variational_feature_experiment_v1 as experiment
from improve import variational_feature_learning_v1 as core
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
OUT = experiment.ROOT/'results/variational_feature_smoke_v1_20260916'


def main():
    target = HERE/'VARIATIONAL_FEATURE_SMOKE_V1.json'
    if target.exists() or OUT.exists():
        raise ValueError('Preserve smoke artifacts')
    unit_path, prior_path = HERE/'VARIATIONAL_FEATURE_UNIT_V1.json',HERE/'VBLL_SMOKE_V1.json'
    unit,prior = base.check_seal(base.read_json(unit_path)),base.check_seal(base.read_json(prior_path))
    if not unit['passed'] or unit['tests_run'] != 4 or prior['core_parameters'] != core.parameters():
        raise ValueError('Implementation or common graph evidence mismatch')
    for path,digest in {**unit['source_sha256'],**prior['source_sha256'],**prior['input_sha256']}.items():
        if base.digest(path) != digest:
            raise ValueError('Checked source or input changed')
    if len(prior['fold_audits']) != 40:
        raise ValueError('Incomplete inherited boundary audit')
    OUT.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = dict(smoke=True,family='variational_feature_learning',implementation_module='variational_feature_learning_v1',
        source_sha256={'variational_feature_learning_v1.py':base.digest(core.__file__)},
        core_parameters=core.parameters(),kinds=list(core.KINDS),epochs=25,check_interval=20,weight_decay=.001,strong_weight=.1)
    records,inputs = [],{}
    for task in ('MiDrug','LncDrug'):
        source = experiment.ROOT/'results/anchor_residual_energy_v2_20260916/inputs'/task/'arrays.npz'
        inputs[str(source)] = base.digest(source)
        a = experiment.nested.load_selected(source,experiment.TRAIN_KEYS)
        query = torch.as_tensor(a['pairs_train'][:,:97]+a['pair_offset'][:,None],device='cuda:1')
        anchor = torch.as_tensor(a['anchor_oof'][:97],device='cuda:1',dtype=torch.float64)
        for kind in ('no_feature_variance',) if task == 'MiDrug' else core.KINDS:
            torch.cuda.reset_peak_memory_stats('cuda:1')
            model,_,trace,detail = experiment.fit(a,p,kind,.001,986,'cuda:1',fixed_epochs=25)
            dest = OUT/task/kind
            dest.mkdir(parents=True)
            torch.save(model.state_dict(),dest/'model.pt')
            base.write_json(dest/'trace.json',trace)
            loaded,xs,graphs,_ = experiment.setup(a,p,kind,'cuda:1')
            loaded.load_state_dict(torch.load(dest/'model.pt',map_location='cuda:1',weights_only=True))
            value = experiment.predict(model,xs,graphs,query,p,anchor).cpu().numpy()
            reverse = experiment.predict(loaded,xs,graphs,query.flip(1),p,anchor.flip(0)).cpu().numpy()[::-1]
            chunks = np.concatenate([experiment.predict(loaded,xs,graphs,query[:,i:i+50],p,anchor[i:i+50]).cpu().numpy()
                                     for i in range(0,query.shape[1],50)])
            reference = experiment.predict(loaded,xs,graphs,query,p,anchor,128).cpu().numpy()
            for actual in (reverse,chunks):
                np.testing.assert_allclose(actual,value,atol=3e-5,rtol=3e-5)
            error = float(abs(reference-value).max())
            if error > p['core_parameters']['quadrature_tolerance']:
                raise ValueError('Predictive quadrature error')
            if task == 'MiDrug':
                if detail['training_steps'] != 25 or detail['gradient_evaluations'] != 25:
                    raise ValueError('Gradient budget mismatch')
            elif trace or detail['trained_neural_model']:
                raise ValueError('Identity was treated as a neural fit')
            if any('anchor_composed' in row for row in trace):
                raise ValueError('Smoke consulted evaluation outcomes')
            record = base.seal(dict(task=task,kind=kind,**detail,
                replay_max_abs=float(max(abs(value-reverse).max(),abs(value-chunks).max())),
                quadrature_max_abs=error,peak_allocated_gib=torch.cuda.max_memory_allocated('cuda:1')/2**30,
                files={n:base.digest(dest/n) for n in ('model.pt','trace.json')}))
            base.write_json(dest/'RESULT.json',record)
            records.append(record)
            print('FEATURE_SMOKE_RECORD',task,kind,detail['training_steps'],flush=True)
            del model,loaded,xs,graphs
            gc.collect()
            torch.cuda.empty_cache()
    sources = [Path(__file__),Path(core.__file__),Path(experiment.__file__),unit_path,prior_path]
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),device='cuda:1',
        core_parameters=core.parameters(),new_short_neural_fits=1,identity_replays=3,formal_fits=0,
        performance_metrics_computed=False,reused_fold_audits=prior['fold_audits'],new_fold_audits=0,
        records=records,input_sha256=inputs,source_sha256={str(path):base.digest(path) for path in sources},
        independent_screen=False,confirmed_final_innovations=0))
    base.write_json(target,result)
    print('FEATURE_SMOKE_COMPLETE',result['sha256'],flush=True)


if __name__ == '__main__':
    main()
