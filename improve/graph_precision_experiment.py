"""Versioned replay/continuation of sealed graph experiments.

Reuse verified fitted parameters; preserve original float32 checkpoint
selection for both reused and new fits. Only final inference uses deterministic
aggregation and float64 probability arithmetic. Incomplete parent artifacts
remain untouched and are not considered verified or recoverable training runs.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from improve import non_tabular_experiment as base
from improve import overlap_block_experiment as common
from improve import lorentz_graph_experiment as geometry
from improve import graph_prediction_precision as precision


def prepare(parent, out, method):
    parent, out = Path(parent).resolve(), Path(out).resolve()
    p = common.protocol(parent)
    if out.exists():
        raise ValueError('Use a new output directory')
    out.mkdir(parents=True)
    shutil.copytree(parent/'source_snapshot', out/'source_snapshot')
    p.update(created_at_utc=datetime.now(timezone.utc).isoformat(), method=method,
             parent_directory=str(parent), parent_protocol_sha256=base.digest(parent/'protocol.json'),
             original_selection=p['selection'],
             final_evaluation='Deterministic torch CUDA graph reductions and cuBLAS; disabled TF32; sigmoid, posterior averaging and inverse-logit in float64. Same 1e-7 probability clamp and 3e-5 absolute/relative replay tolerance.',
             checkpoint_selection_numerics='Unmodified sealed parent train(), including original float32 inference and optimizer; final readout changes do not retroactively select a different epoch.',
             recovery='Reuse only parent fits with result.json and verified hashes. Parent incomplete fit is preserved and rerun from its seed in the new directory.',
             parent_immutable=True)
    for name in ('graph_prediction_precision.py','graph_precision_experiment.py',
                 'test_graph_prediction_precision.py'):
        shutil.copy2(base.ROOT/name, out/'source_snapshot'/name)
        p['source_sha256'][name] = base.digest(base.ROOT/name)
    for task in p['tasks']:
        source = parent/'inputs'/task/'arrays.npz'
        dest = out/'inputs'/task/'arrays.npz'
        dest.parent.mkdir(parents=True)
        shutil.copy2(source, dest)
        p['inputs'][str(source)] = base.digest(source)
        p['inputs'][str(dest)] = base.digest(dest)
    base.write_json(out/'protocol.json', p)
    print('FROZEN', out, flush=True)


def evaluate_and_save(out, p, task, kind, lr, seed, device, a, dest, origin):
    library = common if p['method'] == 'block' else geometry
    model, xs, graphs, _ = library.setup(a, p, kind, device)
    model.load_state_dict(torch.load(dest/'model.pt', map_location=device, weights_only=True), strict=True)
    pv = torch.as_tensor(a['pairs_val']+a['pair_offset'][:,None], device=device)
    def predict(query):
        if p['method'] == 'block':
            return precision.block_prediction(model, xs, graphs, query,
                                               samples=p['posterior_samples']).cpu().numpy()
        return precision.geometry_prediction(model, xs, graphs, query).cpu().numpy()
    pred = predict(pv)
    replay = predict(pv)
    reverse = predict(pv.flip(1))[::-1]
    chunk = np.concatenate([predict(pv[:, start:start+800]) for start in range(0,pv.shape[1],800)])
    errors = {}
    for label, value in (('replay',replay),('reverse',reverse),('chunk',chunk)):
        np.testing.assert_allclose(pred, value, atol=3e-5, rtol=3e-5)
        errors[label] = float(np.max(np.abs(pred-value)))
    scores = dict(y=a['y_val'], pairs=a['pairs_val'], standalone=pred)
    for ref in ('gnn','tabicl','anchor'):
        scores[ref] = a[ref+'_'+str(seed)]
        weight = p['gnn_weight'] if ref == 'gnn' else p['strong_weight']
        scores[ref+'_composed'] = (1-weight)*scores[ref]+weight*pred
    np.savez_compressed(dest/'scores.npz', **scores)
    detail = json.loads((dest/'training_detail.json').read_text())
    result = dict(task=task, kind=kind, lr=lr, seed=seed, **detail, origin=origin,
                  protocol_sha256=base.digest(out/'protocol.json'),
                  replay_max_abs=max(errors.values()), inference_checks=errors,
                  metrics={name:base.metrics(scores['y'],value,scores['pairs'])
                           for name,value in scores.items() if name not in ('y','pairs')},
                  files={n:base.digest(dest/n) for n in ('model.pt','scores.npz','trace.json','training_detail.json')})
    if origin['type'] == 'verified_parent':
        old = common.load(Path(origin['directory'])/'scores.npz')
        result['precision_audit'] = dict(
            max_logit_difference=float(np.max(np.abs(pred-old['standalone']))),
            metric_differences={name:{m:result['metrics'][name][m]-base.metrics(old['y'],old[name],old['pairs'])[m]
                                     for m in ('auc','ap')}
                                for name in ('standalone','anchor_composed')})
    base.write_json(dest/'result.json',result)
    print('VERIFIED', task, kind, lr, seed, result['epoch'], result['metrics']['anchor_composed'],
          origin['type'], errors, flush=True)
    del model, xs, graphs
    torch.cuda.empty_cache()


def run(out, task, device, stage):
    out=Path(out).resolve(); p=common.protocol(out)
    parent=Path(p['parent_directory'])
    if base.digest(parent/'protocol.json') != p['parent_protocol_sha256']:
        raise ValueError('Parent protocol changed')
    a=common.load(out/'inputs'/task/'arrays.npz')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(False)
    ph=base.digest(out/'protocol.json')
    selected = None if stage == 'screen' else json.loads((out/'SELECTION.json').read_text())
    if selected is not None and selected['protocol_sha256'] != ph:
        raise ValueError('Wrong selection protocol')
    seeds=[p['screen_seed']] if stage=='screen' else p['confirm_seeds']
    library=common if p['method']=='block' else geometry
    for kind in p['kinds']:
        lrs=p['lrs'] if stage=='screen' else [selected['chosen'][kind]['lr']]
        for lr in lrs:
            for seed in seeds:
                relative=Path('runs')/task/kind/('lr'+str(lr))/('s'+str(seed))
                dest=out/relative; source=parent/relative
                if (dest/'result.json').exists():
                    common.verify(dest,ph)
                    continue
                dest.mkdir(parents=True,exist_ok=True)
                if (source/'result.json').exists():
                    old=common.verify(source,p['parent_protocol_sha256'])
                    for name in ('model.pt','trace.json'):
                        shutil.copy2(source/name,dest/name)
                    detail={k:old[k] for k in ('epoch','seconds','parameters')}
                    origin=dict(type='verified_parent', directory=str(source),
                                result_sha256=base.digest(source/'result.json'))
                else:
                    # Training artifacts are saved before inference verification,
                    # so an inference failure cannot erase epoch/trace provenance.
                    if (dest/'training_detail.json').exists():
                        detail=json.loads((dest/'training_detail.json').read_text())
                        origin=json.loads((dest/'training_origin.json').read_text())
                        for name,sha in origin['training_files'].items():
                            if base.digest(dest/name) != sha:
                                raise ValueError('Incomplete run artifact changed')
                    else:
                        model,_,trace,detail=library.train(a,p,kind,lr,seed,device)
                        torch.save(model.state_dict(),dest/'model.pt')
                        base.write_json(dest/'trace.json',trace)
                        origin=dict(type='new_fit_original_training',
                                    parent_incomplete_checkpoint_sha256=base.digest(source/'model.pt')
                                        if (source/'model.pt').exists() else None,
                                    training_files={n:base.digest(dest/n) for n in ('model.pt','trace.json')})
                        del model
                base.write_json(dest/'training_detail.json',detail)
                base.write_json(dest/'training_origin.json',origin)
                evaluate_and_save(out,p,task,kind,lr,seed,device,a,dest,origin)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('prepare','screen','select','confirm','report'))
    parser.add_argument('--out-dir',required=True)
    parser.add_argument('--parent')
    parser.add_argument('--method',choices=('block','geometry'))
    parser.add_argument('--task',choices=base.TASKS)
    parser.add_argument('--device',default='cuda:1')
    a=parser.parse_args()
    if a.command=='prepare':
        if not a.parent or not a.method:
            parser.error('prepare requires --parent and --method')
        prepare(a.parent,a.out_dir,a.method)
    elif a.command in ('screen','confirm'):
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
        run(a.out_dir,a.task,a.device,a.command)
    elif a.command=='select':
        common.select(a.out_dir)
    else:
        common.report(a.out_dir)


if __name__=='__main__':
    main()
