"""Frozen REIFM transfer on the actual graph, with a random-weight control.

Five context folds remove all OOF query facts before reification. The released
GAT checkpoint stays frozen. Its scalar calibration sees only OOF training
outcomes. RNA candidates are scored conditional on a drug (SEED_OBJECT), the
same direction and settings for both tasks. No tabular feature bank is used.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch
import torch_geometric

from improve import non_tabular_experiment as base
from improve import typed_sheaf_experiment as graph_source
from improve.graph_signal_calibrated import make_folds, fit_calibration, apply_calibration
from improve.vendor.reifm_20260914.models import ReiFM, SEED_OBJECT
from improve.vendor.reifm_20260914.reify import reify

OUT = base.ROOT/'results'/'reification_transfer_20260914'
ASSETS = base.ROOT/'results'/'reifm_assets_20260914'
VENDOR = base.ROOT/'vendor'/'reifm_20260914'


def facts_from_physical(edges, relation, target_codes, relation_keys):
    """Undirected same-kind edges become two facts without arbitrary orientation."""
    symmetric = np.array([relation_keys[int(r)][0] == relation_keys[int(r)][2] for r in relation])
    if np.any(target_codes[symmetric] >= 0):
        raise ValueError('Target association should have two distinct node types')
    return (np.concatenate((edges, edges[::-1, symmetric]), 1),
            np.concatenate((relation, relation[symmetric])),
            np.concatenate((target_codes, target_codes[symmetric])))


def context_graph(a, omit_codes=(), device='cpu'):
    keep = ~np.isin(a['target_codes'], np.asarray(omit_codes, dtype=np.int64))
    edges, relations = a['edges'][:, keep], a['relation'][keep]
    if np.isin(a['target_codes'][keep], np.asarray(omit_codes, dtype=np.int64)).any():
        raise ValueError('Excluded query fact remains')
    graph = reify(torch.from_numpy(np.ascontiguousarray(edges)),
                  torch.from_numpy(np.ascontiguousarray(relations)),
                  int(a['num_entities']), int(a['num_relations']))
    return graph.to(device)


def model_for(kind, seed, device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = ReiFM(backbone='gat', dim=64, num_layers=12, dropout=0., agg='sum').to(device)
    if kind == 'pretrained':
        model.load_state_dict(torch.load(ASSETS/'gat_seed0'/'best.pt', map_location=device, weights_only=True), strict=True)
    elif kind != 'random':
        raise ValueError('Unknown model')
    model.eval()
    return model


@torch.no_grad()
def score_drugs(model, graph, a, drug_ids, batch_size=8):
    """Columns follow drug_ids; every column scores every RNA candidate."""
    device = next(model.parameters()).device
    drug_ids = np.asarray(drug_ids, dtype=np.int64)
    scores = []
    start = int(a['pair_offset'][0])
    stop = start+int(a['nr'])
    for i in range(0, len(drug_ids), batch_size):
        ids = torch.as_tensor(drug_ids[i:i+batch_size]+int(a['pair_offset'][1]), device=device)
        relation = torch.full_like(ids, int(a['target_relation']))
        roles = torch.full_like(ids, SEED_OBJECT)
        result = model(graph, ids, relation, roles)
        scores.append(result[:, start:stop].T.cpu().numpy())
    return np.concatenate(scores, 1)


def prepare(out):
    out = Path(out)
    if out.exists():
        raise ValueError('Use a new output directory')
    source_protocol = graph_source.protocol(graph_source.OUT)
    verified = json.loads((ASSETS/'VERIFIED.json').read_text())
    if base.digest(ASSETS/'gat_seed0'/'best.pt') != verified['checkpoint_sha256']:
        raise ValueError('Pretrained checkpoint changed')
    args = json.loads((ASSETS/'gat_seed0'/'results.json').read_text())
    out.mkdir(parents=True)
    (out/'source_snapshot').mkdir()
    own = ('reification_transfer.py', 'graph_signal_calibrated.py',
           'non_tabular_experiment.py', 'non_tabular_models.py')
    sources = {str(base.ROOT/n):base.digest(base.ROOT/n) for n in own}
    sources.update({str(VENDOR/n):base.digest(VENDOR/n) for n in ('models.py','reify.py','LICENSE','PROVENANCE.json')})
    for n in own:
        shutil.copy2(base.ROOT/n, out/'source_snapshot'/n)
    shutil.copytree(VENDOR, out/'source_snapshot'/'reifm', ignore=shutil.ignore_patterns('__pycache__'))
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), tasks=list(base.TASKS),
             model_kinds=['pretrained','random'], seed=20260914, folds=5, batch_size=8,
             checkpoint=verified, checkpoint_training_metadata=args,
             architecture='Unmodified released REIFM GAT, 64 dimensions, 12 layers, fixed six meta-relations and anonymous relation-type nodes. Object-seeded scoring for all RNA candidates.',
             adaptation='Existing heterogeneous physical graph; symmetric same-type relations represented in both orientations. No node attributes, graph-feature table or entity/relation-name text.',
             support='Remove every OOF fold query fact before constructing all six edges and anonymous fact/type nodes. Evaluation queries absent from all support graphs.',
             calibration='One scalar Platt map fitted on all OOF training scores, applied to five matching context-fold predictions. Average calibrated logits. Neural checkpoint remains frozen.',
             primary='0.5*original_GNN_logit+0.5*calibrated_REIFM_logit, fixed for both tasks; standalone results and equally calibrated random-weight control required.',
             selection='No checkpoint, orientation, mixture or parameter selection on this historical inner evaluation. If shared gate fails, do not expand to confirmation.',
             novelty='A test of pretrained fact-node vocabulary transfer, not a claim that local application is a newly invented foundation architecture. Random-weight control must be surpassed.',
             count_policy='The five contexts reuse one checkpoint, not five independently trained models or biological repeats.',
             runtime=dict(python='3.12 portable subset', torch=torch.__version__, pyg=torch_geometric.__version__,
                          official_development='Python>=3.13, torch2.12, PyG2.8; portable source subset is tested separately with strict checkpoint loading'),
             original_test_access=False, outer_query_labels_access=False,
             source_sha256=sources, input_sha256={str(ASSETS/'gat_seed0'/'best.pt'):base.digest(ASSETS/'gat_seed0'/'best.pt')})
    for task in p['tasks']:
        src = graph_source.OUT/'inputs'/task
        meta = json.loads((src/'graph.json').read_text())
        with np.load(src/'arrays.npz', allow_pickle=False) as z:
            a = {n:z[n].copy() for n in ('pairs_train','y_train','pairs_val','y_val','pair_offset','nd')}
            edges, relation, codes = facts_from_physical(z['edges'], z['relation'], z['target_codes'], meta['physical_relations'])
            a.update(edges=edges, relation=relation, target_codes=codes,
                     num_entities=np.array(int(z['node_counts'].sum())),
                     num_relations=np.array(len(meta['physical_relations'])),
                     target_relation=np.array([x[1] for x in meta['physical_relations']].index(task)),
                     nr=np.array(int(z['node_counts'][meta['node_types'].index('miRNA' if task=='MiDrug' else 'lncRNA')])))
        a['folds'] = make_folds(a['pairs_train'], a['y_train'], p['seed'], p['folds'])
        codes = a['pairs_val'][0]*int(a['nd'])+a['pairs_val'][1]
        if np.isin(codes, a['target_codes']).any():
            raise ValueError('Evaluation query visible')
        for seed in (666,0,1):
            path = base.GRAPH/('screen' if seed==666 else 'confirm')/task/'base/c0'/('s'+str(seed))/'scores.npz'
            with np.load(path, allow_pickle=False) as z:
                np.testing.assert_array_equal(a['pairs_val'], z['pairs_val'])
                np.testing.assert_array_equal(a['y_val'], z['y_val'])
                a['gnn_'+str(seed)] = z['val'].copy()
            p['input_sha256'][str(path)] = base.digest(path)
        target = out/'inputs'/task/'arrays.npz'
        target.parent.mkdir(parents=True)
        np.savez_compressed(target, **a)
        for path in (src/'arrays.npz',src/'graph.json',target):
            p['input_sha256'][str(path)] = base.digest(path)
    base.write_json(out/'protocol.json', p)
    print('FROZEN',out,flush=True)


def protocol(out):
    p = json.loads((Path(out)/'protocol.json').read_text())
    for n,h in {**p['source_sha256'],**p['input_sha256']}.items():
        if base.digest(n) != h:
            raise ValueError('Source/input changed: '+n)
    return p


def verify_files(dest, row):
    for n,h in row['files'].items():
        if base.digest(dest/n) != h:
            raise ValueError('Artifact changed: '+str(dest/n))


def run(out, task, device, kinds=None):
    out = Path(out)
    p = protocol(out)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    with np.load(out/'inputs'/task/'arrays.npz', allow_pickle=False) as z:
        a = {n:z[n].copy() for n in z.files}
    for kind in (kinds or p['model_kinds']):
        dest = out/'runs'/task/kind
        if (dest/'result.json').exists():
            verify_files(dest,json.loads((dest/'result.json').read_text()))
            continue
        dest.mkdir(parents=True,exist_ok=True)
        model = model_for(kind,p['seed'],device)
        torch.save(model.state_dict(),dest/'model.pt')
        restored = model_for(kind,p['seed']+1,device)
        restored.load_state_dict(torch.load(dest/'model.pt',map_location=device,weights_only=True),strict=True)
        oof = np.full(len(a['y_train']),np.nan)
        validation = []
        records = []
        for fold in range(p['folds']):
            path = dest/('fold_'+str(fold)+'.npz')
            record_path = dest/('fold_'+str(fold)+'.json')
            query_indices = np.flatnonzero(a['folds']==fold)
            q = a['pairs_train'][:,query_indices]
            omit = q[0]*int(a['nd'])+q[1]
            start = time.perf_counter()
            if record_path.exists():
                row = json.loads(record_path.read_text())
                if base.digest(path) != row['sha256']:
                    raise ValueError('Fold score changed')
            else:
                graph = context_graph(a,omit,device)
                scores = score_drugs(model,graph,a,np.arange(int(a['nd'])),p['batch_size'])
                if not np.isfinite(scores).all():
                    raise FloatingPointError('Nonfinite REIFM score')
                # Re-load plus reorder and batch on fixed drug columns, not selected by outcomes.
                cols = np.array([0,1,7,31,100,215])
                replay = score_drugs(restored,graph,a,cols[::-1],3)[:,::-1]
                np.testing.assert_allclose(replay,scores[:,cols],atol=1e-4,rtol=1e-4)
                err = float(np.max(np.abs(replay-scores[:,cols])))
                vq = a['pairs_val']
                np.savez_compressed(path,full=scores,oof_indices=query_indices,
                                    oof=scores[q[0],q[1]],val=scores[vq[0],vq[1]])
                row = dict(fold=fold,seconds=time.perf_counter()-start,sha256=base.digest(path),
                           replay_max_abs=err,reordered_columns_and_batches_checked=True,
                           excluded_query_count=len(q.T),facts=int(graph.num_facts),
                           all_six_edges_removed_before_reification=True)
                base.write_json(record_path,row)
                print('FOLD',task,kind,fold,round(row['seconds'],1),err,flush=True)
                del graph
            with np.load(path,allow_pickle=False) as z:
                np.testing.assert_array_equal(query_indices,z['oof_indices'])
                oof[query_indices] = z['oof']
                validation.append(z['val'].copy())
            records.append(row)
        calibration = fit_calibration(oof,a['y_train'])
        pred = apply_calibration(np.stack(validation),calibration).mean(0)
        scores = dict(pred=pred,y=a['y_val'],pairs=a['pairs_val'],oof=oof,
                      folds=a['folds'],pairs_train=a['pairs_train'],y_train=a['y_train'],fold_validation=np.stack(validation))
        for n,g in a.items():
            if n.startswith('gnn_'):
                scores[n] = g
                scores['composed_'+n] = .5*g+.5*pred
        measured = {n:base.metrics(scores['y'],v,scores['pairs']) for n,v in scores.items()
                    if n=='pred' or 'gnn_' in n}
        np.savez_compressed(dest/'scores.npz',**scores)
        base.write_json(dest/'calibration.json',calibration)
        names = ['model.pt','scores.npz','calibration.json']
        names += ['fold_'+str(f)+ext for f in range(p['folds']) for ext in ('.npz','.json')]
        base.write_json(dest/'result.json',dict(task=task,kind=kind,folds=records,
            metrics=measured,seconds=sum(x['seconds'] for x in records),
            calibration_input='OOF training only',neural_finetuning=False,
            parameters=sum(x.numel() for x in model.parameters()),
            files={n:base.digest(dest/n) for n in names}))
        print('COMPLETE',task,kind,measured,flush=True)


def report(out):
    out = Path(out)
    p = protocol(out)
    rows = []
    for task in p['tasks']:
        for kind in p['model_kinds']:
            dest = out/'runs'/task/kind
            r = json.loads((dest/'result.json').read_text())
            verify_files(dest,r)
            with np.load(dest/'scores.npz',allow_pickle=False) as z:
                for n,old in r['metrics'].items():
                    m = base.metrics(z['y'],z[n],z['pairs'])
                    for key,value in m.items():
                        if abs(value-old[key]) > 1e-12:
                            raise ValueError('Metric mismatch')
                    rows.append(dict(task=task,kind=kind,score=n,**m))
    df = pd.DataFrame(rows)
    df.to_csv(out/'metrics.csv',index=False)
    result = []
    for task in p['tasks']:
        for kind in p['model_kinds']:
            part = df[(df.task==task)&(df.kind==kind)]
            new = part[part.score.str.startswith('composed_')]
            ref = part[part.score.str.startswith('gnn_')]
            row = dict(task=task,kind=kind)
            for metric in ('auc','ap','macro_drug_ap'):
                row[metric] = float(new[metric].mean())
                row['delta_'+metric] = float(new[metric].mean()-ref[metric].mean())
                row['standalone_'+metric] = float(part[part.score=='pred'][metric].iloc[0])
            result.append(row)
    summary = pd.DataFrame(result)
    summary.to_csv(out/'summary.csv',index=False)
    def row(task,kind):
        return next(x for x in result if x['task']==task and x['kind']==kind)
    mi,ln,control = row('MiDrug','pretrained'),row('LncDrug','pretrained'),row('MiDrug','random')
    mi_gate = all(mi['delta_'+m]>0 for m in ('auc','ap'))
    ln_gate = all(ln['delta_'+m]>=-.001 for m in ('auc','ap'))
    mechanism = {m:mi[m]-control[m] for m in ('auc','ap')}
    assessment = dict(stage='Historical inner screening only',mi_performance=mi_gate,
                      lnc_mean_tolerance=ln_gate,pretrained_vs_random=mechanism,
                      shared_screen_gate=mi_gate and ln_gate and all(v>0 for v in mechanism.values()),
                      independent_biological_validation=False,frontier_innovations_confirmed=0,
                      original_test_access=False,neural_models_trained=0,
                      fitted_scalar_calibrators=len(p['tasks'])*len(p['model_kinds']))
    base.write_json(out/'ASSESSMENT.json',assessment)
    base.write_json(out/'COMPLETION.json',dict(all_metrics_recomputed=True,
        all_artifact_hashes_verified=True,context_passes=p['folds']*len(p['tasks'])*len(p['model_kinds']),
        files={n:base.digest(out/n) for n in ('metrics.csv','summary.csv','ASSESSMENT.json')}))
    print(summary.to_string(index=False))
    print(json.dumps(assessment,indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=('prepare','run','report'))
    ap.add_argument('--out-dir',default=str(OUT))
    ap.add_argument('--task',choices=base.TASKS)
    ap.add_argument('--device',default='cuda:0')
    ap.add_argument('--kinds',nargs='+',choices=('pretrained','random'))
    a = ap.parse_args()
    if a.command=='prepare':
        prepare(a.out_dir)
    elif a.command=='run':
        run(a.out_dir,a.task,a.device,a.kinds)
    else:
        report(a.out_dir)


if __name__=='__main__':
    main()
