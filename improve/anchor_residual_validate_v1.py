"""Fixed-partition validation of nested-anchor graph residual mechanisms.

Reuses the six already sealed evaluation references. Only training-side OOF
reference fits are new. No outcome-dependent training/checkpoint selection is
performed on these overlapping development partitions.
"""
import argparse
from datetime import datetime, timezone
import gc
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from improve import graph_mechanism_validate as shared
from improve import anchor_residual_graph_v1 as experiment
from improve import assess_declared_graph as audit

base, ROOT = shared.base, shared.ROOT
CACHE = ROOT / 'results/nested_anchor_v2_20260915/partitions'
OWN = ('anchor_residual_validate_v1.py', 'graph_mechanism_validate.py',
       'assess_declared_graph.py')


def freeze(out, family, cache=CACHE):
    out, cache = Path(out).resolve(), Path(cache).resolve()
    label = 'anchor_residual_' + family + '_v1'
    dest = out / 'candidates' / (label + '.json')
    if dest.exists():
        raise ValueError('Candidate is already frozen')
    parent = ROOT / 'results' / (label + '_20260915')
    dp = shared.inputs.protocol(out)
    p = experiment.common.protocol(parent)
    assessment = audit.assess(parent, 'confirm')
    if not assessment['declared_gate']:
        raise ValueError('Historical candidate confirmation did not pass')
    selection = base.read_json(parent / 'SELECTION.json')
    if not cache.exists():
        experiment.nested.freeze(cache, 'partitions')
    cache_p = experiment.nested.protocol(cache)
    if cache_p['scope'] != 'partitions':
        raise ValueError('Wrong nested reference scope')
    recipes = {}
    for task in p['tasks']:
        recipes[task] = {}
        for kind in p['kinds']:
            lr = selection['chosen'][kind]['lr']
            epochs = [base.read_json(parent / 'runs' / task / kind / ('lr' + str(lr)) /
                ('s' + str(seed)) / 'result.json')['epoch'] for seed in [p['screen_seed']] + p['confirm_seeds']]
            recipes[task][kind] = dict(lr=lr, epochs=int(np.median(epochs)))
    cp = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        label=label, method=p['method'], family=family, primary_kind=p['primary_kind'],
        primary_controls=p['primary_controls'], parent=str(parent),
        parent_protocol_sha256=base.digest(parent / 'protocol.json'),
        data_protocol_sha256=dp['sha256'], study_registry_sha256=base.digest(out / 'STUDY_REGISTRY.json'),
        reference_numerics_sha256=base.digest(out / 'REFERENCE_NUMERICS_V2.json'),
        nested_reference=str(cache), nested_protocol_sha256=cache_p['sha256'],
        recipes=recipes, training_parameters=p, historical_assessment=assessment,
        historical_result_sha256=assessment['result_sha256'], model_seed=dp['model_seed'],
        strong_weight=.1, lnc_absolute_tolerance=.001, gate=p['gate'],
        source_sha256={**p['source_sha256'], **{n: base.digest(ROOT / n) for n in OWN}},
        training_policy='Fixed median historical steps; only training inputs and nested OOF anchor offsets accepted. Evaluation anchor is unchanged. The reference plus 0.1 mean raw residual formula is frozen.',
        this_candidate_partition_predictions_available=False,
        previous_candidates_evaluated_on_these_partitions=True,
        original_test_source_access=False, confirmed_final_innovations=0))
    dest.parent.mkdir(exist_ok=True)
    base.write_json(dest, cp)
    snapshot = out / 'candidate_sources' / label
    snapshot.mkdir(parents=True)
    for name in cp['source_sha256']:
        shutil.copy2(ROOT / name, snapshot / name)
    print('FROZEN_ANCHOR_RESIDUAL_PARTITIONS', family, recipes, flush=True)


def checked(out, family):
    out = Path(out)
    dp, cp = shared.candidate(out, 'anchor_residual_' + family + '_v1')
    if base.digest(out / 'REFERENCE_NUMERICS_V2.json') != cp['reference_numerics_sha256']:
        raise ValueError('Evaluation reference numerical protocol changed')
    if experiment.nested.protocol(cp['nested_reference'])['sha256'] != cp['nested_protocol_sha256']:
        raise ValueError('Nested training reference protocol changed')
    if base.digest(Path(cp['parent']) / 'protocol.json') != cp['parent_protocol_sha256']:
        raise ValueError('Historical training protocol changed')
    return dp, cp


def run(out, family, task, device):
    out = Path(out).resolve()
    dp, cp = checked(out, family)
    p, label = cp['training_parameters'], cp['label']
    torch.set_num_threads(4)
    for split in dp['split_seeds']:
        a, _ = shared.inputs.arrays(out, dp, task, split)
        rd = out / 'references' / task / str(split)
        r = base.read_json(rd / 'result.json')
        shared.verify_artifacts(rd, r)
        if r['data_protocol_sha256'] != dp['sha256']:
            raise ValueError('Evaluation reference input mismatch')
        reference = experiment.common.load(rd / 'scores.npz')
        np.testing.assert_array_equal(reference['y'], a['y_val'])
        np.testing.assert_array_equal(reference['pairs'], a['pairs_val'])
        training = {k: v for k, v in a.items() if k in experiment.TRAIN_KEYS}
        training['anchor_oof'], offset_manifest = experiment.offset_arrays(cp['nested_reference'], task, split, training)
        for kind, recipe in cp['recipes'][task].items():
            dest = out / 'runs' / label / task / str(split) / kind
            dest.mkdir(parents=True, exist_ok=True)
            if (dest / 'result.json').exists():
                r = base.read_json(dest / 'result.json')
                if r['candidate_sha256'] != cp['sha256']:
                    raise ValueError('Existing candidate output mismatch')
                shared.verify_artifacts(dest, r)
                continue
            model, _, trace, detail = experiment.fit(training, p, kind, recipe['lr'],
                cp['model_seed'], device, fixed_epochs=recipe['epochs'])
            torch.save(model.state_dict(), dest / 'model.pt')
            base.write_json(dest / 'trace.json', trace)
            loaded, xs, graphs, _ = experiment.setup(training, p, kind, device)
            loaded.load_state_dict(torch.load(dest / 'model.pt', map_location=device, weights_only=True))
            query = torch.as_tensor(a['pairs_val'] + a['pair_offset'][:, None], device=device)
            value = experiment.predict(model, xs, graphs, query, p).cpu().numpy()
            replay = experiment.predict(loaded, xs, graphs, query.flip(1), p).cpu().numpy()[::-1]
            chunks = np.concatenate([experiment.predict(loaded, xs, graphs, query[:, i:i + 800], p).cpu().numpy()
                for i in range(0, query.shape[1], 800)])
            for actual in (replay, chunks):
                np.testing.assert_allclose(value, actual, atol=3e-5, rtol=3e-5)
            scores = dict(y=a['y_val'], pairs=a['pairs_val'], standalone=value,
                anchor=reference['anchor'], tabicl=reference['tabicl'], offset=reference['anchor'],
                offset_composed=reference['anchor'])
            for name in ('anchor', 'tabicl'):
                scores[name + '_composed'] = scores[name] + .1 * value
            np.savez_compressed(dest / 'scores.npz', **scores)
            result = dict(candidate_sha256=cp['sha256'], data_protocol_sha256=dp['sha256'],
                nested_offset_manifest_sha256=base.digest(offset_manifest),
                task=task, split=split, kind=kind, seed=cp['model_seed'], recipe=recipe, **detail,
                replay_max_abs=float(max(abs(value - replay).max(), abs(value - chunks).max())),
                metrics={n: experiment.common.base.metrics(scores['y'], v, scores['pairs'])
                         for n, v in scores.items() if n not in ('y', 'pairs')},
                files={n: base.digest(dest / n) for n in ('model.pt', 'trace.json', 'scores.npz')})
            base.write_json(dest / 'result.json', result)
            print('ANCHOR_RESIDUAL_PARTITION', family, task, split, kind,
                  result['metrics']['anchor_composed'], flush=True)
            del model, loaded, xs, graphs
            gc.collect()
            torch.cuda.empty_cache()


def report(out, family):
    out = Path(out).resolve()
    dp, cp = checked(out, family)
    label, primary = cp['label'], cp['primary_kind']
    rows, differences, intervals, files = [], [], {}, {}
    for task in dp['tasks']:
        cells = []
        for split in dp['split_seeds']:
            cell, metrics = None, {}
            for kind in cp['recipes'][task]:
                dest = out / 'runs' / label / task / str(split) / kind
                r = base.read_json(dest / 'result.json')
                if r['candidate_sha256'] != cp['sha256']:
                    raise ValueError('Candidate result mismatch')
                shared.verify_artifacts(dest, r)
                files[str(dest / 'result.json')] = base.digest(dest / 'result.json')
                z = experiment.common.load(dest / 'scores.npz')
                if cell is None:
                    cell = dict(y=z['y'], pairs=z['pairs'], scores={n: z[n] for n in ('anchor', 'offset_composed')})
                    metrics.update({n: experiment.common.base.metrics(z['y'], z[n], z['pairs'])
                                    for n in ('anchor', 'offset_composed')})
                for name in ('y', 'pairs'):
                    np.testing.assert_array_equal(z[name], cell[name])
                for name in ('anchor', 'offset_composed'):
                    np.testing.assert_array_equal(z[name], cell['scores'][name])
                np.testing.assert_array_equal(z['anchor'], z['offset_composed'])
                cell['scores'][kind] = z['anchor_composed']
                for name, expected in r['metrics'].items():
                    actual = experiment.common.base.metrics(z['y'], z[name], z['pairs'])
                    for m in ('auc', 'ap', 'macro_drug_ap'):
                        if abs(actual[m] - expected[m]) >= 1e-12:
                            raise ValueError('Stored metric mismatch')
                    rows.append(dict(task=task, split=split, kind=kind, score=name, **actual))
                    if name == 'anchor_composed':
                        metrics[kind] = actual
            for control in ['anchor', 'offset_composed'] + [k for k in cp['recipes'][task] if k != primary]:
                differences.append(dict(task=task, split=split, control=control,
                    **{m: metrics[primary][m] - metrics[control][m] for m in ('auc', 'ap', 'macro_drug_ap')}))
            cells.append(cell)
        intervals[task] = shared.endpoint_intervals(cells, primary=primary)
        print('UNCERTAINTY_COMPLETE', label, task, flush=True)
    df, delta = pd.DataFrame(rows), pd.DataFrame(differences)
    summary = delta.groupby(['task', 'control'])[['auc', 'ap', 'macro_drug_ap']].mean()
    mi = delta[(delta.task == 'MiDrug') & (delta.control == 'anchor')]
    ln = delta[(delta.task == 'LncDrug') & (delta.control == 'anchor')]
    performance = all(mi[m].mean() > 0 and (mi[m] > 0).sum() >= 2 and
                      ln[m].min() >= -cp['lnc_absolute_tolerance'] for m in ('auc', 'ap'))
    mechanism = all(summary.loc[('MiDrug', k), m] > 0 for k in cp['primary_controls'] for m in ('auc', 'ap'))
    nuisance = all(summary.loc[('MiDrug', 'offset_composed'), m] > 0 for m in ('auc', 'ap'))
    dest = out / 'reports' / label
    dest.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest / 'metrics.csv', index=False)
    delta.to_csv(dest / 'contrasts.csv', index=False)
    summary.to_csv(dest / 'summary.csv')
    base.write_json(dest / 'endpoint_uncertainty.json', intervals)
    base.write_json(dest / 'ASSESSMENT.json', dict(candidate_sha256=cp['sha256'],
        data_protocol_sha256=dp['sha256'], primary_kind=primary, primary_controls=cp['primary_controls'],
        performance_gate=bool(performance), primary_mechanism_gate=bool(mechanism),
        offset_null_gate=bool(nuisance), additional_development_gate=bool(performance and mechanism and nuisance),
        all_metrics_recomputed=True, partitions_per_task=3, initializations_per_partition=1,
        original_test_source_access=False, confirmed_final_innovations=0,
        limitations=['Previously exposed overlapping development partitions, not independent cohorts.',
            'Conditional endpoint bootstrap does not cover model selection or training uncertainty.',
            'The 0.001 LncDrug threshold is an engineering tolerance, not demonstrated statistical noninferiority.',
            'Nested reference fitting and fixed residual shrinkage are not new mechanism families.',
            'Frozen outer recheck remains required.'], result_sha256=files,
        artifacts={n: base.digest(dest / n) for n in ('metrics.csv', 'contrasts.csv', 'summary.csv', 'endpoint_uncertainty.json')}))
    print(summary.to_string(), flush=True)
    print('ANCHOR_RESIDUAL_GATES', performance, mechanism, nuisance, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze', 'run', 'report'))
    parser.add_argument('--out', default=str(shared.data_source.OUT))
    parser.add_argument('--family', choices=tuple(experiment.LIBRARIES), required=True)
    parser.add_argument('--task', choices=('MiDrug', 'LncDrug'))
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--cache', default=str(CACHE))
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if args.command == 'freeze':
        freeze(args.out, args.family, args.cache)
    elif args.command == 'run':
        run(args.out, args.family, args.task, args.device)
    else:
        report(args.out, args.family)
