"""Fixed-step residual graph validation on the existing development repartitions.

The support-only observation effects are shared by all graph families. The
primary and mandatory controls come from each sealed residual study, never
from the legacy all-controls validator's architecture dispatch.
"""
import argparse
from datetime import datetime, timezone
import gc
from pathlib import Path
import os
import shutil

import numpy as np
import pandas as pd
import torch

from improve import graph_mechanism_validate as shared
from improve import residual_graph_experiment as experiment
from improve import assess_declared_graph as audit

base, ROOT = shared.base, shared.ROOT
CACHE = 'residual_offset_inputs_v1'
BASE_TRAIN_KEYS = experiment.TRAIN_KEYS - {'offset_nodes', 'offset_intercepts'}
OWN = ('residual_graph_validate.py', 'test_residual_graph_validate.py',
       'assess_declared_graph.py', 'test_assess_declared_graph.py',
       'graph_mechanism_validate.py')


def support_effects(training):
    if set(training) - BASE_TRAIN_KEYS:
        raise ValueError('Offset fitting accepts training inputs only')
    return experiment.offset.context_offsets(training)


def prepare_offsets(out):
    out = Path(out).resolve()
    dp = shared.inputs.protocol(out)
    dest = out / CACHE
    if (dest / 'manifest.json').exists():
        return check_offsets(out, dp)
    if dest.exists():
        raise ValueError('Incomplete offset cache exists; preserve it and inspect the failure')
    dest.mkdir(parents=True)
    sources = {n: base.digest(ROOT / n) for n in OWN + ('graph_observation_offset.py',)}
    record = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                  data_protocol_sha256=dp['sha256'], source_sha256=sources,
                  penalty=1., folds=5, input_sha256={}, files={}, diagnostics={},
                  query_outcomes_used_for_offset_fitting=False,
                  scope='Support-fold row/column MAP effects; no hyperparameter selection. These are classical nuisance effects.')
    snapshot = dest / 'source_snapshot'
    snapshot.mkdir()
    for name in sources:
        shutil.copy2(ROOT / name, snapshot / name)
    for task in dp['tasks']:
        record['diagnostics'][task] = {}
        for split in dp['split_seeds']:
            a, _ = shared.inputs.arrays(out, dp, task, split)
            training = {k: v for k, v in a.items() if k in BASE_TRAIN_KEYS}
            nodes, intercepts, detail = support_effects(training)
            path = dest / task / str(split) / 'offsets.npz'
            path.parent.mkdir(parents=True)
            np.savez_compressed(path, offset_nodes=nodes, offset_intercepts=intercepts)
            source = out / 'inputs' / task / str(split) / 'arrays.npz'
            record['input_sha256'][str(source)] = base.digest(source)
            record['files'][str(path.relative_to(dest))] = base.digest(path)
            record['diagnostics'][task][str(split)] = detail
            print('SUPPORT_OFFSETS_PREPARED', task, split, flush=True)
    record = base.seal(record)
    base.write_json(dest / 'manifest.json', record)
    return record


def check_offsets(out, dp):
    dest = Path(out).resolve() / CACHE
    record = base.check_seal(base.read_json(dest / 'manifest.json'))
    if record['data_protocol_sha256'] != dp['sha256']:
        raise ValueError('Offset data protocol mismatch')
    for name, digest in record['source_sha256'].items():
        if base.digest(ROOT / name) != digest:
            raise ValueError('Offset cache source changed: ' + name)
    for name, digest in record['input_sha256'].items():
        if base.digest(name) != digest:
            raise ValueError('Offset fitting data changed')
    for name, digest in record['files'].items():
        if base.digest(dest / name) != digest:
            raise ValueError('Offset cache changed')
    return record


def freeze(out, family):
    out = Path(out).resolve()
    label = 'residual_' + family
    dest = out / 'candidates' / (label + '.json')
    if dest.exists():
        raise ValueError('Residual candidate is already frozen')
    parent = ROOT / 'results' / (label + '_20260915')
    dp = shared.inputs.protocol(out)
    p = shared.block.protocol(parent)
    assessment = audit.assess(parent, 'confirm')
    if not assessment['declared_gate']:
        raise ValueError('Residual historical confirmation gates did not pass')
    selection = base.read_json(parent / 'SELECTION.json')
    offset_record = prepare_offsets(out)
    seeds = [p['screen_seed']] + p['confirm_seeds']
    recipes = {}
    for task in p['tasks']:
        recipes[task] = {}
        for kind in p['kinds']:
            lr = selection['chosen'][kind]['lr']
            epochs = [base.read_json(parent / 'runs' / task / kind / ('lr' + str(lr)) /
                       ('s' + str(seed)) / 'result.json')['epoch'] for seed in seeds]
            recipes[task][kind] = dict(lr=lr, epochs=int(np.median(epochs)))
    cp = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        label=label, method=label, family=family, primary_kind=p['primary_kind'],
        primary_controls=p['primary_controls'], mandatory_nuisance_control='offset_composed',
        parent=str(parent), parent_protocol_sha256=base.digest(parent / 'protocol.json'),
        data_protocol_sha256=dp['sha256'], study_registry_sha256=base.digest(out / 'STUDY_REGISTRY.json'),
        reference_numerics_sha256=base.digest(out / 'REFERENCE_NUMERICS_V2.json'),
        offset_manifest_sha256=offset_record['sha256'], recipes=recipes, training_parameters=p,
        historical_assessment=assessment, historical_result_sha256=assessment['result_sha256'],
        model_seed=dp['model_seed'], strong_weight=.1,
        source_sha256={**p['source_sha256'], **{n: base.digest(ROOT / n) for n in OWN}},
        training_policy='Fixed median historical checkpoint; support-fold offsets; training dictionary excludes query data and reference predictions. Original KL annealing schedule and private RNG retained.',
        gate=p['gate'],
        gate_scope='The sealed residual protocol supersedes the legacy all-controls gate for this new candidate only. It does not change the result of any predecessor.',
        this_candidate_partition_predictions_available=False,
        previous_candidates_evaluated_on_these_partitions=True,
        original_test_access=False, confirmed_final_innovations=0))
    base.write_json(dest, cp)
    snapshot = out / 'candidate_sources' / label
    snapshot.mkdir(parents=True)
    for name in cp['source_sha256']:
        shutil.copy2(ROOT / name, snapshot / name)
    print('FROZEN_RESIDUAL_PARTITIONS', family, recipes, flush=True)


def checked(out, family):
    dp, cp = shared.candidate(out, 'residual_' + family)
    if base.digest(Path(out) / 'REFERENCE_NUMERICS_V2.json') != cp['reference_numerics_sha256']:
        raise ValueError('Reference numerical protocol changed')
    if check_offsets(out, dp)['sha256'] != cp['offset_manifest_sha256']:
        raise ValueError('Offset cache manifest mismatch')
    return dp, cp


def run(out, family, task, device):
    out = Path(out).resolve()
    dp, cp = checked(out, family)
    p, label = cp['training_parameters'], cp['label']
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for split in dp['split_seeds']:
        a, _ = shared.inputs.arrays(out, dp, task, split)
        rd = out / 'references' / task / str(split)
        r = base.read_json(rd / 'result.json')
        shared.verify_artifacts(rd, r)
        if r['data_protocol_sha256'] != dp['sha256']:
            raise ValueError('Reference data mismatch')
        reference = shared.block.load(rd / 'scores.npz')
        np.testing.assert_array_equal(reference['y'], a['y_val'])
        np.testing.assert_array_equal(reference['pairs'], a['pairs_val'])
        training = {k: v for k, v in a.items() if k in BASE_TRAIN_KEYS}
        training.update(shared.block.load(out / CACHE / task / str(split) / 'offsets.npz'))
        null = experiment.offset.offset_prediction(training, a['pairs_val'])
        for kind, recipe in cp['recipes'][task].items():
            dest = out / 'runs' / label / task / str(split) / kind
            dest.mkdir(parents=True, exist_ok=True)
            if (dest / 'result.json').exists():
                r = base.read_json(dest / 'result.json')
                if r['candidate_sha256'] != cp['sha256']:
                    raise ValueError('Existing candidate result mismatch')
                shared.verify_artifacts(dest, r)
                continue
            model, _, trace, detail = experiment.fit(training, p, kind, recipe['lr'],
                cp['model_seed'], device, fixed_epochs=recipe['epochs'])
            torch.save(model.state_dict(), dest / 'model.pt')
            base.write_json(dest / 'trace.json', trace)
            loaded, xs, graphs, _ = experiment.setup(training, p, kind, device)
            loaded.load_state_dict(torch.load(dest / 'model.pt', weights_only=True, map_location=device))
            query = torch.as_tensor(a['pairs_val'] + a['pair_offset'][:, None], device=device)
            def predict(fitted, pairs):
                return experiment.predict(fitted, xs, graphs, pairs, p).cpu().numpy()
            prediction = predict(model, query)
            replay = predict(loaded, query.flip(1))[::-1]
            chunks = np.concatenate([predict(loaded, query[:, i:i + 800]) for i in range(0, query.shape[1], 800)])
            for value in (replay, chunks):
                np.testing.assert_allclose(prediction, value, atol=3e-5, rtol=3e-5)
            scores = dict(y=a['y_val'], pairs=a['pairs_val'], standalone=prediction,
                          offset=null, anchor=reference['anchor'], tabicl=reference['tabicl'])
            for name in ('anchor', 'tabicl'):
                scores[name + '_composed'] = .9 * scores[name] + .1 * prediction
            scores['offset_composed'] = .9 * scores['anchor'] + .1 * null
            np.savez_compressed(dest / 'scores.npz', **scores)
            result = dict(candidate_sha256=cp['sha256'], data_protocol_sha256=dp['sha256'],
                task=task, split=split, kind=kind, seed=cp['model_seed'], recipe=recipe, **detail,
                replay_max_abs=float(max(abs(prediction - replay).max(), abs(prediction - chunks).max())),
                metrics={n: shared.block.base.metrics(scores['y'], v, scores['pairs'])
                         for n, v in scores.items() if n not in ('y', 'pairs')},
                files={n: base.digest(dest / n) for n in ('model.pt', 'trace.json', 'scores.npz')})
            base.write_json(dest / 'result.json', result)
            print('PARTITION_VERIFIED', label, task, split, kind, result['metrics']['anchor_composed'], flush=True)
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
                z = shared.block.load(dest / 'scores.npz')
                if cell is None:
                    cell = dict(y=z['y'], pairs=z['pairs'],
                                scores={n: z[n] for n in ('anchor', 'offset_composed')})
                    metrics.update({n: shared.block.base.metrics(z['y'], z[n], z['pairs'])
                                    for n in ('anchor', 'offset_composed')})
                for name in ('y', 'pairs'):
                    np.testing.assert_array_equal(z[name], cell[name])
                for name in ('anchor', 'offset_composed'):
                    np.testing.assert_array_equal(z[name], cell['scores'][name])
                cell['scores'][kind] = z['anchor_composed']
                for name, expected in r['metrics'].items():
                    actual = shared.block.base.metrics(z['y'], z[name], z['pairs'])
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
    performance = all(mi[m].mean() > 0 and (mi[m] > 0).sum() >= 2 and ln[m].min() >= -.001 for m in ('auc', 'ap'))
    mechanism = all(summary.loc[('MiDrug', k), m] > 0 for k in cp['primary_controls'] for m in ('auc', 'ap'))
    nuisance = all(summary.loc[('MiDrug', 'offset_composed'), m] > 0 for m in ('auc', 'ap'))
    dest = out / 'reports' / label
    dest.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest / 'metrics.csv', index=False)
    delta.to_csv(dest / 'contrasts.csv', index=False)
    summary.to_csv(dest / 'summary.csv')
    base.write_json(dest / 'endpoint_uncertainty.json', intervals)
    base.write_json(dest / 'ASSESSMENT.json', dict(candidate_sha256=cp['sha256'],
        data_protocol_sha256=dp['sha256'], created_at_utc=datetime.now(timezone.utc).isoformat(),
        primary_kind=primary, primary_controls=cp['primary_controls'],
        performance_gate=bool(performance), primary_mechanism_gate=bool(mechanism),
        offset_null_gate=bool(nuisance), additional_development_gate=bool(performance and mechanism and nuisance),
        all_metrics_recomputed=True, partitions_per_task=3, initializations_per_partition=1,
        original_test_access=False, independent_biological_validation=False, confirmed_final_innovations=0,
        limitations=['These overlapping development partitions were already evaluated for predecessor candidates.',
                     'Each candidate was fixed before its own predictions, not before all previous data use.',
                     'The classical offset and fixed fusion are not innovations.',
                     'Intervals are conditional on selected fitted models and do not correct selection or training uncertainty.',
                     'The 0.001 LncDrug tolerance is an engineering criterion, not demonstrated noise or noninferiority.',
                     'Final reserved evaluation has not been performed.'],
        result_sha256=files, artifacts={n: base.digest(dest / n) for n in
            ('metrics.csv', 'contrasts.csv', 'summary.csv', 'endpoint_uncertainty.json')}))
    print(summary.to_string(), flush=True)
    print('RESIDUAL_DECLARED_GATES', performance, mechanism, nuisance, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare-offsets', 'freeze', 'run', 'report'))
    parser.add_argument('--out', default=str(shared.data_source.OUT))
    parser.add_argument('--family', choices=tuple(experiment.LIBRARIES))
    parser.add_argument('--task', choices=('MiDrug', 'LncDrug'))
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if args.command == 'prepare-offsets':
        prepare_offsets(args.out)
    elif args.command == 'freeze':
        freeze(args.out, args.family)
    elif args.command == 'run':
        run(args.out, args.family, args.task, args.device)
    else:
        report(args.out, args.family)
