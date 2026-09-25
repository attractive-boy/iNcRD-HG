"""Fixed-step validation with the masked-teacher study's declared BCE null.

No-teacher and fixed-remask remain component diagnostics. This separate executor
does not change the earlier all-controls gate for block/geometry/state-space.
"""
import argparse
from datetime import datetime, timezone
import gc
import os
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from improve import graph_mechanism_validate as shared
from improve import masked_teacher_experiment as experiment
from improve.graph_candidate_runner import TRAIN_KEYS

base = shared.base
ROOT = shared.ROOT
LABEL = 'masked_teacher'
OWN = ('masked_teacher_validate.py', 'test_masked_teacher_validate.py', 'graph_candidate_runner.py',
       'graph_mechanism_validate.py')


def freeze(out, parent=experiment.OUT):
    out, parent = Path(out).resolve(), Path(parent).resolve()
    dest = out / 'candidates' / (LABEL + '.json')
    if dest.exists():
        raise ValueError('Masked teacher candidate is already frozen')
    dp = shared.inputs.protocol(out)
    p = shared.block.protocol(parent)
    selection = base.read_json(parent / 'SELECTION.json')
    assert selection['protocol_sha256'] == base.digest(parent / 'protocol.json')
    assert p['primary_kind'] == 'full' and p['primary_controls'] == ['bce']
    table = shared.block.all_rows(parent)
    seeds = [p['screen_seed']] + p['confirm_seeds']
    recipes, files, contrasts = {}, {}, []
    for task in p['tasks']:
        recipes[task] = {}
        for kind in p['kinds']:
            lr = selection['chosen'][kind]['lr']
            rows = table[(table.task == task) & (table.kind == kind) & (table.lr == lr)]
            a = rows[rows.score == 'anchor_composed'].set_index('seed').loc[seeds]
            b = rows[rows.score == 'anchor'].set_index('seed').loc[seeds]
            delta = a[['auc', 'ap']] - b[['auc', 'ap']]
            recipes[task][kind] = dict(lr=lr, epochs=int(np.median(a.epoch)))
            contrasts.append(dict(task=task, kind=kind,
                **{m: float(delta[m].mean()) for m in ('auc', 'ap')},
                **{m + '_wins': int((delta[m] > 0).sum()) for m in ('auc', 'ap')},
                **{m + '_min': float(delta[m].min()) for m in ('auc', 'ap')}))
            for seed in seeds:
                path = parent / 'runs' / task / kind / ('lr' + str(lr)) / ('s' + str(seed)) / 'result.json'
                files[str(path)] = base.digest(path)
    c = {(r['task'], r['kind']): r for r in contrasts}
    mi, ln = c['MiDrug', 'full'], c['LncDrug', 'full']
    performance = all(mi[m] > 0 and mi[m + '_wins'] >= 2 and ln[m + '_min'] >= -.001 for m in ('auc', 'ap'))
    mechanism = all(mi[m] > c['MiDrug', 'bce'][m] for m in ('auc', 'ap'))
    if not performance or not mechanism:
        raise ValueError('Historical declared confirmation gates did not pass')
    cp = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), label=LABEL,
        method=p['method'], primary_kind='full', primary_controls=['bce'],
        parent=str(parent), parent_protocol_sha256=base.digest(parent / 'protocol.json'),
        data_protocol_sha256=dp['sha256'], study_registry_sha256=base.digest(out / 'STUDY_REGISTRY.json'),
        reference_numerics_sha256=base.digest(out / 'REFERENCE_NUMERICS_V2.json'),
        recipes=recipes, training_parameters=p, model_seed=dp['model_seed'], strong_weight=.1,
        source_sha256={**p['source_sha256'], **{n: base.digest(ROOT / n) for n in OWN}},
        historical_result_sha256=files, historical_contrasts=contrasts,
        historical_performance_gate=True, historical_primary_mechanism_gate=True,
        primary_comparison='Complete auxiliary objective versus independently selected pure BCE, as declared in the parent protocol before its screen. Component claims require their own positive contrasts.',
        training_policy='Fixed median historical step, same auxiliary objective and private mask seed offset 17057; no query pairs, scores or outcomes enter training.',
        other_candidate_partition_results_already_seen=True, this_candidate_partition_predictions_available=False,
        original_test_access=False, confirmed_final_innovations=0))
    base.write_json(dest, cp)
    snapshot = out / 'candidate_sources' / LABEL
    snapshot.mkdir(parents=True)
    for name in cp['source_sha256']:
        shutil.copy2(ROOT / name, snapshot / name)
    print('FROZEN_MASKED_TEACHER', recipes, flush=True)


def checked(out):
    dp, cp = shared.candidate(out, LABEL)
    assert cp['reference_numerics_sha256'] == base.digest(Path(out) / 'REFERENCE_NUMERICS_V2.json')
    return dp, cp


def train_fixed(a, p, kind, recipe, seed, device):
    if set(a) - TRAIN_KEYS:
        raise ValueError('Fixed masked-teacher fitting accepts training inputs only')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, xs, graphs, folds = experiment.setup(a, p, kind, device)
    pairs = torch.as_tensor(a['pairs_train'] + a['pair_offset'][:, None], device=device)
    labels = torch.as_tensor(a['y_train'], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe['lr'], weight_decay=p['weight_decay'])
    generator = torch.Generator(device=device).manual_seed(seed + 17057)
    trace, start = [], time.monotonic()
    for step in range(1, recipe['epochs'] + 1):
        fold = (step - 1) % len(graphs)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(xs, graphs[fold], pairs[:, folds[fold]])
        bce = F.binary_cross_entropy_with_logits(prediction, labels[folds[fold]])
        auxiliary, pieces = model.auxiliary(xs, graphs[fold], generator)
        loss = bce + p['auxiliary_weight'] * auxiliary
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite fixed masked-teacher objective')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        model.update_teacher()
        if step % 20 == 0 or step == recipe['epochs']:
            trace.append(dict(step=step, loss=float(loss.detach()), bce=float(bce.detach()),
                              **{k: float(v) for k, v in pieces.items()}))
    return model, trace, dict(epochs=recipe['epochs'], seconds=time.monotonic() - start,
                              query_outcomes_used_for_fitting=False)


def run(out, task, device):
    out = Path(out)
    dp, cp = checked(out)
    p = cp['training_parameters']
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for split in dp['split_seeds']:
        a, _ = shared.inputs.arrays(out, dp, task, split)
        rd = out / 'references' / task / str(split)
        r = base.read_json(rd / 'result.json')
        shared.verify_artifacts(rd, r)
        assert r['data_protocol_sha256'] == dp['sha256']
        reference = shared.block.load(rd / 'scores.npz')
        np.testing.assert_array_equal(a['y_val'], reference['y'])
        np.testing.assert_array_equal(a['pairs_val'], reference['pairs'])
        training = {k: v for k, v in a.items() if k in TRAIN_KEYS}
        for kind, recipe in cp['recipes'][task].items():
            dest = out / 'runs' / LABEL / task / str(split) / kind
            dest.mkdir(parents=True, exist_ok=True)
            if (dest / 'result.json').exists():
                r = base.read_json(dest / 'result.json')
                assert r['candidate_sha256'] == cp['sha256']
                shared.verify_artifacts(dest, r)
                continue
            model, trace, detail = train_fixed(training, p, kind, recipe, cp['model_seed'], device)
            torch.save(model.state_dict(), dest / 'model.pt')
            base.write_json(dest / 'trace.json', trace)
            loaded, xs, graphs, _ = experiment.setup(training, p, kind, device)
            loaded.load_state_dict(torch.load(dest / 'model.pt', weights_only=True, map_location=device))
            query = torch.as_tensor(a['pairs_val'] + a['pair_offset'][:, None], device=device)

            def predict(fitted, pairs):
                return shared.precision.geometry_prediction(fitted, xs, graphs, pairs).cpu().numpy()

            prediction = predict(model, query)
            replay = predict(loaded, query.flip(1))[::-1]
            chunks = np.concatenate([predict(loaded, query[:, i:i + 800]) for i in range(0, query.shape[1], 800)])
            for value in (replay, chunks):
                np.testing.assert_allclose(prediction, value, atol=3e-5, rtol=3e-5)
            scores = dict(y=a['y_val'], pairs=a['pairs_val'], standalone=prediction,
                          anchor=reference['anchor'], tabicl=reference['tabicl'])
            for name in ('anchor', 'tabicl'):
                scores[name + '_composed'] = .9 * scores[name] + .1 * prediction
            np.savez_compressed(dest / 'scores.npz', **scores)
            result = dict(candidate_sha256=cp['sha256'], data_protocol_sha256=dp['sha256'],
                task=task, split=split, kind=kind, seed=cp['model_seed'], recipe=recipe, **detail,
                replay_max_abs=float(max(abs(prediction - replay).max(), abs(prediction - chunks).max())),
                metrics={n: shared.block.base.metrics(scores['y'], v, scores['pairs'])
                         for n, v in scores.items() if n not in ('y', 'pairs')},
                files={n: base.digest(dest / n) for n in ('model.pt', 'trace.json', 'scores.npz')})
            base.write_json(dest / 'result.json', result)
            print('PARTITION_VERIFIED', LABEL, task, split, kind, result['metrics']['anchor_composed'], flush=True)
            del model, loaded, xs, graphs
            gc.collect()
            torch.cuda.empty_cache()


def report(out):
    out = Path(out)
    dp, cp = checked(out)
    rows, differences, intervals, files = [], [], {}, {}
    for task in dp['tasks']:
        cells = []
        for split in dp['split_seeds']:
            cell = None
            metrics = {}
            for kind in cp['recipes'][task]:
                dest = out / 'runs' / LABEL / task / str(split) / kind
                r = base.read_json(dest / 'result.json')
                assert r['candidate_sha256'] == cp['sha256']
                shared.verify_artifacts(dest, r)
                files[str(dest / 'result.json')] = base.digest(dest / 'result.json')
                z = shared.block.load(dest / 'scores.npz')
                if cell is None:
                    cell = dict(y=z['y'], pairs=z['pairs'], scores={'anchor': z['anchor']})
                    metrics['anchor'] = shared.block.base.metrics(z['y'], z['anchor'], z['pairs'])
                np.testing.assert_array_equal(z['y'], cell['y'])
                np.testing.assert_array_equal(z['pairs'], cell['pairs'])
                np.testing.assert_array_equal(z['anchor'], cell['scores']['anchor'])
                cell['scores'][kind] = z['anchor_composed']
                for name, expected in r['metrics'].items():
                    actual = shared.block.base.metrics(z['y'], z[name], z['pairs'])
                    for m in ('auc', 'ap', 'macro_drug_ap'):
                        assert abs(actual[m] - expected[m]) < 1e-12
                    rows.append(dict(task=task, split=split, kind=kind, score=name, **actual))
                    if name == 'anchor_composed':
                        metrics[kind] = actual
            for control in ('anchor', 'bce', 'no_teacher', 'fixed_remask'):
                differences.append(dict(task=task, split=split, control=control,
                    **{m: metrics['full'][m] - metrics[control][m] for m in ('auc', 'ap', 'macro_drug_ap')}))
            cells.append(cell)
        intervals[task] = shared.endpoint_intervals(cells, primary='full')
        print('UNCERTAINTY_COMPLETE', LABEL, task, flush=True)
    df, delta = pd.DataFrame(rows), pd.DataFrame(differences)
    summary = delta.groupby(['task', 'control'])[['auc', 'ap', 'macro_drug_ap']].mean()
    mi = delta[(delta.task == 'MiDrug') & (delta.control == 'anchor')]
    ln = delta[(delta.task == 'LncDrug') & (delta.control == 'anchor')]
    performance = all(mi[m].mean() > 0 and (mi[m] > 0).sum() >= 2 and ln[m].min() >= -.001 for m in ('auc', 'ap'))
    mechanism = all(summary.loc[('MiDrug', 'bce'), m] > 0 for m in ('auc', 'ap'))
    dest = out / 'reports' / LABEL
    dest.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest / 'metrics.csv', index=False)
    delta.to_csv(dest / 'contrasts.csv', index=False)
    summary.to_csv(dest / 'summary.csv')
    base.write_json(dest / 'endpoint_uncertainty.json', intervals)
    base.write_json(dest / 'ASSESSMENT.json', dict(candidate_sha256=cp['sha256'],
        data_protocol_sha256=dp['sha256'], created_at_utc=datetime.now(timezone.utc).isoformat(),
        primary_kind='full', primary_controls=['bce'], performance_gate=bool(performance),
        primary_mechanism_gate=bool(mechanism), additional_development_gate=bool(performance and mechanism),
        component_superiority={k: {m: float(summary.loc[('MiDrug', k), m]) for m in ('auc', 'ap')}
                              for k in ('no_teacher', 'fixed_remask')},
        all_metrics_recomputed=True, partitions_per_task=3, initializations_per_partition=1,
        original_test_access=False, independent_biological_validation=False, confirmed_final_innovations=0,
        limitations=['These overlapping development partitions were already evaluated for earlier candidates.',
                     'This candidate was fixed before its own partition predictions, not before all previous data use.',
                     'Conditional endpoint intervals do not correct model selection or training uncertainty.',
                     'The 0.001 LncDrug tolerance is an engineering gate, not a demonstrated noise bound.',
                     'Component improvements and final reserved performance are not implied by the BCE comparison.'],
        result_sha256=files, artifacts={n: base.digest(dest / n) for n in
            ('metrics.csv', 'contrasts.csv', 'summary.csv', 'endpoint_uncertainty.json')}))
    print(summary.to_string(), flush=True)
    print('DECLARED_GATES', performance, mechanism, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('freeze', 'run', 'report'))
    parser.add_argument('--out', default=str(shared.data_source.OUT))
    parser.add_argument('--task', choices=('MiDrug', 'LncDrug'))
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if args.command == 'freeze':
        freeze(args.out)
    elif args.command == 'run':
        run(args.out, args.task, args.device)
    else:
        report(args.out)
