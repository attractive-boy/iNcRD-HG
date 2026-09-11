"""Bounded rescue of support adaptation while retaining the strong GNN prior.

Four configurations per baseline/candidate; confirm the two best with two
additional seeds. Mechanistic training controls run only for tasks that clear
the predeclared performance screen. No outer or test evaluation is available.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from improve import pair_exchange as parent
from improve import support_operator_pilot as previous
from improve.dual_pair_tuning import choose_epoch
from improve.expression_scale_mechanism import state_digest
from improve.first_round_features import json_digest, sha256, write_json
from improve.local_support_modules import LocalSupportModel
from improve.network_feature_ablation import score
from improve.pair_exchange_modules import remove_target_pairs, sample_queries


SEARCH_ARMS = ('pca_query', 'case_joint')
CONTROLS = ('local_mlp', 'case_drug')
OWN_FILES = ('local_support_tuning.py', 'local_support_modules.py', 'test_local_support.py')


def sources():
    return dict(previous.sources(), **{name: sha256(Path(__file__).with_name(name)) for name in OWN_FILES})


def grid(smoke):
    common = dict(topk=16, ridge=.1, temperature=.2, max_correction=.25, branch_lr_multiplier=1.)
    if smoke:
        return {arm: [dict(common, id='smoke', lr_multiplier=1., weight_decay=0.)] for arm in SEARCH_ARMS}
    baseline, candidate = [], []
    for i, (lr, wd) in enumerate((lr, wd) for lr in (.3, 1.) for wd in (0., 1e-4)):
        baseline.append(dict(common, id='b'+str(i), lr_multiplier=lr, weight_decay=wd))
        candidate.append(dict(common, id='c'+str(i), lr_multiplier=lr, weight_decay=wd,
                              branch_lr_multiplier=3. if lr == .3 else 1.,
                              topk=16 if wd == 0 else 64, ridge=.1 if wd == 0 else 1.,
                              max_correction=.25 if wd == 0 else .5))
    return dict(pca_query=baseline, case_joint=candidate)


def prepare(args):
    out, reference = Path(args.out_dir).resolve(), Path(args.reference_dir).resolve()
    if (out / 'protocol.json').exists() or (out / 'runs').exists():
        raise ValueError('Preserve existing experiment; use a new directory')
    ref = previous.load_protocol(reference)
    verification = json.loads((reference / 'VALIDATION.json').read_text())
    if not verification['passed'] or verification['smoke']:
        raise ValueError('A verified full prototype screen is required')
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), smoke=bool(args.smoke),
        reference_dir=str(reference), reference_protocol_sha256=ref['sha256'],
        reference_results_sha256=sha256(reference / 'results.csv'), tasks=ref['tasks'],
        training={t: dict(c, train_epochs=3 if args.smoke else c['train_epochs']) for t, c in ref['training'].items()},
        search_arms=list(SEARCH_ARMS), control_arms=list(CONTROLS), configurations=grid(args.smoke),
        screening_seed=666, confirmation_seeds=[] if args.smoke else [0, 1],
        finalists=1 if args.smoke else 2, smoothing_window=2 if args.smoke else 5,
        query_fraction=.5, ramp_fraction=.2, rank=16,
        selection='Screen every configuration with seed 666 by peak trailing-5 internal mean(AUC,AP). Confirm top two per arm with seeds 0 and 1. Select one configuration per arm by mean of the three per-run peak smoothed scores. Each run retains its own selected checkpoint.',
        controls='Only tasks with positive mean AUC AND AP and >=2/3 positive paired mean-metric deltas versus BOTH the retuned original and the previous original-model reference receive local_mlp/case_drug training controls. Controls use the selected candidate recipe, all three seeds, the same epoch cap and checkpoint rule, without independent hyperparameter search.',
        optimizer='RMSprop alpha=.99, unweighted BCE; baseline and candidate each receive four recipes; all base parameters retained; branch learning-rate multiplier is frozen by recipe.',
        graph_policy='All arms retain the original graph schema and concat decoder. Each training query positive is removed from target and reverse edges. Stage graph contains fitting positives only; auxiliary/other-task relations remain frozen.',
        memory_policy='Support is only the fitting set minus the entire current query batch, including negative queries. Raw input features choose cases; no held-out labels enter memory.',
        residual_policy='Support prior predictions are detached and conditioned on the current support graph, which can contain support positives. These are conditional fitting errors, not out-of-fold calibration residuals.',
        mechanism='Joint RNA/drug retrieval plus a local regularized logistic Newton update; tanh bounds logit corrections; no performance guarantee from the bound.',
        limitations=['This internal holdout was already used to reject the first prototype; adaptive exploratory reuse.',
                     'Three initializations are not independent data splits; no test or outer validation is read.',
                     'Two best recipes are chosen from one screening seed; configurations are not exhaustively searched.',
                     'Allocated and effective control capacities differ; control hyperparameters are not separately optimized.',
                     'Unknown sampled zero labels are not experimentally confirmed sensitivity.',
                     'Support residuals are not out-of-fold; the current query is still excluded from both memory and graph.'],
        no_outer_evaluation=True, no_test_evaluation=True, source_sha256=sources(), files={},
        reference_means={}, audit=ref['audit'])
    old = pd.read_csv(reference / 'results.csv')
    for task in p['tasks']:
        rows = old[(old.task == task) & (old.arm == 'pca_query')].sort_values('seed')
        p['reference_means'][task] = dict(auc=float(rows.inner_auc.mean()), ap=float(rows.inner_aupr.mean()),
                                         by_seed={str(int(r.seed)): dict(auc=r.inner_auc, ap=r.inner_aupr) for r in rows.itertuples()})
    out.mkdir(parents=True, exist_ok=True)
    for task in p['tasks']:
        dest = out / 'inputs' / task
        dest.mkdir(parents=True, exist_ok=True)
        for name in ('inner.npz', 'indices.npz', 'train_graph.pth'):
            shutil.copy2(reference / 'inputs' / task / name, dest / name)
            p['files'][str((dest / name).relative_to(out))] = sha256(dest / name)
    snapshot = out / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for name in p['source_sha256']:
        shutil.copy2(Path(__file__).with_name(name), snapshot / name)
    p['sha256'] = json_digest(p)
    write_json(out / 'protocol.json', p)
    print('FROZEN', out, 'smoke' if args.smoke else '16 screening + 16 confirmation + conditional controls', flush=True)


def load_protocol(out):
    out = Path(out)
    p = json.loads((out / 'protocol.json').read_text())
    if p['sha256'] != json_digest({k: v for k, v in p.items() if k != 'sha256'}):
        raise ValueError('Protocol changed')
    if sources() != p['source_sha256']:
        raise ValueError('Frozen source changed')
    for name, digest in p['files'].items():
        if sha256(out / name) != digest:
            raise ValueError('Input changed: ' + name)
    return p


def read_decision(out, filename, p):
    value = json.loads((Path(out) / filename).read_text())
    if value['sha256'] != json_digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('Decision changed')
    if value['protocol_sha256'] != p['sha256']:
        raise ValueError('Decision belongs to a different protocol')
    return value


def freeze_decision(out, filename, value):
    path = Path(out) / filename
    if path.exists():
        raise FileExistsError('Preserve frozen decision: ' + str(path))
    value['sha256'] = json_digest(value)
    write_json(path, value)


def directory(out, task, arm, config_id, seed):
    return Path(out) / 'runs' / task / arm / config_id / ('s' + str(seed))


def jobs(out, p, stage):
    if stage == 'screen':
        for task in p['tasks']:
            for arm in SEARCH_ARMS:
                for cfg in p['configurations'][arm]:
                    yield task, arm, cfg, p['screening_seed']
    elif stage == 'confirm':
        finalists = read_decision(out, 'finalists.json', p)
        for task in p['tasks']:
            for arm in SEARCH_ARMS:
                for cfg in finalists['chosen'][task][arm]:
                    for seed in p['confirmation_seeds']:
                        yield task, arm, cfg, seed
    elif stage == 'controls':
        selected = read_decision(out, 'selection.json', p)
        for task in selected['control_tasks']:
            cfg = selected['chosen'][task]['case_joint']['config']
            for arm in CONTROLS:
                for seed in [p['screening_seed']] + p['confirmation_seeds']:
                    yield task, arm, cfg, seed
    else:
        raise ValueError(stage)


def build(arrays, graph, task, arm, seed, cfg, p, device):
    base, data, pairs, weights = parent.make_model(arrays, graph, task, 'pca', seed, device)
    torch.manual_seed(seed + 941731)
    model = LocalSupportModel(base, data, p['training'][task]['rna'], cfg, arm, p['rank']).to(device)
    return model, data, pairs, weights


def optimizer_for(model, base_lr, cfg):
    lr = base_lr * cfg['lr_multiplier']
    groups = [dict(params=list(model.base.parameters()), lr=lr)]
    if model.adapter is not None:
        groups.append(dict(params=list(model.adapter.parameters()), lr=lr * cfg['branch_lr_multiplier']))
    return torch.optim.RMSprop(groups, lr=lr, alpha=.99, weight_decay=cfg['weight_decay'])


@torch.no_grad()
def predict(model, data, pairs, weights, labels, arrays, diagnostics=False):
    model.eval()
    complete, prior, correction = model(data.x_dict, data.edge_index_dict, pairs['val'], weights,
                                         pairs['train'], labels, return_components=True)
    outputs = dict(val=complete.cpu().numpy(), prior=prior.cpu().numpy(), correction=correction.cpu().numpy())
    if diagnostics and model.mode == 'case_joint':
        changed = previous.shuffled_labels(arrays, labels.device)
        outputs['shuffled_labels'] = model(data.x_dict, data.edge_index_dict, pairs['val'], weights,
                                            pairs['train'], changed).cpu().numpy()
        for intervention in ('drug_only', 'uniform_selected'):
            outputs[intervention] = model(data.x_dict, data.edge_index_dict, pairs['val'], weights,
                                           pairs['train'], labels, intervention=intervention).cpu().numpy()
    return outputs


def train(arrays, graph, task, arm, seed, cfg, p, device):
    started = time.monotonic()
    model, data, pairs, weights = build(arrays, graph, task, arm, seed, cfg, p, device)
    initial, base_initial = state_digest(model), state_digest(model.base)
    optimizer = optimizer_for(model, p['training'][task]['lr'], cfg)
    labels = torch.from_numpy(arrays['y_train']).to(device)
    epochs = p['training'][task]['train_epochs']
    best, state, saved, selected = -np.inf, None, None, None
    trace, active = [], set()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        query = sample_queries(arrays['y_train'], p['query_fraction'], (seed + 1) * 1000003 + epoch)
        held = arrays['pairs_train'][:, query[arrays['y_train'][query] == 1]]
        edges, step_weights = remove_target_pairs(data.edge_index_dict, weights, task,
            p['training'][task]['rna'], data['drug'].num_nodes, held)
        ramp = min(1., (epoch - 1) / max(1., p['ramp_fraction'] * epochs))
        logits = model(data.x_dict, edges, pairs['train'][:, query], step_weights, pairs['train'], labels, ramp=ramp)
        loss = F.binary_cross_entropy_with_logits(logits, labels[query])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite loss')
        loss.backward()
        active.update(n for n, value in model.named_parameters() if value.grad is not None)
        optimizer.step()
        outputs = predict(model, data, pairs, weights, labels, arrays)
        auc, ap = score(arrays['y_val'], outputs['val'])
        trace.append(dict(epoch=epoch, loss=float(loss.detach()), queries=len(query), ramp=ramp,
                          inner_auc=auc, inner_aupr=ap, criterion=(auc + ap) / 2,
                          correction_mean_abs=float(np.mean(np.abs(outputs['correction'])))))
        if epoch >= p['smoothing_window']:
            smooth = float(np.mean([r['criterion'] for r in trace[-p['smoothing_window']:]]))
            if smooth > best:
                best, selected, saved = smooth, epoch, outputs['val'].copy()
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 100 == 0:
            print('EPOCH', task, arm, cfg['id'], seed, epoch, round((auc + ap) / 2, 6), flush=True)
    model.load_state_dict(state)
    outputs = predict(model, data, pairs, weights, labels, arrays, True)
    np.testing.assert_allclose(outputs['val'], saved, atol=5e-5, rtol=5e-5)
    restore_error = float(np.max(np.abs(outputs['val'] - saved)))
    outputs['val'] = saved
    auc, ap = score(arrays['y_val'], saved)
    result = dict(task=task, arm=arm, seed=seed, config=cfg, config_id=cfg['id'],
                  inner_auc=auc, inner_aupr=ap, criterion=(auc + ap) / 2,
                  selected_epoch=selected, selection_score=best, trained_epochs=epochs,
                  initial_state_sha256=initial, initial_base_sha256=base_initial,
                  params=sum(v.numel() for v in model.parameters()),
                  parameters_with_gradient=sum(v.numel() for n, v in model.named_parameters() if n in active),
                  correction_mean_abs=float(np.mean(np.abs(outputs['correction']))),
                  restore_max_abs=restore_error, elapsed_seconds=time.monotonic()-started,
                  protocol_sha256=p['sha256'], device=device)
    return result, outputs, trace, state


def run(args):
    out = Path(args.out_dir)
    p = load_protocol(out)
    for task in [args.task] if args.task else p['tasks']:
        with (out / (task + '.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            arrays, graph = parent.load_stage(out, p, task, 'inner')
            for t, arm, cfg, seed in jobs(out, p, args.stage):
                if t != task:
                    continue
                dest = directory(out, task, arm, cfg['id'], seed)
                if (dest / 'result.json').exists():
                    parent.checked_result(dest, p)
                    print('SKIP', task, arm, cfg['id'], seed, flush=True)
                    continue
                dest.mkdir(parents=True, exist_ok=True)
                print('START', task, arm, cfg['id'], seed, flush=True)
                result, outputs, trace, state = train(arrays, graph, task, arm, seed, cfg, p, args.device)
                np.savez_compressed(dest / 'scores.npz', **outputs, pairs_val=arrays['pairs_val'], y_val=arrays['y_val'])
                pd.DataFrame(trace).to_csv(dest / 'trace.csv', index=False)
                torch.save(state, dest / 'checkpoint.pt')
                result['artifacts_sha256'] = {name: sha256(dest / name) for name in ('scores.npz', 'trace.csv', 'checkpoint.pt')}
                write_json(dest / 'result.json', result)
                print('DONE', task, arm, cfg['id'], seed, '%.6f/%.6f epoch %d %.1fs' % (
                    result['inner_auc'], result['inner_aupr'], result['selected_epoch'],
                    result['elapsed_seconds']), flush=True)


def finalists(args):
    out, chosen = Path(args.out_dir), {}
    p = load_protocol(out)
    for task in p['tasks']:
        chosen[task] = {}
        for arm in SEARCH_ARMS:
            ranked = []
            for cfg in p['configurations'][arm]:
                r = parent.checked_result(directory(out, task, arm, cfg['id'], p['screening_seed']), p)
                ranked.append((r['selection_score'], cfg['id'], cfg))
            ranked.sort(key=lambda value: (-value[0], value[1]))
            chosen[task][arm] = [r[2] for r in ranked[:p['finalists']]]
    freeze_decision(out, 'finalists.json', dict(protocol_sha256=p['sha256'], chosen=chosen))
    print(json.dumps(chosen, indent=2), flush=True)


def select(args):
    out, chosen, gates = Path(args.out_dir), {}, {}
    p = load_protocol(out)
    candidates = read_decision(out, 'finalists.json', p)
    seeds = [p['screening_seed']] + p['confirmation_seeds']
    for task in p['tasks']:
        chosen[task] = {}
        for arm in SEARCH_ARMS:
            ranked = []
            for cfg in candidates['chosen'][task][arm]:
                records = [parent.checked_result(directory(out, task, arm, cfg['id'], seed), p) for seed in seeds]
                ranked.append(dict(config=cfg, score=float(np.mean([r['selection_score'] for r in records])),
                                   mean_auc=float(np.mean([r['inner_auc'] for r in records])),
                                   mean_ap=float(np.mean([r['inner_aupr'] for r in records])),
                                   by_seed={str(r['seed']): dict(auc=r['inner_auc'], ap=r['inner_aupr']) for r in records}))
            ranked.sort(key=lambda value: (-value['score'], value['config']['id']))
            chosen[task][arm] = ranked[0]
        candidate, baseline = chosen[task]['case_joint'], chosen[task]['pca_query']
        old = p['reference_means'][task]
        positive = sum(candidate['by_seed'][str(seed)]['auc'] + candidate['by_seed'][str(seed)]['ap'] >
                       baseline['by_seed'][str(seed)]['auc'] + baseline['by_seed'][str(seed)]['ap'] for seed in seeds)
        old_positive = sum(candidate['by_seed'][str(seed)]['auc'] + candidate['by_seed'][str(seed)]['ap'] >
                           old['by_seed'][str(seed)]['auc'] + old['by_seed'][str(seed)]['ap'] for seed in seeds)
        threshold = 1 if p['smoke'] else 2
        gates[task] = dict(delta_auc=candidate['mean_auc'] - baseline['mean_auc'],
                           delta_ap=candidate['mean_ap'] - baseline['mean_ap'],
                           delta_old_auc=candidate['mean_auc'] - old['auc'],
                           delta_old_ap=candidate['mean_ap'] - old['ap'], positive_seeds=positive,
                           positive_old_seeds=old_positive,
                           passed=candidate['mean_auc'] > baseline['mean_auc'] and candidate['mean_ap'] > baseline['mean_ap']
                           and candidate['mean_auc'] > old['auc'] and candidate['mean_ap'] > old['ap']
                           and positive >= threshold and old_positive >= threshold)
    # Smoke must execute both controls even when three-step metrics are poor.
    tasks = list(p['tasks']) if p['smoke'] else [t for t in p['tasks'] if gates[t]['passed']]
    freeze_decision(out, 'selection.json', dict(protocol_sha256=p['sha256'],
                    finalists_sha256=candidates['sha256'], chosen=chosen, gates=gates, control_tasks=tasks))
    print(json.dumps(dict(chosen=chosen, gates=gates, control_tasks=tasks), indent=2), flush=True)


def report(args):
    out = Path(args.out_dir)
    p = load_protocol(out)
    selection = read_decision(out, 'selection.json', p)
    all_jobs = list(jobs(out, p, 'screen')) + list(jobs(out, p, 'confirm')) + list(jobs(out, p, 'controls'))
    rows, interventions, replay, initial = [], [], [], {}
    cache = {}
    for task, arm, cfg, seed in all_jobs:
        if task not in cache:
            cache[task] = parent.load_stage(out, p, task, 'inner')
        arrays, graph = cache[task]
        dest = directory(out, task, arm, cfg['id'], seed)
        r = parent.checked_result(dest, p)
        trace = pd.read_csv(dest / 'trace.csv')
        if len(trace) != p['training'][task]['train_epochs'] or choose_epoch([trace], p['smoothing_window'])['epochs'] != r['selected_epoch']:
            raise AssertionError('Epoch budget/selection mismatch')
        with np.load(dest / 'scores.npz') as saved:
            np.testing.assert_array_equal(saved['y_val'], arrays['y_val'])
            np.testing.assert_array_equal(saved['pairs_val'], arrays['pairs_val'])
            scores = {k: saved[k].copy() for k in saved.files if k not in ('y_val', 'pairs_val')}
        auc, ap = score(arrays['y_val'], scores['val'])
        np.testing.assert_allclose([auc, ap], [r['inner_auc'], r['inner_aupr']], atol=1e-12, rtol=0)
        if (task, seed) in initial and initial[task, seed] != r['initial_base_sha256']:
            raise AssertionError('Unmatched base initialization')
        initial[task, seed] = r['initial_base_sha256']
        model, data, pairs, weights = build(arrays, graph, task, arm, seed, cfg, p, args.device)
        if state_digest(model) != r['initial_state_sha256']:
            raise AssertionError('Initialization not reproducible')
        model.load_state_dict(torch.load(dest / 'checkpoint.pt', map_location='cpu'))
        labels = torch.from_numpy(arrays['y_train']).to(args.device)
        actual = predict(model, data, pairs, weights, labels, arrays, True)
        if set(actual) != set(scores):
            raise AssertionError('Prediction keys differ')
        errors = {}
        for name in actual:
            np.testing.assert_allclose(actual[name], scores[name], atol=5e-5, rtol=5e-5)
            errors[name] = float(np.max(np.abs(actual[name] - scores[name])))
        macro, eligible = previous.macro_ap(arrays['y_val'], scores['val'], arrays['pairs_val'])
        selected = arm in CONTROLS or cfg['id'] == selection['chosen'][task][arm]['config']['id']
        r.update(macro_drug_ap=macro, eligible_drugs=eligible, selected_config=selected)
        rows.append(r)
        if arm == 'case_joint' and selected:
            for name in ('prior', 'shuffled_labels', 'drug_only', 'uniform_selected'):
                a, b = score(arrays['y_val'], scores[name])
                m, _ = previous.macro_ap(arrays['y_val'], scores[name], arrays['pairs_val'])
                interventions.append(dict(task=task, seed=seed, intervention=name,
                                           auc=a, ap=b, delta_auc=a-auc, delta_ap=b-ap,
                                           macro_drug_ap=m, delta_macro_drug_ap=m-macro))
        replay.append(dict(task=task, arm=arm, config_id=cfg['id'], seed=seed, max_abs=errors))
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'all_results.csv', index=False)
    chosen = frame[frame.selected_config]
    chosen.to_csv(out / 'selected_results.csv', index=False)
    pd.DataFrame(interventions).to_csv(out / 'interventions.csv', index=False)
    summary = chosen.groupby(['task', 'arm'])[['inner_auc', 'inner_aupr', 'macro_drug_ap']].agg(['mean', 'std'])
    summary.to_csv(out / 'summary.csv')
    comparisons = []
    for task in p['tasks']:
        block = chosen[chosen.task == task]
        candidate = block[block.arm == 'case_joint'].set_index('seed').sort_index()
        for control in ['pca_query'] + ([*CONTROLS] if task in selection['control_tasks'] else []):
            baseline = block[block.arm == control].set_index('seed').sort_index()
            da = candidate.inner_auc.to_numpy() - baseline.inner_auc.to_numpy()
            dp = candidate.inner_aupr.to_numpy() - baseline.inner_aupr.to_numpy()
            dm = candidate.macro_drug_ap.to_numpy() - baseline.macro_drug_ap.to_numpy()
            comparisons.append(dict(task=task, control=control, delta_auc=float(da.mean()), delta_ap=float(dp.mean()),
                                    delta_macro_ap=float(dm.mean()), positive_auc_seeds=int((da > 0).sum()),
                                    positive_ap_seeds=int((dp > 0).sum()), positive_criterion_seeds=int((da+dp > 0).sum())))
    pd.DataFrame(comparisons).to_csv(out / 'contrasts.csv', index=False)
    mechanism = {}
    for task in p['tasks']:
        cs = [c for c in comparisons if c['task'] == task and c['control'] in CONTROLS]
        mechanism[task] = None if not cs else all(c['delta_auc'] > 0 and c['delta_ap'] > 0 and
                                                c['positive_criterion_seeds'] >= (1 if p['smoke'] else 2) for c in cs)
    validation = dict(passed=True, smoke=p['smoke'], runs=len(rows), checkpoint_replays=len(replay),
                      protocol_sha256=p['sha256'], selection_sha256=selection['sha256'],
                      no_outer_evaluation=True, no_test_evaluation=True,
                      performance_by_task=selection['gates'], mechanism_by_task=mechanism,
                      overall_performance=all(v['passed'] for v in selection['gates'].values()),
                      overall_mechanism=all(v is True for v in mechanism.values()), replay=replay)
    write_json(out / 'VALIDATION.json', validation)
    print(summary.to_string(), flush=True)
    print('PERFORMANCE', validation['overall_performance'], 'MECHANISM', validation['overall_mechanism'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'run', 'finalists', 'select', 'report'):
        cmd = sub.add_parser(name)
        cmd.add_argument('--out-dir', required=True)
        if name == 'prepare':
            cmd.add_argument('--reference-dir', default='improve/results/support_operator_20260910')
            cmd.add_argument('--smoke', action='store_true')
        elif name in ('run', 'report'):
            cmd.add_argument('--device', default='cuda:0')
        if name == 'run':
            cmd.add_argument('--task', choices=previous.TASKS)
            cmd.add_argument('--stage', choices=('screen', 'confirm', 'controls'), required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    globals()[args.command](args)


if __name__ == '__main__':
    main()
