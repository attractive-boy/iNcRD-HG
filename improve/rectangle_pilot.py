"""Bounded, validation-only pilot of learned bipartite rectangle updates.

Two recipes for each of three arms, then two additional initializations of
the selected recipe: 24 full runs. Never opens outer or test arrays. Historical
training sources and results are kept intact.
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
from improve.network_feature_ablation import score
from improve.pair_exchange_modules import remove_target_pairs, sample_queries
from improve.rectangle_modules import RectangleModel


ARMS = ('pca_query', 'additive', 'rectangle')
OWN_FILES = ('rectangle_pilot.py', 'rectangle_modules.py', 'test_rectangle.py')


def sources():
    return dict(previous.sources(), **{n: sha256(Path(__file__).with_name(n)) for n in OWN_FILES})


def prepare(args):
    out, reference = Path(args.out_dir).resolve(), Path(args.reference_dir).resolve()
    if (out / 'protocol.json').exists() or (out / 'runs').exists():
        raise ValueError('Preserve the frozen experiment; use a new directory')
    ref = previous.load_protocol(reference)
    checked = json.loads((reference / 'VALIDATION.json').read_text())
    if not checked['passed'] or checked['smoke']:
        raise ValueError('A verified full reference is required')
    configs = [dict(id='c0', lr_multiplier=.3, branch_lr_multiplier=3., weight_decay=0.),
               dict(id='c1', lr_multiplier=1., branch_lr_multiplier=1., weight_decay=0.)]
    if args.smoke:
        configs = [dict(id='smoke', lr_multiplier=1., branch_lr_multiplier=1., weight_decay=0.)]
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), smoke=bool(args.smoke),
        reference_dir=str(reference), reference_protocol_sha256=ref['sha256'],
        reference_results_sha256=sha256(reference / 'results.csv'),
        tasks=ref['tasks'], arms=list(ARMS), primary='rectangle', configurations=configs,
        screening_seed=666, confirmation_seeds=[] if args.smoke else [0, 1],
        training={t: dict(c, train_epochs=3 if args.smoke else c['train_epochs']) for t, c in ref['training'].items()},
        width=16, depth=2, max_correction=1., query_fraction=.5, ramp_fraction=.2,
        smoothing_window=2 if args.smoke else 5,
        selection='Screen two recipes per arm with seed 666; freeze one per arm by first peak trailing-five inner mean(AUC,AP), ties by config id. Confirm the selected recipe with seeds 0 and 1. Each run selects its own checkpoint by the same rule. No subsequent recipe selection.',
        performance_screen='Positive mean AUC AND AP, and at least 2/3 positive paired mean-metric deltas against this round original AND the previous original reference; report each task separately and require both for a joint claim.',
        mechanism_screen='Additionally beat the additive operator in mean AUC AND AP with at least 2/3 positive paired mean-metric deltas. Macro-drug AP must be reported; no biological mechanism claim follows from a ranking gain.',
        definitions=dict(pca_query='Original PCA GNN and concat head; identical query masks and recipe budget',
                         additive='Same latent pair field, evidence, parameters and depth; additive three-position message',
                         rectangle='Product of three latent pair factors completing a bipartite rectangle; repeated endpoints excluded'),
        graph_policy='Stage graph restricted to fitting positives; each query positive removed both ways. Auxiliary/other-task relations follow the existing single-task protocol.',
        evidence_policy='The branch distinguishes known positive, sampled fitting zero, and unknown. Entire current query batch is excluded, including zeros. Other RNA task contributes existing graph positives only.',
        latent_policy='Dense lncRNA-drug and miRNA-drug pair features are computed from node encodings. They are not independent pair ID parameters or hard pseudo-labels. CircRNA remains in the original GNN only.',
        operator_policy='Each elementary message averages over s!=r,e!=d; later layers may depend indirectly on earlier query states. No claim of total query-state independence or exact 2-WL expressivity.',
        optimizer='RMSprop alpha=.99; ordinary unweighted BCE; original prior preserved at initialization; correction ramps during first 20% of training and is tanh-bounded by one logit.',
        no_outer_evaluation=True, no_test_evaluation=True,
        limitations=['The internal holdout has already been reused for model development.',
                     'Three initializations are not three independent data splits.',
                     'One screening seed chooses recipes; this is a bounded pilot, not an exhaustive search.',
                     'Allocated parameter counts and initial values match for additive/product; functional capacity is different.',
                     'A sampled zero denotes a benchmark label, not verified sensitivity.',
                     'Matrix completion, higher-order link encoders and multiplicative pair updates have prior art.'],
        reference_means={}, source_sha256=sources(), files={}, audit=ref['audit'])
    old = pd.read_csv(reference / 'results.csv')
    for task in p['tasks']:
        block = old[(old.task == task) & (old.arm == 'pca_query')]
        p['reference_means'][task] = dict(auc=float(block.inner_auc.mean()), ap=float(block.inner_aupr.mean()),
            by_seed={str(int(r.seed)): dict(auc=r.inner_auc, ap=r.inner_aupr) for r in block.itertuples()})
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
    print('FROZEN', out, '6 smoke runs' if args.smoke else '12 screening + 12 confirmation', flush=True)


def load_protocol(out):
    out = Path(out)
    p = json.loads((out / 'protocol.json').read_text())
    if p['sha256'] != json_digest({k: v for k, v in p.items() if k != 'sha256'}):
        raise ValueError('Protocol changed')
    if p['source_sha256'] != sources():
        raise ValueError('Frozen source changed')
    for name, digest in p['files'].items():
        if sha256(out / name) != digest:
            raise ValueError('Input changed: ' + name)
    return p


def directory(out, task, arm, config, seed):
    return Path(out) / 'runs' / task / arm / config / ('s' + str(seed))


def selection(out, p):
    value = json.loads((Path(out) / 'selection.json').read_text())
    if value['sha256'] != json_digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('Selection changed')
    if value['protocol_sha256'] != p['sha256']:
        raise ValueError('Selection belongs to another protocol')
    for name, digest in value['screen_results_sha256'].items():
        if sha256(Path(out) / name) != digest:
            raise ValueError('Screen changed after selection')
    return value


def jobs(out, p, stage):
    if stage == 'screen':
        for task in p['tasks']:
            for arm in ARMS:
                for cfg in p['configurations']:
                    yield task, arm, cfg, p['screening_seed']
    elif stage == 'confirm':
        chosen = selection(out, p)['chosen']
        for task in p['tasks']:
            for arm in ARMS:
                for seed in p['confirmation_seeds']:
                    yield task, arm, chosen[task][arm], seed
    else:
        raise ValueError(stage)


def build(arrays, graph, task, arm, seed, p, device):
    base, data, pairs, weights = parent.make_model(arrays, graph, task, 'pca', seed, device)
    torch.manual_seed(seed + 804721)
    cfg = p['training'][task]
    model = RectangleModel(base, cfg['channels'], cfg['rna'], mode=arm, width=p['width'],
        depth=p['depth'], max_correction=p['max_correction']).to(device)
    return model, data, pairs, weights


@torch.no_grad()
def predict(model, data, pairs, weights, labels, interventions=False):
    model.eval()
    full, prior, correction = model(data.x_dict, data.edge_index_dict, pairs['val'], weights,
        pairs['train'], labels, return_components=True)
    out = dict(val=full.cpu().numpy(), prior=prior.cpu().numpy(), correction=correction.cpu().numpy())
    if interventions and model.mode == 'rectangle':
        for name in ('no_evidence', 'no_message', 'additive'):
            out[name] = model(data.x_dict, data.edge_index_dict, pairs['val'], weights,
                pairs['train'], labels, intervention=name).cpu().numpy()
    return out


def train(arrays, graph, task, arm, seed, cfg, p, device):
    started = time.monotonic()
    model, data, pairs, weights = build(arrays, graph, task, arm, seed, p, device)
    initial, base_initial = state_digest(model), state_digest(model.base)
    lr = p['training'][task]['lr'] * cfg['lr_multiplier']
    groups = [dict(params=list(model.base.parameters()), lr=lr)]
    if model.branch is not None:
        groups.append(dict(params=list(model.branch.parameters()), lr=lr * cfg['branch_lr_multiplier']))
    optimizer = torch.optim.RMSprop(groups, lr=lr, alpha=.99, weight_decay=cfg['weight_decay'])
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
        logits = model(data.x_dict, edges, pairs['train'][:, query], step_weights,
                       pairs['train'], labels, ramp=ramp)
        loss = F.binary_cross_entropy_with_logits(logits, labels[query])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Non-finite loss')
        loss.backward()
        active.update(n for n, v in model.named_parameters() if v.grad is not None)
        optimizer.step()
        outputs = predict(model, data, pairs, weights, labels)
        auc, ap = score(arrays['y_val'], outputs['val'])
        trace.append(dict(epoch=epoch, loss=float(loss.detach()), queries=len(query), ramp=ramp,
                          inner_auc=auc, inner_aupr=ap, criterion=(auc + ap) / 2,
                          correction_mean_abs=float(np.mean(np.abs(outputs['correction'])))))
        if epoch >= p['smoothing_window']:
            smooth = float(np.mean([r['criterion'] for r in trace[-p['smoothing_window']:]]))
            if smooth > best:
                best, selected, saved = smooth, epoch, outputs['val'].copy()
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch % 50 == 0:
            print('EPOCH', task, arm, cfg['id'], seed, epoch, round((auc + ap) / 2, 6),
                  'seconds', round(time.monotonic() - started, 1), flush=True)
    model.load_state_dict(state)
    outputs = predict(model, data, pairs, weights, labels, True)
    np.testing.assert_allclose(outputs['val'], saved, atol=5e-5, rtol=5e-5)
    restore_error = float(np.max(np.abs(outputs['val'] - saved)))
    outputs['val'] = saved
    auc, ap = score(arrays['y_val'], saved)
    result = dict(task=task, arm=arm, seed=seed, config=cfg, config_id=cfg['id'],
        inner_auc=auc, inner_aupr=ap, criterion=(auc + ap) / 2, selected_epoch=selected,
        selection_score=best, trained_epochs=epochs, initial_state_sha256=initial,
        initial_base_sha256=base_initial, params=sum(v.numel() for v in model.parameters()),
        parameters_with_gradient=sum(v.numel() for n, v in model.named_parameters() if n in active),
        correction_mean_abs=float(np.mean(np.abs(outputs['correction']))),
        restore_max_abs=restore_error, elapsed_seconds=time.monotonic() - started,
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
                r, outputs, trace, state = train(arrays, graph, task, arm, seed, cfg, p, args.device)
                np.savez_compressed(dest / 'scores.npz', **outputs, y_val=arrays['y_val'], pairs_val=arrays['pairs_val'])
                pd.DataFrame(trace).to_csv(dest / 'trace.csv', index=False)
                torch.save(state, dest / 'checkpoint.pt')
                r['artifacts_sha256'] = {n: sha256(dest / n) for n in ('scores.npz', 'trace.csv', 'checkpoint.pt')}
                write_json(dest / 'result.json', r)
                print('DONE', task, arm, cfg['id'], seed, '%.6f/%.6f epoch %d %.1fs' % (
                    r['inner_auc'], r['inner_aupr'], r['selected_epoch'], r['elapsed_seconds']), flush=True)


def select(args):
    out = Path(args.out_dir)
    p = load_protocol(out)
    if (out / 'selection.json').exists():
        print(json.dumps(selection(out, p)['chosen'], indent=2), flush=True)
        return
    chosen, hashes = {}, {}
    for task in p['tasks']:
        chosen[task] = {}
        for arm in ARMS:
            records = []
            for cfg in p['configurations']:
                dest = directory(out, task, arm, cfg['id'], p['screening_seed'])
                r = parent.checked_result(dest, p)
                expected = choose_epoch([pd.read_csv(dest / 'trace.csv')], p['smoothing_window'])
                if expected['epochs'] != r['selected_epoch']:
                    raise ValueError('Invalid screen checkpoint')
                np.testing.assert_allclose(expected['selection_score'], r['selection_score'], atol=1e-12, rtol=0)
                records.append(r)
                hashes[str((dest / 'result.json').relative_to(out))] = sha256(dest / 'result.json')
            records.sort(key=lambda r: (-r['selection_score'], r['config_id']))
            chosen[task][arm] = records[0]['config']
    value = dict(protocol_sha256=p['sha256'], chosen=chosen, screen_results_sha256=hashes)
    value['sha256'] = json_digest(value)
    write_json(out / 'selection.json', value)
    print(json.dumps(chosen, indent=2), flush=True)


def report(args):
    out = Path(args.out_dir)
    p = load_protocol(out)
    chosen = selection(out, p)
    rows, replay, interventions, initial, branch_initial, cache = [], [], [], {}, {}, {}
    for task, arm, cfg, seed in list(jobs(out, p, 'screen')) + list(jobs(out, p, 'confirm')):
        if task not in cache:
            cache[task] = parent.load_stage(out, p, task, 'inner')
        arrays, graph = cache[task]
        dest = directory(out, task, arm, cfg['id'], seed)
        r = parent.checked_result(dest, p)
        trace = pd.read_csv(dest / 'trace.csv')
        epoch_check = choose_epoch([trace], p['smoothing_window'])
        if len(trace) != p['training'][task]['train_epochs'] or epoch_check['epochs'] != r['selected_epoch']:
            raise AssertionError('Epoch budget/selection mismatch')
        np.testing.assert_allclose(epoch_check['selection_score'], r['selection_score'], atol=1e-12, rtol=0)
        with np.load(dest / 'scores.npz') as saved:
            np.testing.assert_array_equal(saved['y_val'], arrays['y_val'])
            np.testing.assert_array_equal(saved['pairs_val'], arrays['pairs_val'])
            scores = {k: saved[k].copy() for k in saved.files if k not in ('y_val', 'pairs_val')}
        auc, ap = score(arrays['y_val'], scores['val'])
        np.testing.assert_allclose([auc, ap], [r['inner_auc'], r['inner_aupr']], atol=1e-12, rtol=0)
        if (task, seed) in initial and initial[task, seed] != r['initial_base_sha256']:
            raise AssertionError('Unmatched base initialization')
        initial[task, seed] = r['initial_base_sha256']
        if arm != 'pca_query':
            if (task, seed) in branch_initial and branch_initial[task, seed] != r['initial_state_sha256']:
                raise AssertionError('Unmatched branch initialization')
            branch_initial[task, seed] = r['initial_state_sha256']
        model, data, pairs, weights = build(arrays, graph, task, arm, seed, p, args.device)
        if state_digest(model) != r['initial_state_sha256']:
            raise AssertionError('Initialization not reproducible')
        model.load_state_dict(torch.load(dest / 'checkpoint.pt', map_location='cpu'))
        actual = predict(model, data, pairs, weights, torch.from_numpy(arrays['y_train']).to(args.device), True)
        if set(actual) != set(scores):
            raise AssertionError('Prediction keys differ')
        errors = {}
        for name, values in actual.items():
            np.testing.assert_allclose(values, scores[name], atol=5e-5, rtol=5e-5)
            errors[name] = float(np.max(np.abs(values - scores[name])))
        macro, eligible = previous.macro_ap(arrays['y_val'], scores['val'], arrays['pairs_val'])
        selected = cfg['id'] == chosen['chosen'][task][arm]['id']
        r.update(macro_drug_ap=macro, eligible_drugs=eligible, selected_config=selected)
        rows.append(r)
        if arm == 'rectangle' and selected:
            for name in ('prior', 'no_evidence', 'no_message', 'additive'):
                a, b = score(arrays['y_val'], scores[name])
                m, _ = previous.macro_ap(arrays['y_val'], scores[name], arrays['pairs_val'])
                interventions.append(dict(task=task, seed=seed, intervention=name, auc=a, ap=b,
                    delta_auc=a - auc, delta_ap=b - ap, macro_drug_ap=m, delta_macro_drug_ap=m - macro))
        replay.append(dict(task=task, arm=arm, config_id=cfg['id'], seed=seed, max_abs=errors))
        print('REPLAY', task, arm, cfg['id'], seed, max(errors.values()), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / 'all_results.csv', index=False)
    selected = frame[frame.selected_config]
    selected.to_csv(out / 'selected_results.csv', index=False)
    pd.DataFrame(interventions).to_csv(out / 'interventions.csv', index=False)
    summary = selected.groupby(['task', 'arm'])[['inner_auc', 'inner_aupr', 'macro_drug_ap']].agg(['mean', 'std'])
    summary.to_csv(out / 'summary.csv')
    contrasts, gates = [], {}
    threshold = 1 if p['smoke'] else 2
    for task in p['tasks']:
        c = selected[(selected.task == task) & (selected.arm == 'rectangle')].set_index('seed').sort_index()
        task_contrasts = []
        for control in ('pca_query', 'additive', 'previous_original'):
            if control == 'previous_original':
                old = p['reference_means'][task]['by_seed']
                av = np.asarray([old[str(s)]['auc'] for s in c.index])
                pv = np.asarray([old[str(s)]['ap'] for s in c.index])
                dm = None
            else:
                b = selected[(selected.task == task) & (selected.arm == control)].set_index('seed').loc[c.index]
                av, pv = b.inner_auc.to_numpy(), b.inner_aupr.to_numpy()
                dm = float((c.macro_drug_ap - b.macro_drug_ap).mean())
            da, dp = c.inner_auc.to_numpy() - av, c.inner_aupr.to_numpy() - pv
            contrast = dict(task=task, control=control, delta_auc=float(da.mean()), delta_ap=float(dp.mean()),
                delta_macro_drug_ap=dm, positive_auc_seeds=int((da > 0).sum()),
                positive_ap_seeds=int((dp > 0).sum()), positive_criterion_seeds=int((da + dp > 0).sum()),
                passed=bool(da.mean() > 0 and dp.mean() > 0 and (da + dp > 0).sum() >= threshold))
            contrasts.append(contrast)
            task_contrasts.append(contrast)
        gates[task] = dict(performance=all(v['passed'] for v in task_contrasts if v['control'] != 'additive'),
                           mechanism=all(v['passed'] for v in task_contrasts))
    pd.DataFrame(contrasts).to_csv(out / 'contrasts.csv', index=False)
    validation = dict(passed=True, smoke=p['smoke'], runs=len(rows), checkpoint_replays=len(replay),
        protocol_sha256=p['sha256'], selection_sha256=chosen['sha256'], gates=gates,
        no_outer_evaluation=True, no_test_evaluation=True, replay=replay)
    write_json(out / 'VALIDATION.json', validation)
    print(summary.to_string(), flush=True)
    print('GATES', json.dumps(gates), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'run', 'select', 'report'):
        cmd = commands.add_parser(name)
        cmd.add_argument('--out-dir', required=True)
        if name == 'prepare':
            cmd.add_argument('--reference-dir', default='improve/results/support_operator_20260910')
            cmd.add_argument('--smoke', action='store_true')
        if name in ('run', 'report'):
            cmd.add_argument('--device', default='cuda:0')
        if name == 'run':
            cmd.add_argument('--task', choices=previous.TASKS)
            cmd.add_argument('--stage', choices=('screen', 'confirm'), required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    globals()[args.command](args)


if __name__ == '__main__':
    main()
