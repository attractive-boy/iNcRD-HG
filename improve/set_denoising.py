"""Frozen validation of a drug-conditioned, multi-positive set denoiser.

prepare -> features inner -> run screen -> select screen -> run confirm ->
select confirm -> features outer -> run outer -> report -> validate.
Two tasks can run independently on separate GPUs. No test arrays are loaded.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import pickle
import shutil
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.nn import functional as F

from improve import backbone_replacement as parent
from improve.graph_tabm_modules import PairFeatureEncoder, GraphTabM, crossfit_features, fit_transform
from improve.set_denoising_modules import (
    ARMS, SetDenoiser, choose_checkpoint, corrupt_positive_set, drug_balanced_loss,
    fold_assignments, full_candidate_recall, ranking_metrics, support_and_queries,
)


OWN = ('set_denoising.py', 'set_denoising_modules.py', 'test_set_denoising.py')


def sources():
    names = list(dict.fromkeys(list(parent.SOURCE_FILES) + ['graph_tabm_modules.py'] + list(OWN)))
    return {name: parent.digest(Path(__file__).with_name(name)) for name in names}


def prepare(args):
    out, ref, tab = Path(args.out_dir).resolve(), Path(args.reference_dir).resolve(), Path(args.tabm_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new output directory')
    reference = parent.protocol(ref)
    teacher_recipes = json.loads((tab / 'REFIT.json').read_text())
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), smoke=args.smoke,
        reference_dir=str(ref), reference_protocol_sha256=reference['sha256'], tabm_dir=str(tab),
        teacher_recipe_sha256=parent.digest(tab / 'REFIT.json'),
        tasks=list(parent.TASKS), arms=list(ARMS), folds=2 if args.smoke else 5,
        fold_seed=20260910, inner_split_seed=2026091101, inner_fraction=.2,
        teacher_seed=666, teacher_batch_size=1024, prediction_batch_size=8192,
        teacher={t: teacher_recipes['chosen'][t]['tabm'] for t in parent.TASKS},
        screening_seed=666, confirmation_seeds=[0] if args.smoke else [0, 1],
        refit_seeds=[666] if args.smoke else [666, 0, 1],
        configurations=[dict(id='c'+str(i), lr=lr, weight_decay=1e-4, width=64, depth=2,
                            slots=8, dropout=.1, residual_cap=2., residual_penalty=.0025)
                        for i, lr in enumerate([1e-3] if args.smoke else [3e-4, 1e-3])],
        steps={'LncDrug': 12 if args.smoke else 400, 'MiDrug': 12 if args.smoke else 600},
        batch_drugs=8 if args.smoke else 16, evaluation_interval=4 if args.smoke else 20,
        smoothing_window=2 if args.smoke else 3,
        mask_times=[.25, .5, .75, 1.], mask_probabilities=[1/6, 1/6, 1/6, .5],
        source_sha256=sources(), files={},
        selection='Same two learning rates per arm on seed 666. Earliest best trailing-three checkpoint score: .5 macro-drug AP + .25 (global AP + AUC). Step zero competes. Freeze learning rates, confirm seeds 0/1, use median selected steps for all three outer refits.',
        mask_policy='Teacher and structural features exclude the entire query fold. Only query positives may be revealed as denoising context; these have no loss. Masked positives receive 1/t weights. Sampled zero labels never become known negative tokens. Evaluation is terminal t=1 with fitting positives only.',
        teacher_policy='A common fixed-seed TabM prior is shared by all four arms and all reranker seeds. Each training fold has a separate teacher fitted without that entire fold; teacher training features themselves use nested cross-fitting. Full teacher uses fitting rows only. Teacher settings are inherited and never tuned here.',
        graph_policy='Original PCA/drug/circRNA input and other-task protocol. Every drug retains every RNA candidate. Structural features and prior logits at known support positives are neutral zeros; positive flags and endpoint attributes represent those context tokens.',
        objective='Per-drug mean empirical BCE plus bounded-logit residual penalty, identical in all arms; denoising uses inverse mask-probability weights. This is not a PU risk estimator or a claim to recover biological true negatives.',
        screen_rules=dict(macro_lnc_min=.005, auc_noninferiority=.001, ap_noninferiority=.002,
                          macro_mi_noninferiority=.005, mechanism_positive_seeds=2),
        no_test_evaluation=True,
        limitations=['Outer validation has been repeatedly used historically; the new inner split is development data, not independent external evidence.',
            'Reranker seed variance is conditional on a single fixed teacher family.',
            'Sampled zero labels are unobserved associations and remain an empirical benchmark convention.',
            'Current graph contains other-task and circRNA relations; this is not strict cold-drug evaluation.',
            'The prototype is positive-set masking with one terminal forward pass, not an official RADD/KREPE reproduction or iterative generation.'])
    if args.smoke:
        for t in p['tasks']:
            p['teacher'][t] = dict(p['teacher'][t], epochs=3)
    out.mkdir(parents=True)
    for task in p['tasks']:
        dest = out / 'inputs' / task
        dest.mkdir(parents=True)
        for name in ('outer.npz', 'train_graph.pth'):
            src = ref / 'inputs' / task / name
            shutil.copy2(src, dest / name)
            p['files'][str((dest / name).relative_to(out))] = parent.digest(dest / name)
        # Read fitting rows/features only. Historical outer labels are not
        # opened while constructing or selecting the inner experiment.
        with np.load(dest / 'outer.npz') as archive:
            arrays = {k: archive[k].copy() for k in archive.files if k not in ('pairs_val', 'y_val')}
        pairs, labels = arrays.pop('pairs_train'), arrays.pop('y_train')
        fit, hold = train_test_split(np.arange(len(labels)), test_size=p['inner_fraction'],
            random_state=p['inner_split_seed'], stratify=labels)
        fit, hold = np.sort(fit), np.sort(hold)
        arrays.update(pairs_train=pairs[:, fit], y_train=labels[fit], pairs_val=pairs[:, hold], y_val=labels[hold])
        np.savez_compressed(dest / 'inner.npz', **arrays)
        np.savez_compressed(dest / 'inner_indices.npz', fit=fit, hold=hold)
        for name in ('inner.npz', 'inner_indices.npz'):
            p['files'][str((dest / name).relative_to(out))] = parent.digest(dest / name)
    (out / 'source_snapshot').mkdir()
    for name in p['source_sha256']:
        shutil.copy2(Path(__file__).with_name(name), out / 'source_snapshot' / name)
    p['sha256'] = parent.json_digest(p)
    parent.write_json(out / 'protocol.json', p)
    print('FROZEN', out, p['sha256'], flush=True)


def protocol(out):
    out = Path(out)
    p = json.loads((out / 'protocol.json').read_text())
    if p['sha256'] != parent.json_digest({k: v for k, v in p.items() if k != 'sha256'}):
        raise ValueError('Protocol changed')
    if p['source_sha256'] != sources():
        raise ValueError('Frozen source changed; use a new experiment')
    for name, sha in p['files'].items():
        if parent.digest(out / name) != sha:
            raise ValueError('Input changed: ' + name)
    return p


def decision(out, name, p):
    value = json.loads((Path(out) / name).read_text())
    if value['protocol_sha256'] != p['sha256'] or value['sha256'] != parent.json_digest({k:v for k,v in value.items() if k != 'sha256'}):
        raise ValueError('Invalid frozen decision')
    for name, sha in value['results_sha256'].items():
        if parent.digest(Path(out) / name) != sha:
            raise ValueError('Decision input changed')
    return value


def make_teacher(dim, recipe, seed, device):
    parent.set_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = recipe['config']
    return GraphTabM(dim, width=cfg['width'], depth=cfg['depth'], dropout=cfg['dropout'], members=cfg['members']).to(device)


@torch.no_grad()
def teacher_predict(model, x, batch_size=8192):
    model.eval()
    result = []
    device = next(model.parameters()).device
    for i in range(0, len(x), batch_size):
        batch = torch.as_tensor(x[i:i+batch_size], device=device)
        result.append(torch.sigmoid(model(batch)).mean(1).cpu().numpy())
    return np.concatenate(result)


def train_teacher(x, y, recipe, seed, device, batch_size):
    started = time.monotonic()
    model = make_teacher(x.shape[1], recipe, seed, device)
    cfg = recipe['config']
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    xx, yy = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
    trace = []
    for epoch in range(1, recipe['epochs'] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total = 0.
        for i in range(0, len(x), batch_size):
            logits = model(xx[i:i+batch_size])
            target = yy[i:i+batch_size, None].expand_as(logits)
            loss = F.binary_cross_entropy_with_logits(logits, target, reduction='sum') / (len(x) * logits.shape[1])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Teacher loss')
            loss.backward()
            total += float(loss.detach())
        optimizer.step()
        trace.append(dict(epoch=epoch, loss=total))
    return model, trace, time.monotonic() - started


def checked_manifest(folder, p):
    m = json.loads((Path(folder) / 'manifest.json').read_text())
    if m['protocol_sha256'] != p['sha256']:
        raise ValueError('Cache belongs to a different protocol')
    for name, sha in m['artifacts_sha256'].items():
        if parent.digest(Path(folder) / name) != sha:
            raise ValueError('Cached artifact changed: ' + str(Path(folder) / name))
    return m


def build_features(out, p, task, stage, device):
    if stage not in ('inner', 'outer'):
        raise ValueError(stage)
    if stage == 'outer':
        decision(out, 'REFIT.json', p)
    dest = Path(out) / 'features' / stage / task
    if (dest / 'manifest.json').exists():
        return checked_manifest(dest, p)
    dest.mkdir(parents=True, exist_ok=True)
    a, graph, _ = parent.stage_inputs(out, task, stage)
    pairs, labels = a['pairs_train'], a['y_train']
    encoder = PairFeatureEncoder(graph, task)
    fold_ids = fold_assignments(pairs, labels, p['folds'], p['fold_seed'])
    for fold in list(range(p['folds'])) + [None]:
        name = 'full' if fold is None else 'f'+str(fold)
        folder = dest / name
        if (folder / 'manifest.json').exists():
            checked_manifest(folder, p)
            continue
        folder.mkdir(parents=True, exist_ok=True)
        support, query, target, fit = support_and_queries(pairs, labels, fold_ids, fold, encoder.nr, encoder.nd)
        dd, rr = np.nonzero(~support)
        grid_pairs = np.stack([rr, dd])
        fit_pairs, fit_y = pairs[:, fit], labels[fit]
        raw, raw_grid, names, nested = crossfit_features(graph, task, fit_pairs, fit_y, grid_pairs,
            folds=p['folds'], seed=p['fold_seed'])
        x, xgrid, scaler = fit_transform(raw, raw_grid)
        seed = p['teacher_seed'] if fold is None else p['teacher_seed'] + 17 * (fold+1)
        teacher, trace, elapsed = train_teacher(x, fit_y, p['teacher'][task], seed, device, p['teacher_batch_size'])
        probability = np.clip(teacher_predict(teacher, xgrid, p['prediction_batch_size']), 1e-6, 1-1e-6)
        grid_features = np.zeros((encoder.nd, encoder.nr, x.shape[1]), dtype=np.float32)
        base = np.zeros((encoder.nd, encoder.nr), dtype=np.float32)
        grid_features[dd, rr], base[dd, rr] = xgrid, np.log(probability) - np.log1p(-probability)
        np.save(folder / 'features.npy', grid_features)
        np.save(folder / 'base.npy', base)
        np.savez_compressed(folder / 'masks.npz', support=support, query=query, target=target, fit_indices=fit,
            query_indices=np.flatnonzero(fold_ids == fold) if fold is not None else np.array([], dtype=np.int64))
        np.savez_compressed(folder / 'teacher_fit.npz', x=x, y=fit_y, pairs=fit_pairs, nested_fold_ids=nested)
        with (folder / 'scaler.pkl').open('wb') as stream:
            pickle.dump(scaler, stream, protocol=4)
        torch.save({k:v.detach().cpu() for k,v in teacher.state_dict().items()}, folder / 'teacher.pt')
        pd.DataFrame(trace).to_csv(folder / 'teacher_trace.csv', index=False)
        files = ('features.npy', 'base.npy', 'masks.npz', 'teacher_fit.npz', 'scaler.pkl', 'teacher.pt', 'teacher_trace.csv')
        m = dict(protocol_sha256=p['sha256'], task=task, stage=stage, fold=fold, seed=seed,
            feature_names=names, nr=encoder.nr, nd=encoder.nd, teacher_recipe=p['teacher'][task],
            teacher_seconds=elapsed, unknown_pairs=len(grid_pairs.T), fit_rows=len(fit),
            target_positive_edges=int(support.sum()), artifacts_sha256={f:parent.digest(folder/f) for f in files})
        parent.write_json(folder / 'manifest.json', m)
        print('TEACHER', stage, task, name, 'rows', len(fit), 'seconds', round(elapsed, 1), flush=True)
        del teacher, grid_features, base, raw, raw_grid, x, xgrid
    np.savez_compressed(dest / 'data.npz', rna=encoder.unit(encoder.xr).astype(np.float32),
        drug=encoder.unit(encoder.xd).astype(np.float32), pairs_train=pairs, y_train=labels,
        pairs_val=a['pairs_val'], y_val=a['y_val'], fold_ids=fold_ids)
    files = ['data.npz'] + [name+'/manifest.json' for name in ['f'+str(i) for i in range(p['folds'])] + ['full']]
    m = dict(protocol_sha256=p['sha256'], task=task, stage=stage, folds=p['folds'], nr=encoder.nr, nd=encoder.nd,
             artifacts_sha256={f:parent.digest(dest/f) for f in files})
    parent.write_json(dest / 'manifest.json', m)
    print('FEATURES_READY', stage, task, flush=True)
    return m


class FeaturePack:
    def __init__(self, out, p, task, stage, device):
        self.folder = Path(out) / 'features' / stage / task
        self.manifest = checked_manifest(self.folder, p)
        self.manifest_sha256 = parent.digest(self.folder / 'manifest.json')
        self.device = device
        with np.load(self.folder / 'data.npz') as archive:
            self.data = {k:archive[k].copy() for k in archive.files}
        self.rna = torch.from_numpy(self.data['rna']).to(device)
        self.drug = torch.from_numpy(self.data['drug']).to(device)
        self.scenarios = []
        for name in ['f'+str(i) for i in range(p['folds'])] + ['full']:
            folder = self.folder / name
            checked_manifest(folder, p)
            with np.load(folder / 'masks.npz') as archive:
                masks = {k:archive[k].copy() for k in ('support', 'query', 'target')}
            scenario = dict(features=torch.from_numpy(np.load(folder/'features.npy')).to(device),
                            base=torch.from_numpy(np.load(folder/'base.npy')).to(device))
            scenario.update({k:torch.from_numpy(v).to(device) for k,v in masks.items()})
            scenario['eligible'] = np.flatnonzero(masks['query'].sum(1))
            self.scenarios.append(scenario)
        self.folds = p['folds']
        self.val_drugs = np.unique(self.data['pairs_val'][1])
        self.feature_dim = self.scenarios[-1]['features'].shape[-1]

    def baseline(self):
        grid = self.scenarios[-1]['base'].cpu().numpy()
        pairs = self.data['pairs_val']
        return grid[pairs[1], pairs[0]], grid


def make_model(pack, cfg, seed):
    parent.set_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return SetDenoiser(pack.feature_dim, pack.rna.shape[1], pack.drug.shape[1],
        **{k:cfg[k] for k in ('width', 'depth', 'slots', 'dropout', 'residual_cap')}).to(pack.device)


@torch.no_grad()
def predict(model, pack, arm, batch_drugs, intervention=None):
    model.eval()
    scenario = pack.scenarios[-1]
    grid = np.zeros(scenario['base'].shape, dtype=np.float32)
    for start in range(0, len(pack.val_drugs), batch_drugs):
        dd = torch.as_tensor(pack.val_drugs[start:start+batch_drugs], device=pack.device)
        support = scenario['support'][dd]
        if intervention == 'no_support':
            support = torch.zeros_like(support)
        elif intervention == 'shuffle_support':
            # Fixed permutation independent of labels; keeps each drug's count.
            support = support.roll(19, dims=1)
        logits = model(scenario['features'][dd], scenario['base'][dd], pack.rna, pack.drug[dd],
            support, torch.ones(len(dd), device=pack.device),
            interaction=ARMS[arm]['interaction'] and intervention != 'no_interaction')
        grid[dd.cpu().numpy()] = logits.cpu().numpy()
    pairs = pack.data['pairs_val']
    return grid[pairs[1], pairs[0]], grid


def evaluate(score, grid, pack):
    m, rows = ranking_metrics(pack.data['y_val'], score, pack.data['pairs_val'])
    m.update(full_candidate_recall(pack.data['y_val'], pack.data['pairs_val'], grid,
                                   pack.scenarios[-1]['support'].cpu().numpy()))
    return m, rows


def train(pack, p, task, arm, cfg, seed, stage, steps):
    started = time.monotonic()
    model = make_model(pack, cfg, seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    base_score, base_grid = pack.baseline()
    base_metrics, _ = evaluate(base_score, base_grid, pack) if stage != 'outer' else ({}, None)
    # No outer scores are computed during fixed-step refitting.
    trace = [dict(step=0, loss=None, **base_metrics)] if stage != 'outer' else [dict(step=0, loss=None)]
    saved_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    saved_score = base_score.copy() if stage != 'outer' else None
    selected, best = 0, base_metrics.get('criterion')
    rng = np.random.RandomState(seed + 193091)
    for step in range(1, steps + 1):
        model.train()
        scenario = pack.scenarios[(step - 1) % pack.folds]
        chosen = rng.choice(scenario['eligible'], size=min(p['batch_drugs'], len(scenario['eligible'])), replace=False)
        dd = torch.as_tensor(chosen, device=pack.device)
        sampled_time = rng.choice(p['mask_times'], size=len(chosen), p=p['mask_probabilities']).astype(np.float32)
        # Draw for EVERY arm, keeping drug sampling paired across the 2x2 design.
        uniform = torch.from_numpy(rng.random_sample((len(chosen), pack.rna.shape[0])).astype(np.float32)).to(pack.device)
        tt = torch.from_numpy(sampled_time if ARMS[arm]['denoising'] else np.ones_like(sampled_time)).to(pack.device)
        target, query, support = (scenario[k][dd] for k in ('target', 'query', 'support'))
        observed, weight, revealed = corrupt_positive_set(support, query, target, tt, uniform)
        logits = model(scenario['features'][dd], scenario['base'][dd], pack.rna, pack.drug[dd], observed, tt,
                       interaction=ARMS[arm]['interaction'])
        loss = drug_balanced_loss(logits, scenario['base'][dd], target, query, weight, cfg['residual_penalty'])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError('Nonfinite set loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if not all(bool(torch.isfinite(v.grad).all()) for v in model.parameters() if v.grad is not None):
            raise FloatingPointError('Nonfinite set gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step % p['evaluation_interval'] == 0 or step == steps:
            row = dict(step=step, loss=float(loss.detach()), revealed_positive=int(revealed.sum()),
                       seconds=time.monotonic()-started)
            if stage != 'outer':
                score, grid = predict(model, pack, arm, p['batch_drugs'])
                m, _ = evaluate(score, grid, pack)
                row.update(m)
            trace.append(row)
            if stage != 'outer':
                choice, value = choose_checkpoint(trace, p['smoothing_window'])
                if choice != selected:
                    if choice != step:
                        raise AssertionError('Unexpected checkpoint selection')
                    selected, best = choice, value
                    saved_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    saved_score = score.copy()
        if step == 1 or step % 100 == 0 or step == steps:
            print('STEP', stage, task, arm, cfg['id'], seed, step, 'loss', round(float(loss.detach()), 5),
                  'seconds', round(time.monotonic()-started, 1), flush=True)
    if stage == 'outer':
        selected = steps
        saved_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    else:
        check_step, check_value = choose_checkpoint(trace, p['smoothing_window'])
        if check_step != selected or abs(check_value-best) > 1e-12:
            raise AssertionError('Checkpoint selection drift')
    model.load_state_dict(saved_state)
    score, grid = predict(model, pack, arm, p['batch_drugs'])
    restore_error = 0.
    if saved_score is not None:
        restore_error = float(np.max(np.abs(score - saved_score)))
        np.testing.assert_allclose(score, saved_score, atol=2e-5, rtol=2e-5)
    m, per_drug = evaluate(score, grid, pack)
    interventions = {}
    if stage == 'outer' and arm == 'set_denoise':
        for name in ('no_support', 'shuffle_support', 'no_interaction'):
            s, g = predict(model, pack, arm, p['batch_drugs'], name)
            interventions[name] = dict(metrics=evaluate(s, g, pack)[0], scores=s)
    result = dict(task=task, arm=arm, config=cfg, config_id=cfg['id'], seed=seed, stage=stage,
        selected_step=selected, trained_steps=steps, selection_score=best if stage != 'outer' else None,
        params=sum(v.numel() for v in model.parameters()), elapsed_seconds=time.monotonic()-started,
        restore_max_abs=restore_error, protocol_sha256=p['sha256'],
        feature_manifest_sha256=pack.manifest_sha256, **m)
    return result, score, grid, per_drug, trace, saved_state, interventions


def directory(out, stage, task, arm, cfg, seed):
    return Path(out) / stage / task / arm / cfg / ('s'+str(seed))


def checked_result(folder, p):
    r = json.loads((Path(folder) / 'result.json').read_text())
    if r['protocol_sha256'] != p['sha256']:
        raise ValueError('Result protocol mismatch')
    for name, sha in r['artifacts_sha256'].items():
        if parent.digest(Path(folder) / name) != sha:
            raise ValueError('Result artifact changed')
    return r


def jobs(out, p, stage):
    chosen = decision(out, 'PARAMETERS.json', p)['chosen'] if stage != 'screen' else None
    refits = decision(out, 'REFIT.json', p)['chosen'] if stage == 'outer' else None
    for task in p['tasks']:
        for arm in p['arms']:
            configs = p['configurations'] if stage == 'screen' else [chosen[task][arm]]
            seeds = [p['screening_seed']] if stage == 'screen' else p['confirmation_seeds'] if stage == 'confirm' else p['refit_seeds']
            for cfg in configs:
                for seed in seeds:
                    steps = refits[task][arm]['steps'] if stage == 'outer' else p['steps'][task]
                    yield task, arm, cfg, seed, steps


def run(args):
    out, p = Path(args.out_dir), protocol(args.out_dir)
    planned = list(jobs(out, p, args.stage))
    for task in ([args.task] if args.task else p['tasks']):
        with (out / (task+'.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            pack = FeaturePack(out, p, task, 'outer' if args.stage == 'outer' else 'inner', args.device)
            for t, arm, cfg, seed, steps in planned:
                if t != task or (args.arm and args.arm != arm):
                    continue
                folder = directory(out, args.stage, task, arm, cfg['id'], seed)
                if (folder / 'result.json').exists():
                    checked_result(folder, p)
                    continue
                folder.mkdir(parents=True, exist_ok=True)
                r, score, grid, per_drug, trace, state, interventions = train(pack, p, task, arm, cfg, seed, args.stage, steps)
                np.savez_compressed(folder/'scores.npz', val=score, y_val=pack.data['y_val'], pairs_val=pack.data['pairs_val'],
                    full_grid=grid, **{k:v['scores'] for k,v in interventions.items()})
                pd.DataFrame(trace).to_csv(folder/'trace.csv', index=False)
                pd.DataFrame(per_drug).to_csv(folder/'per_drug.csv', index=False)
                torch.save(state, folder/'checkpoint.pt')
                r['interventions'] = {k:v['metrics'] for k,v in interventions.items()}
                r['artifacts_sha256'] = {n:parent.digest(folder/n) for n in ('scores.npz', 'trace.csv', 'per_drug.csv', 'checkpoint.pt')}
                parent.write_json(folder/'result.json', r)
                print('DONE', args.stage, task, arm, cfg['id'], seed, 'AP', round(r['ap'],6),
                      'macro', round(r['macro_drug_ap'],6), 'step', r['selected_step'], flush=True)
            del pack


def select(args):
    out, p = Path(args.out_dir), protocol(args.out_dir)
    if args.stage not in ('screen', 'confirm'):
        raise ValueError('Outer results cannot select settings')
    name = 'PARAMETERS.json' if args.stage == 'screen' else 'REFIT.json'
    if (out/name).exists():
        print(json.dumps(decision(out, name, p)['chosen'], indent=2), flush=True)
        return
    selected = decision(out, 'PARAMETERS.json', p)['chosen'] if args.stage == 'confirm' else None
    value = dict(protocol_sha256=p['sha256'], chosen={}, results_sha256={})
    for task in p['tasks']:
        value['chosen'][task] = {}
        for arm in p['arms']:
            if args.stage == 'screen':
                paths = [directory(out, 'screen', task, arm, c['id'], p['screening_seed']) for c in p['configurations']]
            else:
                cfg = selected[task][arm]
                paths = [directory(out, 'screen', task, arm, cfg['id'], p['screening_seed'])]
                paths += [directory(out, 'confirm', task, arm, cfg['id'], s) for s in p['confirmation_seeds']]
            rows = [checked_result(path, p) for path in paths]
            value['results_sha256'].update({str((path/'result.json').relative_to(out)):parent.digest(path/'result.json') for path in paths})
            if args.stage == 'screen':
                value['chosen'][task][arm] = sorted(rows, key=lambda r:(-r['selection_score'], r['config_id']))[0]['config']
            else:
                median = int(round(float(np.median([r['selected_step'] for r in rows]))))
                value['chosen'][task][arm] = dict(config=cfg, steps=median,
                    inner_macro_ap=float(np.mean([r['macro_drug_ap'] for r in rows])),
                    inner_ap=float(np.mean([r['ap'] for r in rows])),
                    nonzero_selected_seeds=sum(r['selected_step'] > 0 for r in rows))
    value['sha256'] = parent.json_digest(value)
    parent.write_json(out/name, value)
    print('SELECTED', name, json.dumps(value['chosen'], indent=2), flush=True)


def feature_command(args):
    out, p = Path(args.out_dir), protocol(args.out_dir)
    for task in ([args.task] if args.task else p['tasks']):
        with (out/(task+'.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            build_features(out, p, task, args.stage, args.device)


def baseline_result(out, p, task, stage):
    folder = Path(out) / 'features' / stage / task
    checked_manifest(folder, p)
    with np.load(folder / 'data.npz') as archive:
        y, pairs = archive['y_val'], archive['pairs_val']
    grid = np.load(folder / 'full' / 'base.npy')
    with np.load(folder / 'full' / 'masks.npz') as archive:
        support = archive['support']
    score = grid[pairs[1], pairs[0]]
    metrics, per_drug = ranking_metrics(y, score, pairs)
    metrics.update(full_candidate_recall(y, pairs, grid, support))
    return dict(task=task, arm='tabm_prior', stage=stage, seed=p['teacher_seed'], **metrics), per_drug, score, y, pairs


def report(args):
    out, p = Path(args.out_dir), protocol(args.out_dir)
    rows = []
    for stage in ('screen', 'confirm', 'outer'):
        for task, arm, cfg, seed, _ in jobs(out, p, stage):
            rows.append(checked_result(directory(out, stage, task, arm, cfg['id'], seed), p))
    frame = pd.DataFrame([{k:v for k,v in r.items() if k not in ('config', 'interventions', 'artifacts_sha256')} for r in rows])
    frame.to_csv(out / 'runs.csv', index=False)
    baseline_rows, baseline_drugs, contrasts, reference_rows = [], [], [], []
    metric_names = ('auc', 'ap', 'macro_drug_ap', 'macro_drug_auc', 'macro_drug_ndcg10',
                    'macro_drug_recall10', 'full_candidate_macro_recall20')
    outer = frame[frame.stage == 'outer']
    ref_p = parent.protocol(p['reference_dir'])
    ref_recipe = json.loads((Path(p['reference_dir']) / 'REFIT.json').read_text())['chosen']
    for task in p['tasks']:
        for stage in ('inner', 'outer'):
            base, per_drug, score, y, pairs = baseline_result(out, p, task, stage)
            baseline_rows.append(base)
            baseline_drugs.extend([dict(task=task, stage=stage, **r) for r in per_drug])
            if stage != 'outer':
                continue
            for _, row in outer[outer.task == task].iterrows():
                contrasts.append(dict(task=task, arm=row.arm, reference='tabm_prior', seed=int(row.seed),
                    **{name:float(row[name] - base[name]) for name in metric_names}))
            # These frozen GNN runs use exactly the same outer pairs/labels.
            # They are context references, not a newly tuned arm in this run.
            for seed in p['refit_seeds']:
                cfg = ref_recipe[task]['base']['config']['id']
                folder = parent.directory(p['reference_dir'], 'outer', task, 'base', cfg, seed)
                r = parent.checked_result(folder, ref_p)
                with np.load(folder / 'scores.npz') as archive:
                    np.testing.assert_array_equal(archive['y_val'], y)
                    np.testing.assert_array_equal(archive['pairs_val'], pairs)
                    vals = archive['val']
                m, _ = ranking_metrics(y, vals, pairs)
                reference_rows.append(dict(task=task, arm='base_gnn', stage='outer', seed=seed, **m,
                                           source=str(folder), source_result_sha256=parent.digest(folder/'result.json')))
    pd.DataFrame(baseline_rows).to_csv(out / 'baselines.csv', index=False)
    pd.DataFrame(baseline_drugs).to_csv(out / 'baseline_per_drug.csv', index=False)
    pd.DataFrame(reference_rows).to_csv(out / 'gnn_reference.csv', index=False)
    pd.DataFrame(contrasts).to_csv(out / 'paired_deltas.csv', index=False)
    comparison = pd.concat([outer, pd.DataFrame([r for r in baseline_rows if r['stage'] == 'outer']),
                            pd.DataFrame(reference_rows)], ignore_index=True)
    summary_rows = []
    for (task, arm), group in comparison.groupby(['task', 'arm'], sort=False):
        r = dict(task=task, arm=arm, runs=len(group))
        for name in metric_names:
            r[name] = float(group[name].mean())
            r[name+'_std'] = float(group[name].std(ddof=1)) if len(group) > 1 else 0.
        summary_rows.append(r)
    pd.DataFrame(summary_rows).to_csv(out / 'summary.csv', index=False)
    # Predeclared criteria separate practical performance from the two
    # distinct mechanism claims. No outer value changes a trained setting.
    screens = []
    deltas = pd.DataFrame(contrasts)
    for task in p['tasks']:
        main = deltas[(deltas.task == task) & (deltas.arm == 'set_denoise')]
        macro_min = p['screen_rules']['macro_lnc_min'] if task == 'LncDrug' else -p['screen_rules']['macro_mi_noninferiority']
        performance = bool(main.macro_drug_ap.mean() >= macro_min and
            main.auc.mean() >= -p['screen_rules']['auc_noninferiority'] and
            main.ap.mean() >= -p['screen_rules']['ap_noninferiority'])
        screens.append(dict(task=task, comparison='set_denoise_vs_tabm', passed=performance,
            delta_auc=float(main.auc.mean()), delta_ap=float(main.ap.mean()),
            delta_macro_ap=float(main.macro_drug_ap.mean()), positive_macro_seeds=int((main.macro_drug_ap > 0).sum())))
        for control in ('set_plain', 'point_denoise'):
            a = outer[(outer.task == task) & (outer.arm == 'set_denoise')].set_index('seed')
            b = outer[(outer.task == task) & (outer.arm == control)].set_index('seed')
            diff = a['macro_drug_ap'] - b['macro_drug_ap']
            passed = bool(diff.mean() > 0 and (diff > 0).sum() >= min(p['screen_rules']['mechanism_positive_seeds'], len(diff)))
            screens.append(dict(task=task, comparison='set_denoise_vs_'+control, passed=passed,
                delta_auc=float((a.auc-b.auc).mean()), delta_ap=float((a.ap-b.ap).mean()),
                delta_macro_ap=float(diff.mean()), positive_macro_seeds=int((diff > 0).sum())))
    pd.DataFrame(screens).to_csv(out / 'screens.csv', index=False)
    all_passed = all(r['passed'] for r in screens)
    text = ['# 多正例集合去噪：冻结实验报告', '',
        '**流程检查数据，不用于性能结论。**' if p['smoke'] else
        ('**主候选通过预设性能与机制筛选。**' if all_passed else '**主候选未通过全部预设性能与机制筛选。**'), '',
        '四组共享同一组折外 TabM 初始模型；表中精排标准差仅来自精排器初始化。TabM 行是一个固定初始模型，不能把它当作三次独立训练。', '',
        '| 任务 | 模型 | AUC | AP | 按药物宏 AP | 全候选已知留出正例 Recall@20 |',
        '|---|---|---:|---:|---:|---:|']
    for task in p['tasks']:
        for arm in ('base_gnn', 'tabm_prior', 'pointwise', 'set_plain', 'point_denoise', 'set_denoise'):
            r = next(v for v in summary_rows if v['task'] == task and v['arm'] == arm)
            recall = '—' if np.isnan(r['full_candidate_macro_recall20']) else '%.6f' % r['full_candidate_macro_recall20']
            text.append('| %s | %s | %.6f | %.6f | %.6f | %s |' % (task, arm, r['auc'], r['ap'], r['macro_drug_ap'], recall))
    text += ['', '## 主候选相对对照', '',
             '| 任务 | 对照 | Δ AUC | Δ AP | Δ 宏 AP | 宏 AP 上升种子 | 筛选 |',
             '|---|---|---:|---:|---:|---:|---|']
    for r in screens:
        text.append('| %s | %s | %+.6f | %+.6f | %+.6f | %d/%d | %s |' % (
            r['task'], r['comparison'], r['delta_auc'], r['delta_ap'], r['delta_macro_ap'],
            r['positive_macro_seeds'], len(p['refit_seeds']), '通过' if r['passed'] else '未通过'))
    teachers = (p['folds'] + 1) * 2 * len(p['tasks'])
    text += ['', '## 训练与结论范围', '',
        f'- 完成 {len(rows)} 个精排实验作业，其中 {sum(r["trained_steps"] > 0 for r in rows)} 个实际执行优化；零步重训表示内部选择了原样回退，不是新模型提升。',
        f'- 单独拟合 {teachers} 个 TabM 初始模型，训练精排器的每一折均排除该折标签，并在其拟合集内部再次生成折外结构特征。',
        '- 本轮内部划分来自原外层训练部分，种子预先固定。超参数和重训步数在打开外层评价前固定；外层仍是历史上反复使用的开发验证数据。',
        '- GNN 行复用此前同一外层样本的冻结结果，其参数未在本轮重调。四组精排对照使用相同新特征、损失和搜索预算，机制比较应以四组之间的差异为依据。',
        '- 宏 AP/样本内 Top-K 只覆盖含两类评测标签的药物；全候选 Recall@20 则在全部未观测候选中检索已知留出正例，不能解释为实验阳性率。',
        '- 采样零标签没有变成真实阴性，模型没有获得因果或校准的生物学概率保证。其他 RNA 任务仍按现有协议保留，因此不构成严格冷药物实验。',
        '- `no_support`、`shuffle_support` 只干预正例状态 token，固定结构特征与初始分数；它们不是删除图中全部相关证据的因果实验。',
        '- 推理使用一次终端前向。多正例遮蔽训练和低秩集合交互是本原型的实现，不能当作官方 RADD/KREPE 的复现或已确认的原创贡献。', '',
        '实现与复现见 [SET_DENOISING.md](../../SET_DENOISING.md)。逐运行结果、参数决策、初始模型和检查点均保存在本目录。测试集未使用。', '']
    (out / 'REPORT.md').write_text('\n'.join(text))
    validations = [out/('VALIDATION_'+t+'.json') for t in p['tasks']]
    if all(path.exists() for path in validations):
        parts = [json.loads(path.read_text()) for path in validations]
        parent.write_json(out/'VALIDATION.json', dict(passed=all(v['passed'] for v in parts), smoke=p['smoke'],
            runs=sum(v['runs'] for v in parts), teacher_models=sum(v['teacher_models'] for v in parts),
            protocol_sha256=p['sha256'], parts={path.name:parent.digest(path) for path in validations}))
    print('REPORT', out, 'all_screens_passed', all_passed, flush=True)


def validate(args):
    out, p = Path(args.out_dir), protocol(args.out_dir)
    for task in ([args.task] if args.task else p['tasks']):
        run_records, teacher_records = [], []
        for phase in ('inner', 'outer'):
            pack = FeaturePack(out, p, task, phase, args.device)
            a, graph, _ = parent.stage_inputs(out, task, phase)
            encoder = PairFeatureEncoder(graph, task)
            pairs, labels, folds = a['pairs_train'], a['y_train'], pack.data['fold_ids']
            for i, scenario in enumerate(pack.scenarios):
                fold = i if i < p['folds'] else None
                name = 'full' if fold is None else 'f'+str(fold)
                folder = pack.folder / name
                m = checked_manifest(folder, p)
                support, query, target, fit = support_and_queries(pairs, labels, folds, fold, encoder.nr, encoder.nd)
                for key, expected in [('support', support), ('query', query), ('target', target)]:
                    np.testing.assert_array_equal(scenario[key].cpu().numpy(), expected)
                with np.load(folder/'teacher_fit.npz') as archive:
                    fit_pairs, fit_y, train_x, nested = archive['pairs'], archive['y'], archive['x'], archive['nested_fold_ids']
                np.testing.assert_array_equal(fit_pairs, pairs[:, fit])
                np.testing.assert_array_equal(fit_y, labels[fit])
                if fold is not None and set(map(tuple, fit_pairs.T)) & set(map(tuple, pairs[:, folds == fold].T)):
                    raise AssertionError('Teacher/query leakage')
                with (folder/'scaler.pkl').open('rb') as stream:
                    scaler = pickle.load(stream)
                # Recompute all nested teacher training rows, including their
                # fold-specific graph supports, independently of saved x.
                raw = np.zeros_like(train_x)
                for f in range(p['folds']):
                    ids = np.flatnonzero(nested == f)
                    fit_pos = fit_pairs[:, (nested != f) & (fit_y == 1)]
                    raw[ids] = encoder.transform(fit_pairs[:, ids], fit_pos).values
                np.testing.assert_allclose(scaler.transform(raw), train_x, atol=3e-6, rtol=3e-6)
                dd, rr = np.nonzero(~support)
                sample = np.random.RandomState(117 + i).choice(len(dd), size=min(64, len(dd)), replace=False)
                raw_query = encoder.transform(np.stack([rr[sample], dd[sample]]), fit_pairs[:, fit_y == 1]).values
                expected_x = scaler.transform(raw_query)
                grid_x = scenario['features'].cpu().numpy()
                np.testing.assert_allclose(expected_x, grid_x[dd[sample], rr[sample]], atol=3e-6, rtol=3e-6)
                teacher = make_teacher(train_x.shape[1], m['teacher_recipe'], m['seed'], args.device)
                teacher.load_state_dict(torch.load(folder/'teacher.pt', map_location=args.device, weights_only=False))
                prob = np.clip(teacher_predict(teacher, grid_x[dd, rr], p['prediction_batch_size']), 1e-6, 1-1e-6)
                replay = np.log(prob)-np.log1p(-prob)
                error = float(np.max(np.abs(replay - scenario['base'].cpu().numpy()[dd, rr])))
                np.testing.assert_allclose(replay, scenario['base'].cpu().numpy()[dd, rr], atol=3e-5, rtol=3e-5)
                teacher_records.append(dict(phase=phase, scenario=name, max_logit_error=error,
                    nested_training_rows=len(fit), unknown_pairs=len(dd), query_features_checked=len(sample)))
                del teacher
            stages = ('screen', 'confirm') if phase == 'inner' else ('outer',)
            for stage in stages:
                for t, arm, cfg, seed, steps in jobs(out, p, stage):
                    if t != task:
                        continue
                    folder = directory(out, stage, task, arm, cfg['id'], seed)
                    result = checked_result(folder, p)
                    model = make_model(pack, cfg, seed)
                    model.load_state_dict(torch.load(folder/'checkpoint.pt', map_location=args.device, weights_only=False))
                    score, grid = predict(model, pack, arm, p['batch_drugs'])
                    with np.load(folder/'scores.npz') as archive:
                        np.testing.assert_array_equal(archive['y_val'], pack.data['y_val'])
                        np.testing.assert_array_equal(archive['pairs_val'], pack.data['pairs_val'])
                        saved = archive['val'].copy()
                        np.testing.assert_allclose(score, saved, atol=3e-5, rtol=3e-5)
                        np.testing.assert_allclose(grid, archive['full_grid'], atol=3e-5, rtol=3e-5)
                    metrics, _ = evaluate(score, grid, pack)
                    for key, value in metrics.items():
                        if not np.isclose(value, result[key], atol=2e-5, rtol=2e-5):
                            raise AssertionError('Metric replay failed: '+key)
                    trace = pd.read_csv(folder/'trace.csv').to_dict('records')
                    if stage != 'outer':
                        chosen, value = choose_checkpoint(trace, p['smoothing_window'])
                        if chosen != result['selected_step'] or not np.isclose(value, result['selection_score'], atol=1e-10):
                            raise AssertionError('Checkpoint rule did not reproduce')
                    elif result['selected_step'] != steps:
                        raise AssertionError('Outer step count was not frozen')
                    if result['params'] != sum(v.numel() for v in model.parameters()):
                        raise AssertionError('Parameter count')
                    run_records.append(dict(stage=stage, arm=arm, config_id=cfg['id'], seed=seed,
                        max_logit_error=float(np.max(np.abs(score-saved)))))
                    del model
            del pack
        validation = dict(passed=True, smoke=p['smoke'], protocol_sha256=p['sha256'], task=task,
            runs=len(run_records), teacher_models=len(teacher_records), reranker_replay=run_records,
            teacher_replay=teacher_records, no_test_evaluation=True)
        parent.write_json(out/('VALIDATION_'+task+'.json'), validation)
        print('VALIDATED', task, len(run_records), 'runs', len(teacher_records), 'teachers', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('prepare', 'features', 'run', 'select', 'report', 'validate'):
        q = sub.add_parser(command)
        q.add_argument('--out-dir', required=True)
        if command == 'prepare':
            q.add_argument('--reference-dir', default='improve/results/backbone_replacement_20260910')
            q.add_argument('--tabm-dir', default='improve/results/graph_tabm_20260910')
            q.add_argument('--smoke', action='store_true')
        if command in ('features', 'run', 'select'):
            q.add_argument('--stage', required=True)
        if command in ('features', 'run', 'validate'):
            q.add_argument('--task', choices=list(parent.TASKS))
            q.add_argument('--device', default='cuda:0')
        if command == 'run':
            q.add_argument('--arm', choices=list(ARMS))
    args = parser.parse_args()
    dispatch = dict(prepare=prepare, features=feature_command, run=run, select=select,
                    report=report, validate=validate)
    dispatch[args.command](args)


if __name__ == '__main__':
    main()
