"""TGSBM-inspired experiment with explicit strong-reference and mechanism gates.

prepare -> screen (seed 666) -> select -> confirm (0/1) -> report.
Only the historical development partition is used here. A passing result is
still a candidate for separately frozen resplit confirmation, not a success.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import shutil
import time

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold
import torch
from torch.nn import functional as F

from improve import non_tabular_experiment as base
from improve import overlap_block_models as model_lib

OUT = base.ROOT / 'results/overlap_block_20260915'
PHYSICAL = base.ROOT / 'results/typed_sheaf_20260914/inputs'
REF = base.ROOT / 'results/tabicl_pathway_context_20260913/predictions/inner'
KINDS = ('full', 'deterministic', 'no_stick', 'no_expander')


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in z.files}


def prepare(out, smoke=False):
    from improve import context_anchor_verify as anchor
    out = Path(out)
    if out.exists():
        raise ValueError('Use a fresh output directory')
    out.mkdir(parents=True)
    (out / 'source_snapshot').mkdir()
    own = ('overlap_block_models.py', 'overlap_block_experiment.py', 'test_overlap_block.py')
    sources = own + ('non_tabular_experiment.py', 'context_anchor_verify.py', 'anchor_shrinkage.py')
    for name in sources:
        shutil.copy2(base.ROOT / name, out / 'source_snapshot' / name)
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), smoke=smoke,
             tasks=list(base.TASKS), screen_seed=666, confirm_seeds=[0, 1],
             kinds=list(KINDS), width=32 if smoke else 64, depth=1 if smoke else 3,
             heads=4, communities=16 if smoke else 32, dropout=.1, expander_degree=6,
             overlay_seed=20260915, fold_seed=20260915, folds=5,
             lrs=[.001] if smoke else [.001, .003], epochs=2 if smoke else 400,
             check_interval=1 if smoke else 20, weight_decay=.001,
             beta=.001, temperature=.8, posterior_samples=2 if smoke else 16,
             strong_weight=.1, gnn_weight=.5, lnc_absolute_tolerance=.001,
             source_sha256={n: base.digest(base.ROOT/n) for n in sources}, inputs={},
             literature=dict(title='TGSBM: Transformer-Guided Stochastic Block Model for Link Prediction',
                             doi='10.1145/3774904.3792497', arxiv='2601.20646v1'),
             adaptation='Typed bipartite endpoint readouts; cumulative-product Beta(10,1) overlapping-feature prior instead of the paper exclusive stick formula; exact Gamma-ratio Beta sampling and logistic Binary-Concrete noise; beta-weighted per-variable mean KL; no feature reconstruction. These changes are explicit, not an official reproduction.',
             boundary='Every queried target association is removed before physical counts and inverse arcs. Label-blind expander arcs use their own type; they are not observed associations. Evaluation averages the five matched support graphs.',
             selection='Each run selects one checkpoint by mean global AUC/AP of the fixed 0.1-logit correction to the shared-anchor reference of the same inference seed. Select one shared learning rate per mechanism on screening results only, prioritizing the joint MiDrug-up/LncDrug-loss<=0.001 gate, then mean MiDrug AUC/AP gain. No coefficient sweep.',
             gate='MiDrug mean AUC and AP increase, with at least 2/3 paired seeds positive each; LncDrug mean and every seed drop <=0.001 absolute on both. Full mechanism must exceed every independently selected matched control on both mean MiDrug metrics.',
             success_scope='A historical inner pass is screening evidence only. Confirmation on new frozen partitions and endpoint-group uncertainty is required before counting an innovation.',
             no_original_test_access=True, no_original_outer_query_label_access=True,
             existing_default_replaced=False)
    for task in p['tasks']:
        path = PHYSICAL/task/'arrays.npz'
        p['inputs'][str(path)] = base.digest(path)
        a = load(path)
        folds = np.zeros(len(a['y_train']), dtype=np.int64)
        for f, (_, ix) in enumerate(StratifiedKFold(5, shuffle=True, random_state=p['fold_seed']).split(
                a['pairs_train'].T, a['y_train'])):
            folds[ix] = f
        a['folds'] = folds
        dest = out/'inputs'/task
        dest.mkdir(parents=True)
        bank_path = anchor.INNER / ('context76_'+task+'.npz')
        bank = load(bank_path)
        p['inputs'][str(bank_path)] = base.digest(bank_path)
        for key in ('pairs_train', 'pairs_val', 'y_train', 'y_val'):
            np.testing.assert_array_equal(bank[key], a[key])
        members, logits = {}, []
        if smoke:
            component = np.zeros(len(a['y_val']))
        else:
            for name in ('hgb', 'et', 'mlp', 'lr'):
                member = anchor.learner(name)
                member.fit(bank['x_train'], bank['y_train'])
                members[name] = member
                probability = member.predict_proba(bank['x_val'])[:, 1]
                logits.append(anchor.anchors.logit(probability))
            component = np.stack(logits).mean(0)
            with (dest/'anchor_members.pkl').open('wb') as stream:
                pickle.dump(members, stream, protocol=4)
        historical = pd.read_csv(anchor.OUT/'metrics.csv') if not smoke else None
        anchor_replay = []
        for seed in (666, 0, 1):
            rp = REF/task/('s'+str(seed))/'scores.npz'
            p['inputs'][str(rp)] = base.digest(rp)
            z = load(rp)
            for key in ('pairs_val', 'y_val'):
                np.testing.assert_array_equal(a[key], z[key])
            a['tabicl_'+str(seed)] = logit(z['tiered_0p5000'].astype(np.float64))
            a['anchor_'+str(seed)] = .7*a['tabicl_'+str(seed)] + .3*component
            if not smoke:
                expected = historical[(historical.recipe == 'context76|hgb+et+mlp+lr|w0.30') &
                    (historical.scope == 'inner') & (historical.task == task) &
                    (historical.key == 'seed'+str(seed))].iloc[0]
                actual = base.metrics(a['y_val'], expit(a['anchor_'+str(seed)]), a['pairs_val'])
                for metric in ('auc', 'ap'):
                    if abs(actual[metric] - expected[metric]) > 1e-10:
                        raise AssertionError('Shared-anchor reconstruction mismatch')
                anchor_replay.append(dict(seed=seed, auc=actual['auc'], ap=actual['ap']))
            phase = 'screen' if seed == 666 else 'confirm'
            gp = base.GRAPH/phase/task/'base/c0'/('s'+str(seed))/'scores.npz'
            p['inputs'][str(gp)] = base.digest(gp)
            z = load(gp)
            np.testing.assert_array_equal(z['pairs_val'], a['pairs_val'])
            np.testing.assert_array_equal(z['y_val'], a['y_val'])
            a['gnn_'+str(seed)] = z['val']
        np.savez_compressed(dest/'arrays.npz', **a)
        base.write_json(dest/'reference_replay.json', anchor_replay)
        p['inputs'][str(dest/'arrays.npz')] = base.digest(dest/'arrays.npz')
        print('PREPARED', task, 'anchor replay', anchor_replay, flush=True)
    p['sha256'] = base.digest(out/'source_snapshot/overlap_block_experiment.py')
    base.write_json(out/'protocol.json', p)
    print('FROZEN', out, flush=True)


def protocol(out):
    out = Path(out)
    p = json.loads((out/'protocol.json').read_text())
    for n, sha in p['source_sha256'].items():
        if base.digest(base.ROOT/n) != sha:
            raise ValueError('Frozen source changed: '+n)
    for path, sha in p['inputs'].items():
        if base.digest(path) != sha:
            raise ValueError('Frozen input changed: '+path)
    return p


def setup(a, p, kind, device):
    n = int(a['node_counts'].sum())
    nr = int(a['relation'].max()+1)
    overlay = model_lib.expander_edges(n, p['expander_degree'], p['overlay_seed'])
    if kind == 'no_expander':
        overlay = overlay[:, :0]
    graphs, queries = [], []
    for f in range(p['folds']):
        ix = np.flatnonzero(a['folds'] == f)
        edges, relation = model_lib.masked_physical(a, ix)
        graphs.append(model_lib.graph_inputs(edges, relation, overlay, n, nr, device))
        queries.append(torch.as_tensor(ix, device=device))
    xs = [torch.as_tensor(a['x_'+str(i)], device=device) for i in range(len(a['node_counts']))]
    model = model_lib.OverlapBlock([x.shape[1] for x in xs], list(map(int, a['node_counts'])),
                nr, p['width'], p['depth'], p['heads'], p['communities'], p['dropout'], kind).to(device)
    return model, xs, graphs, queries


def train(a, p, kind, lr, seed, device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, xs, graphs, queries = setup(a, p, kind, device)
    pt = torch.as_tensor(a['pairs_train']+a['pair_offset'][:, None], device=device)
    pv = torch.as_tensor(a['pairs_val']+a['pair_offset'][:, None], device=device)
    y = torch.as_tensor(a['y_train'], device=device)
    generator = torch.Generator(device=device).manual_seed(seed+29015)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=p['weight_decay'])
    best, best_epoch, state = -np.inf, None, None
    trace = []
    start = time.monotonic()
    for epoch in range(1, p['epochs']+1):
        f = (epoch-1) % len(graphs)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred, kl = model(xs, graphs[f], pt[:, queries[f]], generator, p['temperature'])
        bce = F.binary_cross_entropy_with_logits(pred, y[queries[f]])
        anneal = min(1., epoch/(p['epochs']*.2))
        loss = bce + p['beta']*anneal*kl
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training objective')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if epoch % p['check_interval'] == 0 or epoch == p['epochs']:
            candidate = model_lib.posterior_prediction(model, xs, graphs, pv,
                samples=p['posterior_samples']).cpu().numpy()
            mixed = (1-p['strong_weight'])*a['anchor_'+str(seed)] + p['strong_weight']*candidate
            mm = base.metrics(a['y_val'], mixed, a['pairs_val'])
            raw = base.metrics(a['y_val'], candidate, a['pairs_val'])
            criterion = (mm['auc']+mm['ap'])/2
            trace.append(dict(epoch=epoch, bce=float(bce.detach()), kl=float(kl.detach()),
                              anchor_composed=mm, standalone=raw))
            if criterion > best:
                best, best_epoch = criterion, epoch
                state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            if epoch % max(100, p['check_interval']) == 0:
                print('TRAIN', kind, lr, seed, epoch, round(raw['auc'], 6),
                      round(raw['ap'], 6), round(time.monotonic()-start, 1), flush=True)
    model.load_state_dict(state)
    pred = model_lib.posterior_prediction(model, xs, graphs, pv,
        samples=p['posterior_samples']).cpu().numpy()
    return model, pred, trace, dict(epoch=best_epoch, seconds=time.monotonic()-start,
        parameters=sum(v.numel() for v in model.parameters()))


def verify(dest, protocol_hash):
    result = json.loads((dest/'result.json').read_text())
    if result['protocol_sha256'] != protocol_hash:
        raise ValueError('Wrong run protocol')
    for n, sha in result['files'].items():
        if base.digest(dest/n) != sha:
            raise ValueError('Result artifact changed')
    return result


def run(out, task, device, stage='screen'):
    out = Path(out)
    p = protocol(out)
    a = load(out/'inputs'/task/'arrays.npz')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol_hash = base.digest(out/'protocol.json')
    selected = None if stage == 'screen' else json.loads((out/'SELECTION.json').read_text())
    seeds = [p['screen_seed']] if stage == 'screen' else p['confirm_seeds']
    for kind in p['kinds']:
        lrs = p['lrs'] if stage == 'screen' else [selected['chosen'][kind]['lr']]
        for lr in lrs:
            for seed in seeds:
                dest = out/'runs'/task/kind/('lr'+str(lr))/('s'+str(seed))
                if (dest/'result.json').exists():
                    verify(dest, protocol_hash)
                    continue
                dest.mkdir(parents=True, exist_ok=True)
                model, pred, trace, detail = train(a, p, kind, lr, seed, device)
                torch.save(model.state_dict(), dest/'model.pt')
                replay_model, xs, graphs, _ = setup(a, p, kind, device)
                replay_model.load_state_dict(torch.load(dest/'model.pt', map_location=device,
                                                        weights_only=True), strict=True)
                pv = torch.as_tensor(a['pairs_val']+a['pair_offset'][:, None], device=device)
                replay = model_lib.posterior_prediction(replay_model, xs, graphs, pv,
                    samples=p['posterior_samples']).cpu().numpy()
                reverse = model_lib.posterior_prediction(replay_model, xs, graphs, pv.flip(1),
                    samples=p['posterior_samples']).cpu().numpy()[::-1]
                for value in (replay, reverse):
                    np.testing.assert_allclose(pred, value, atol=3e-5, rtol=3e-5)
                scores = dict(y=a['y_val'], pairs=a['pairs_val'], standalone=pred)
                for ref in ('gnn', 'tabicl', 'anchor'):
                    scores[ref] = a[ref+'_'+str(seed)]
                    weight = p['gnn_weight'] if ref == 'gnn' else p['strong_weight']
                    scores[ref+'_composed'] = (1-weight)*scores[ref] + weight*pred
                np.savez_compressed(dest/'scores.npz', **scores)
                base.write_json(dest/'trace.json', trace)
                result = dict(task=task, kind=kind, lr=lr, seed=seed, **detail,
                    protocol_sha256=protocol_hash,
                    replay_max_abs=float(max(np.max(np.abs(pred-replay)), np.max(np.abs(pred-reverse)))),
                    metrics={name:base.metrics(scores['y'], value, scores['pairs'])
                             for name,value in scores.items() if name not in ('y', 'pairs')},
                    files={n:base.digest(dest/n) for n in ('model.pt', 'scores.npz', 'trace.json')})
                base.write_json(dest/'result.json', result)
                print('VERIFIED', task, kind, lr, seed, result['epoch'],
                      result['metrics']['anchor_composed'], flush=True)
                del model, replay_model, xs, graphs
                torch.cuda.empty_cache()


def all_rows(out):
    out = Path(out)
    h = base.digest(out/'protocol.json')
    rows = []
    for dest in sorted((out/'runs').glob('*/*/*/*')):
        if not (dest/'result.json').exists():
            continue
        r = verify(dest, h)
        z = load(dest/'scores.npz')
        for name in r['metrics']:
            actual = base.metrics(z['y'], z[name], z['pairs'])
            for key in ('auc', 'ap', 'macro_drug_ap'):
                if abs(actual[key]-r['metrics'][name][key]) > 1e-12:
                    raise AssertionError('Stored metric mismatch')
            rows.append(dict(task=r['task'], kind=r['kind'], lr=r['lr'], seed=r['seed'],
                             epoch=r['epoch'], score=name, **actual))
    return pd.DataFrame(rows)


def select(out):
    out = Path(out)
    p = protocol(out)
    table = all_rows(out)
    chosen = {}
    for kind in p['kinds']:
        candidates = []
        for lr in p['lrs']:
            differences = {}
            for task in p['tasks']:
                t = table[(table.task == task) & (table.kind == kind) & (table.lr == lr) &
                          (table.seed == p['screen_seed'])].set_index('score')
                differences[task] = {m:float(t.loc['anchor_composed',m]-t.loc['anchor',m])
                                     for m in ('auc', 'ap')}
            passes = all(differences['MiDrug'][m] > 0 and
                         differences['LncDrug'][m] >= -p['lnc_absolute_tolerance'] for m in ('auc','ap'))
            candidates.append(dict(lr=lr, gate=passes, differences=differences,
                priority=sum(differences['MiDrug'].values())/2))
        chosen[kind] = max(candidates, key=lambda x:(x['gate'], x['priority'], -x['lr']))
    result = dict(protocol_sha256=base.digest(out/'protocol.json'), chosen=chosen,
                  screen_only=True, passes_full_screen=chosen['full']['gate'],
                  original_test_access=False, innovations_confirmed=0)
    base.write_json(out/'SELECTION.json', result)
    print(json.dumps(result, indent=2), flush=True)


def report(out):
    out = Path(out)
    p = protocol(out)
    table = all_rows(out)
    table.to_csv(out/'metrics.csv', index=False)
    contrasts = []
    for (task,kind,lr,seed), sub in table.groupby(['task','kind','lr','seed']):
        t = sub.set_index('score')
        for ref in ('gnn','tabicl','anchor'):
            contrasts.append(dict(task=task,kind=kind,lr=lr,seed=seed,reference=ref,
                **{m:float(t.loc[ref+'_composed',m]-t.loc[ref,m]) for m in ('auc','ap','macro_drug_ap')}))
    contrasts = pd.DataFrame(contrasts)
    contrasts.to_csv(out/'contrasts.csv', index=False)
    s = contrasts.groupby(['task','kind','lr','reference'])[['auc','ap','macro_drug_ap']].mean()
    s.to_csv(out/'summary.csv')
    base.write_json(out/'STATUS.json', dict(completed_fits=len(table)//7,
        source_and_input_hashes_verified=True, all_completed_fit_metrics_recomputed=True,
        confirmed_innovations=0, original_test_access=False, accepted_default_replaced=False,
        note='Development screening/initialization checks only; new-partition confirmation remains required.'))
    print(s.xs('anchor', level='reference').to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare','screen','select','confirm','report'))
    parser.add_argument('--out-dir', default=str(OUT))
    parser.add_argument('--task', choices=base.TASKS)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.out_dir, args.smoke)
    elif args.command in ('screen','confirm'):
        run(args.out_dir, args.task, args.device, args.command)
    elif args.command == 'select':
        select(args.out_dir)
    else:
        report(args.out_dir)


if __name__ == '__main__':
    main()
