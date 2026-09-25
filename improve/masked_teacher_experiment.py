"""Frozen joint-supervision study of typed masked graph reconstruction."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from improve import overlap_block_experiment as common
from improve import overlap_block_models as graph_lib
from improve import graph_prediction_precision as precision
from improve import masked_graph_teacher as method

OUT = common.base.ROOT / 'results/masked_graph_teacher_20260915'
KINDS = method.MaskedGraphTeacher.KINDS


def prepare(out, smoke=False):
    out = Path(out).resolve()
    if out.exists():
        raise ValueError('Use a new output directory')
    parent = common.protocol(common.OUT)
    own = ('masked_graph_teacher.py', 'masked_teacher_experiment.py', 'test_masked_graph_teacher.py',
           'graph_prediction_precision.py', 'overlap_block_experiment.py',
           'overlap_block_models.py', 'non_tabular_experiment.py')
    research = common.base.ROOT / 'research/new_mechanisms_20260915'
    sources = [research / 'graphmae2_paper.pdf', research / 'MASKED_SOURCE_METADATA.json']
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), method='masked_teacher', smoke=smoke,
        tasks=parent['tasks'], kinds=list(KINDS), screen_seed=666, confirm_seeds=[0, 1],
        width=16 if smoke else 64, depth=1 if smoke else 3, heads=4, dropout=.1,
        folds=5, epochs=2 if smoke else 400, check_interval=1 if smoke else 20,
        lrs=[.001] if smoke else [.001, .003], weight_decay=.001,
        auxiliary_weight=.1, mask_fraction=.5, remask_fraction=.5, remask_views=2, momentum=.99,
        strong_weight=.1, gnn_weight=.5, lnc_absolute_tolerance=.001,
        selection=parent['selection'],
        primary_kind='full', primary_controls=['bce'],
        gate='MiDrug mean global AUC and AP increase with at least 2/3 paired confirmations positive each. LncDrug every metric loses at most 0.001 absolute. Primary mechanism comparison is full versus bce on both mean MiDrug metrics. No-teacher and fixed-remask arms diagnose component claims separately; they are not additional techniques.',
        success_scope='Reused historical development evidence only until fixed choices are validated on additional partitions and final reserved evaluation. No biological or statistical noninferiority claim.',
        boundary=parent['boundary'].split('Label-blind')[0] + ' No expander. Teacher, encoder and decoder all use the identical support graph after query-fact deletion.',
        literature=[dict(title='GraphMAE2: A Decoding-Enhanced Masked Self-Supervised Graph Learner',
                         doi='10.1145/3543507.3583379', arxiv='2304.04779', read='Methods 2.1-2.2, equations 2-6 and architecture caption'),
                    dict(title='Rethinking link prediction: A multi-scale graph masked autoencoder',
                         doi='10.1016/j.neucom.2025.132115', read='Metadata only; not the implementation source')],
        adaptation='Joint BCE plus 0.1 times masked attribute cosine reconstruction and EMA latent prediction. Separate attribute decoders and mask tokens per node type. Two random remasked decoder views averaged rather than summed. Latent cosine includes all nodes. Original paper pretrains and then transfers; this is an explicitly different supervised regularization study.',
        controls='All arms have identical initialized modules and 400 optimizer steps. bce disables auxiliary losses; no_teacher omits the EMA target loss; fixed_remask uses the input mask for both decoder views, retaining matched decoder calls. Auxiliary arms incur extra compute, which is reported, not claimed to be compute matched.',
        fresh_partition_scores_already_seen_for_other_methods=True,
        original_test_access=False, original_outer_query_label_access=False,
        existing_default_replaced=False, confirmed_innovations=0,
        source_sha256={n: common.base.digest(common.base.ROOT / n) for n in own},
        literature_sha256={str(f): common.base.digest(f) for f in sources}, inputs={})
    out.mkdir(parents=True)
    (out / 'source_snapshot').mkdir()
    for name in own:
        shutil.copy2(common.base.ROOT / name, out / 'source_snapshot' / name)
    for task in p['tasks']:
        source = common.OUT / 'inputs' / task / 'arrays.npz'
        dest = out / 'inputs' / task / 'arrays.npz'
        dest.parent.mkdir(parents=True)
        shutil.copy2(source, dest)
        for f in (source, dest):
            p['inputs'][str(f)] = common.base.digest(f)
    common.base.write_json(out / 'protocol.json', p)
    print('FROZEN', out, flush=True)


def setup(a, p, kind, device):
    n, nr = int(a['node_counts'].sum()), int(a['relation'].max() + 1)
    graphs, folds = [], []
    for fold in range(p['folds']):
        ids = np.flatnonzero(a['folds'] == fold)
        edge, relation = graph_lib.masked_physical(a, ids)
        graphs.append(graph_lib.graph_inputs(edge, relation, np.empty((2, 0), dtype=np.int64), n, nr, device))
        folds.append(torch.as_tensor(ids, device=device))
    xs = [torch.as_tensor(a['x_' + str(i)], device=device) for i in range(len(a['node_counts']))]
    model = method.MaskedGraphTeacher([x.shape[1] for x in xs], list(map(int, a['node_counts'])), nr,
        p['width'], p['depth'], p['heads'], p['dropout'], kind,
        p['mask_fraction'], p['remask_fraction'], p['remask_views'], p['momentum']).to(device)
    return model, xs, graphs, folds


def train(a, p, kind, lr, seed, device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, xs, graphs, folds = setup(a, p, kind, device)
    pt = torch.as_tensor(a['pairs_train'] + a['pair_offset'][:, None], device=device)
    pv = torch.as_tensor(a['pairs_val'] + a['pair_offset'][:, None], device=device)
    y = torch.as_tensor(a['y_train'], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=p['weight_decay'])
    generator = torch.Generator(device=device).manual_seed(seed + 17057)
    best, state, best_epoch = -np.inf, None, None
    trace, start = [], time.monotonic()
    for epoch in range(1, p['epochs'] + 1):
        fold = (epoch - 1) % len(graphs)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pred = model(xs, graphs[fold], pt[:, folds[fold]])
        bce = F.binary_cross_entropy_with_logits(pred, y[folds[fold]])
        auxiliary, parts = model.auxiliary(xs, graphs[fold], generator)
        loss = bce + p['auxiliary_weight'] * auxiliary
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite joint masked graph loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        model.update_teacher()
        if epoch % p['check_interval'] == 0 or epoch == p['epochs']:
            prediction = precision.geometry_prediction(model, xs, graphs, pv).cpu().numpy()
            mixed = (1 - p['strong_weight']) * a['anchor_' + str(seed)] + p['strong_weight'] * prediction
            mm = common.base.metrics(a['y_val'], mixed, a['pairs_val'])
            raw = common.base.metrics(a['y_val'], prediction, a['pairs_val'])
            criterion = (mm['auc'] + mm['ap']) / 2
            trace.append(dict(epoch=epoch, loss=float(loss.detach()), bce=float(bce.detach()),
                              **{k: float(v) for k, v in parts.items()}, anchor_composed=mm, standalone=raw))
            if criterion > best:
                best, best_epoch = criterion, epoch
                state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if epoch % max(100, p['check_interval']) == 0:
                print('TRAIN', kind, lr, seed, epoch, round(raw['auc'], 6), round(raw['ap'], 6),
                      round(time.monotonic() - start, 1), flush=True)
    model.load_state_dict(state)
    prediction = precision.geometry_prediction(model, xs, graphs, pv).cpu().numpy()
    return model, prediction, trace, dict(epoch=best_epoch, seconds=time.monotonic() - start,
        parameters=sum(v.numel() for v in model.parameters()),
        trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad))


def run(out, task, device, stage):
    out = Path(out)
    p = common.protocol(out)
    a = common.load(out / 'inputs' / task / 'arrays.npz')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    ph = common.base.digest(out / 'protocol.json')
    selected = None if stage == 'screen' else json.loads((out / 'SELECTION.json').read_text())
    if selected is not None and selected['protocol_sha256'] != ph:
        raise ValueError('Selection protocol changed')
    seeds = [p['screen_seed']] if stage == 'screen' else p['confirm_seeds']
    for kind in p['kinds']:
        for lr in p['lrs'] if stage == 'screen' else [selected['chosen'][kind]['lr']]:
            for seed in seeds:
                dest = out / 'runs' / task / kind / ('lr' + str(lr)) / ('s' + str(seed))
                if (dest / 'result.json').exists():
                    common.verify(dest, ph)
                    continue
                dest.mkdir(parents=True, exist_ok=True)
                model, pred, trace, detail = train(a, p, kind, lr, seed, device)
                torch.save(model.state_dict(), dest / 'model.pt')
                common.base.write_json(dest / 'trace.json', trace)
                common.base.write_json(dest / 'training_detail.json', detail)
                loaded, xs, graphs, _ = setup(a, p, kind, device)
                loaded.load_state_dict(torch.load(dest / 'model.pt', map_location=device, weights_only=True))
                pairs = torch.as_tensor(a['pairs_val'] + a['pair_offset'][:, None], device=device)
                replay = precision.geometry_prediction(loaded, xs, graphs, pairs.flip(1)).cpu().numpy()[::-1]
                chunks = np.concatenate([precision.geometry_prediction(loaded, xs, graphs, pairs[:, i:i + 800]).cpu().numpy()
                                         for i in range(0, pairs.shape[1], 800)])
                for value in (replay, chunks):
                    np.testing.assert_allclose(pred, value, atol=3e-5, rtol=3e-5)
                scores = dict(y=a['y_val'], pairs=a['pairs_val'], standalone=pred)
                for name in ('gnn', 'tabicl', 'anchor'):
                    scores[name] = a[name + '_' + str(seed)]
                    weight = p['gnn_weight'] if name == 'gnn' else p['strong_weight']
                    scores[name + '_composed'] = (1 - weight) * scores[name] + weight * pred
                np.savez_compressed(dest / 'scores.npz', **scores)
                result = dict(task=task, kind=kind, lr=lr, seed=seed, **detail, protocol_sha256=ph,
                    replay_max_abs=float(max(abs(pred - replay).max(), abs(pred - chunks).max())),
                    metrics={n: common.base.metrics(scores['y'], value, scores['pairs'])
                             for n, value in scores.items() if n not in ('y', 'pairs')},
                    files={n: common.base.digest(dest / n) for n in ('model.pt', 'trace.json', 'training_detail.json', 'scores.npz')})
                common.base.write_json(dest / 'result.json', result)
                print('VERIFIED', task, kind, lr, seed, result['epoch'], result['metrics']['anchor_composed'], flush=True)
                del model, loaded, xs, graphs
                torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'screen', 'select', 'confirm', 'report'))
    parser.add_argument('--out', default=str(OUT))
    parser.add_argument('--task', choices=common.base.TASKS)
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if args.command == 'prepare':
        prepare(args.out, args.smoke)
    elif args.command in ('screen', 'confirm'):
        run(args.out, args.task, args.device, args.command)
    elif args.command == 'select':
        common.select(args.out)
    else:
        common.report(args.out)
