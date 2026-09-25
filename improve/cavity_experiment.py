"""NBA-GNN adaptation with labeled-support edges and a backtracking control."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from improve import graph_candidate_runner as runner
from improve import cavity_graph as method

common = runner.common
OUT = common.base.ROOT / 'results/cavity_graph_20260915'


def parameters(smoke=False):
    return dict(method='cavity', executor_module='improve.cavity_experiment', kinds=list(method.KINDS),
        width=16 if smoke else 64, depth=1 if smoke else 4, dropout=.1, folds=5,
        epochs=2 if smoke else 400, check_interval=1 if smoke else 20,
        lrs=[.001] if smoke else [.001, .003], weight_decay=.001,
        strong_weight=.1, gnn_weight=.5, lnc_absolute_tolerance=.001)


def prepare(out):
    out = Path(out).resolve()
    if out.exists():
        raise ValueError('Use a new experiment directory')
    parent = common.protocol(common.OUT)
    own = ('cavity_graph.py', 'cavity_experiment.py', 'test_cavity_graph.py',
           'graph_candidate_runner.py', 'test_graph_candidate_runner.py',
           'graph_prediction_precision.py', 'overlap_block_models.py',
           'overlap_block_experiment.py', 'non_tabular_experiment.py')
    p = dict(**parameters(), created_at_utc=datetime.now(timezone.utc).isoformat(), smoke=False,
        tasks=parent['tasks'], screen_seed=666, confirm_seeds=[0, 1],
        selection=parent['selection'], primary_kind='full', primary_controls=['backtracking'],
        gate='MiDrug AUC/AP mean increase and each positive in at least 2/3 confirmations; every LncDrug loss <=0.001 absolute. Full must beat backtracking on both mean MiDrug metrics. Positive-only and edge-MLP diagnose the added observation channel and recurrent propagation, and are not counted as distinct techniques.',
        success_scope='Development confirmation only until further frozen evaluation; no noninferiority or independent biological validation claim.',
        boundary='Each queried fitting pair is deleted from positive and sampled-zero support before physical inverse arcs, degrees, message initialization or updates. Original validation positives are absent from all supplied graphs. Other-task positive relations remain in the historical transductive protocol.',
        literature=[dict(title='Non-backtracking Graph Neural Networks', arxiv='2310.07430v2',
                         venue='Transactions on Machine Learning Research, September 2024',
                         url='https://openreview.net/forum?id=64HdQKnyTc')],
        adaptation='Typed 64-dimensional directed edge states, four residual cavity layers, mean aggregation, both endpoint features at initialization, separate incoming/outgoing pooling. Adds sampled zero-labeled fitting pairs as an observation relation; this is not biological negative evidence. Uses incidence sums and grouped subtraction instead of materialized line-graph adjacency. All reverse arcs from the preceding endpoint are removed even across parallel relation types. Degree-one endpoint groups use the paper\'s begrudging return.',
        controls='Backtracking uses the exact same edge states, inputs, labels, parameters and optimization budget but includes reverse messages. Positive-only suppresses sampled-zero propagation while retaining its marginal degree counts. Edge-MLP disables cross-edge recurrent messages while preserving edge initialization, layers and pooling.',
        fresh_partition_scores_already_seen_for_other_methods=True,
        original_test_access=False, original_outer_query_label_access=False,
        existing_default_replaced=False, confirmed_innovations=0,
        source_sha256={n: common.base.digest(common.base.ROOT / n) for n in own}, inputs={})
    out.mkdir(parents=True)
    (out / 'source_snapshot').mkdir()
    for name in own:
        shutil.copy2(common.base.ROOT / name, out / 'source_snapshot' / name)
    for task in p['tasks']:
        source = common.OUT / 'inputs' / task / 'arrays.npz'
        dest = out / 'inputs' / task / 'arrays.npz'
        dest.parent.mkdir(parents=True)
        shutil.copy2(source, dest)
        for path in (source, dest):
            p['inputs'][str(path)] = common.base.digest(path)
    for name in ('nonbacktracking_gnn.pdf', 'nonbacktracking_metadata.json'):
        path = common.base.ROOT / 'research/new_mechanisms_20260915' / name
        p['inputs'][str(path)] = common.base.digest(path)
    common.base.write_json(out / 'protocol.json', p)
    print('FROZEN', out, flush=True)


def setup(a, p, kind, device):
    graphs, folds = [], []
    for fold in range(p['folds']):
        ids = np.flatnonzero(a['folds'] == fold)
        graphs.append(method.support_graph(a, ids, kind, device))
        folds.append(torch.as_tensor(ids, device=device))
    xs = [torch.as_tensor(a['x_' + str(i)], device=device) for i in range(len(a['node_counts']))]
    nr = int(a['relation'].max() + 2)
    model = method.CavityGraph([x.shape[1] for x in xs], list(map(int, a['node_counts'])), nr,
                              p['width'], p['depth'], p['dropout'], kind).to(device)
    return model, xs, graphs, folds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'screen', 'select', 'confirm', 'report'))
    parser.add_argument('--out', default=str(OUT))
    parser.add_argument('--task', choices=common.base.TASKS)
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if args.command == 'prepare':
        prepare(args.out)
    elif args.command in ('screen', 'confirm'):
        runner.run(args.out, args.task, args.device, args.command)
    elif args.command == 'select':
        common.select(args.out)
    else:
        common.report(args.out)
