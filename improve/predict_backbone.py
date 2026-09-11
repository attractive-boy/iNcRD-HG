"""Score new RNA/drug pairs with a verified replacement-backbone checkpoint.

Examples:
  python -m improve.predict_backbone --task LncDrug --drug-index 0 --out scores.csv
  python -m improve.predict_backbone --task MiDrug --pairs pairs.csv --out scores.csv

CSV inputs contain rna_index and drug_index (indices in the frozen graph).
The default is the TabM experiment, using the predetermined seed 666.
"""
import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch

from improve import backbone_replacement as graph_runner
from improve import graph_tabm as table_runner
from improve.graph_tabm_modules import PairFeatureEncoder
from improve.pencil_backbone import SupportGraph


def read_pairs(args, arrays, data, rna):
    n_rna, n_drug = data[rna].num_nodes, data['drug'].num_nodes
    positive = arrays['pairs_train'][:, arrays['y_train'] == 1]
    known = positive[0]*n_drug + positive[1]
    if args.drug_index is not None:
        if not 0 <= args.drug_index < n_drug:
            raise ValueError('drug-index is outside the frozen graph')
        pairs = np.stack([np.arange(n_rna), np.full(n_rna, args.drug_index)])
        pairs = pairs[:, ~np.isin(pairs[0]*n_drug+pairs[1], known)]
    else:
        frame = pd.read_csv(args.pairs)
        if not {'rna_index', 'drug_index'}.issubset(frame.columns):
            raise ValueError('CSV requires rna_index and drug_index columns')
        values = frame[['rna_index', 'drug_index']].to_numpy()
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError('Pair indices must be finite integers')
        pairs = values.astype(np.int64).T
        if np.any(pairs < 0) or np.any(pairs[0] >= n_rna) or np.any(pairs[1] >= n_drug):
            raise ValueError('Pair index is outside the frozen graph')
        if np.isin(pairs[0]*n_drug+pairs[1], known).any():
            raise ValueError('CSV contains known fitting positives; provide unknown candidate pairs')
    if pairs.shape[1] == 0:
        raise ValueError('No unknown candidate pairs remain')
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment', default='improve/results/graph_tabm_20260910')
    parser.add_argument('--task', required=True, choices=graph_runner.TASKS)
    parser.add_argument('--arm', choices=('base', 'pencil', 'mlp', 'tabm'))
    parser.add_argument('--seed', type=int, default=666)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--pairs')
    source.add_argument('--drug-index', type=int)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = Path(args.out)
    if out.exists() or out.with_suffix(out.suffix+'.json').exists():
        raise ValueError('Output already exists; choose a new filename')
    experiment = Path(args.experiment)
    raw = json.loads((experiment/'protocol.json').read_text())
    table = 'network' in raw
    runner = table_runner if table else graph_runner
    p = runner.protocol(experiment)
    validation = json.loads((experiment/'VALIDATION.json').read_text())
    if p['smoke'] or not validation['passed'] or not validation['all_checkpoints_replayed']:
        raise ValueError('Prediction requires a verified full experiment')
    if validation['protocol_sha256'] != p['sha256']:
        raise ValueError('Verification belongs to another experiment')
    arm = args.arm or ('tabm' if table else 'pencil')
    if arm not in p['arms'] or args.seed not in p['refit_seeds']:
        raise ValueError('Requested arm/seed is not an exported refit')
    chosen = graph_runner.decision(experiment, 'REFIT.json', p)['chosen'][args.task][arm]
    cfg = chosen['config']
    dest = graph_runner.directory(experiment, 'outer', args.task, arm, cfg['id'], args.seed)
    result = graph_runner.checked_result(dest, p)
    reference = p['reference_dir'] if table else experiment
    arrays, data, weights = graph_runner.stage_inputs(reference, args.task, 'outer')
    pairs = read_pairs(args, arrays, data, graph_runner.TASKS[args.task]['rna'])
    if table:
        cache = table_runner.features(experiment, p, args.task, 'outer')
        if cache['feature_manifest_sha256'] != result['feature_manifest_sha256']:
            raise ValueError('Checkpoint preprocessing does not match')
        path = experiment/'features'/'outer'/args.task
        manifest = json.loads((path/'manifest.json').read_text())
        with (path/'transform.pkl').open('rb') as stream:
            transform = pickle.load(stream)
        encoder = PairFeatureEncoder(data, args.task)
        frame = encoder.transform(pairs, arrays['pairs_train'][:, arrays['y_train']==1])
        if list(frame.columns) != manifest['names']:
            raise ValueError('Feature schema mismatch')
        x = torch.from_numpy(transform.transform(frame.values).astype(np.float32)).to(args.device)
        model = table_runner.model_for(arm, x.shape[1], cfg, args.seed, args.device)
        model.load_state_dict(torch.load(dest/'checkpoint.pt', map_location='cpu', weights_only=False))
        probability = table_runner.predict(model, x, p['batch_size'])
    else:
        model, data, weights = graph_runner.make_model(data, weights, args.task, arm, cfg, args.seed, args.device)
        model.load_state_dict(torch.load(dest/'checkpoint.pt', map_location='cpu', weights_only=False))
        context = SupportGraph(data.x_dict, data.edge_index_dict, weights) if arm=='pencil' else None
        sampled = context.sample(pairs, graph_runner.TASKS[args.task]['rna'], cfg['budget'], p['evaluation_sampling_seed']) if context else None
        logits = graph_runner.predict(model, data, weights, pairs, context, sampled, p['batch_size'])
        probability = torch.sigmoid(torch.from_numpy(logits)).numpy()
    frame = pd.DataFrame(dict(rna_index=pairs[0], drug_index=pairs[1], association_score=probability))
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    graph_runner.write_json(out.with_suffix(out.suffix+'.json'), dict(task=args.task, arm=arm,
        seed=args.seed, experiment=str(experiment.resolve()), protocol_sha256=p['sha256'],
        checkpoint_sha256=result['artifacts_sha256']['checkpoint.pt'], rows=len(frame),
        score_definition='Sigmoid association score; not a calibrated biological resistance probability'))
    print('WROTE', out, len(frame), 'candidate scores', flush=True)


if __name__ == '__main__':
    main()
