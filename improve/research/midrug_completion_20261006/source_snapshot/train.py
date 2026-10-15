"""Fit once to convergence; only then evaluate the internal held-out pairs."""
import argparse
import json
import time
import numpy as np
from scipy.special import expit, logit
from sklearn.metrics import roc_auc_score, average_precision_score
from common import HERE, load_inner, make_graph, sha256
from model import CONFIGS, laplacian, fit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', choices=CONFIGS, required=True)
    parser.add_argument('--fold', type=int, required=True)
    args = parser.parse_args()
    out = HERE/'results'
    tag = f'{args.config}_f{args.fold}'
    assert not (out/f'{tag}.json').exists()
    plan = json.loads((out/'plan.json').read_text())
    for path, h in plan['source_hashes'].items():
        assert sha256(path) == h
    data = load_inner(args.fold)
    graph = make_graph(data)
    lr = laplacian(graph['miRNA'].num_nodes, graph['miRNA','MiMi','miRNA'].edge_index.numpy())
    ld = laplacian(graph['drug'].num_nodes, graph['drug','DrugDrug','drug'].edge_index.numpy())
    start = time.time()
    def callback(record):
        if record['iteration'] % 250 == 0:
            print(tag, json.dumps(record), flush=True)
    x, fit_info = fit(data['train_pairs'], data['train_y'].astype(float), lr, ld, CONFIGS[args.config],
                      max_steps=plan['max_steps'], tolerance=plan['stationarity_tolerance'], callback=callback)
    # Align the constant-score prior for blending; this is not a calibration claim.
    score = x+fit_info['nominal_intercept']
    r,d = data['val_pairs'].T
    heldout_score = score[r,d]
    np.savez_compressed(out/f'{tag}.npz', residual=x, intercept=fit_info['intercept'],
                        nominal_intercept=fit_info['nominal_intercept'])
    np.savez_compressed(out/f'{tag}_inner.npz', pairs=data['val_pairs'], y=data['val_y'],
                        indices=data['val_indices'], score=heldout_score, probability=expit(heldout_score))
    result = dict(tag=tag, config=args.config, settings=CONFIGS[args.config], fold=args.fold,
        inner_auc=float(roc_auc_score(data['val_y'], heldout_score)),
        inner_ap=float(average_precision_score(data['val_y'], heldout_score)),
        checkpoint_sha256=sha256(out/f'{tag}.npz'), prediction_sha256=sha256(out/f'{tag}_inner.npz'),
        elapsed_seconds=time.time()-start, fit_info=fit_info,
        no_lncrna=True, outer_validation_evaluated=False, test_evaluated=False,
        training_stopping='training objective and stationarity only; no heldout inputs to fit')
    (out/f'{tag}.json').write_text(json.dumps(result, indent=2))
    print('DONE', tag, result['inner_auc'], result['inner_ap'], fit_info['converged'], flush=True)


if __name__ == '__main__':
    main()
