"""Audit saved copula posterior numerics without loading evaluation outcomes."""
from datetime import datetime, timezone
from pathlib import Path
import gc

import numpy as np
import torch

from improve import anchor_copula_experiment_v1 as experiment
from improve import bernoulli_copula_graph_v1 as core
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
OUT = experiment.ROOT / 'results/anchor_residual_bernoulli_copula_v1_20260916'


def freeze():
    path = HERE / 'COPULA_POSTERIOR_AUDIT_PROTOCOL_V1.json'
    if path.exists():
        raise ValueError('Preserve the previous audit protocol')
    base.write_json(path, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=base.digest(__file__), experiment_protocol_sha256=base.digest(OUT / 'protocol.json'),
        criterion='Frozen eight-step posterior vs 32-step numerical reference, conditional residual maximum absolute difference <= 3e-5. No model parameter changes or evaluation outcomes.',
        maximum_residual_difference=3e-5, posterior_steps_reference=32,
        scope='All completed full-arm fits, including unselected screening configurations.',
        numerical_audit_not_additional_performance_selection=True)))


@torch.no_grad()
def main():
    path = HERE / 'COPULA_POSTERIOR_AUDIT_V1.json'
    if path.exists():
        raise ValueError('Preserve existing posterior audit')
    protocol = base.check_seal(base.read_json(HERE / 'COPULA_POSTERIOR_AUDIT_PROTOCOL_V1.json'))
    if protocol['source_sha256'] != base.digest(__file__):
        raise ValueError('Numerical audit source changed')
    if protocol['experiment_protocol_sha256'] != base.digest(OUT / 'protocol.json'):
        raise ValueError('Copula experiment changed')
    queue = base.check_seal(base.read_json(HERE / 'COPULA_QUEUE_RESULT_V1.json'))
    p = experiment.common.protocol(OUT)
    torch.set_num_threads(4)
    rows, inputs = [], {}
    with experiment.precision.deterministic_evaluation():
        for task in p['tasks']:
            source = OUT / 'inputs' / task / 'arrays.npz'
            a = experiment.nested.load_selected(source, experiment.TRAIN_KEYS)
            identities = experiment.nested.load_selected(source, ('pairs_val',))['pairs_val']
            pairs = torch.as_tensor(identities + a['pair_offset'][:, None])
            for result_path in sorted((OUT / 'runs' / task / 'full').rglob('result.json')):
                record = base.read_json(result_path)
                if record['protocol_sha256'] != base.digest(OUT / 'protocol.json'):
                    raise ValueError('Fit protocol mismatch')
                model_path = result_path.parent / 'model.pt'
                if base.digest(model_path) != record['files']['model.pt']:
                    raise ValueError('Saved model changed')
                model, xs, graphs, _ = experiment.setup(a, p, 'full', 'cpu')
                model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
                model.eval()
                query_logits = torch.as_tensor(experiment.nested.load_selected(source,
                    ('anchor_' + str(record['seed']),))['anchor_' + str(record['seed'])])
                for fold, graph in enumerate(graphs):
                    h = model.encode(xs, graph)
                    residual, factor, diagonal = model.pair_parameters(h, graph.support_pairs)
                    threshold = core.latent_threshold(graph.support_logits + residual)
                    m8, s8, detail = core.laplace_posterior(factor, diagonal, threshold,
                        graph.support_labels, p['core_parameters']['posterior_steps'])
                    m32, s32, longer = core.laplace_posterior(factor, diagonal, threshold,
                        graph.support_labels, protocol['posterior_steps_reference'])
                    residual_q, pq, kq = model.pair_parameters(h, pairs)
                    bq = core.latent_threshold(query_logits + residual_q)
                    delta8 = core.posterior_logit_shift(pq, kq, bq, m8, s8)
                    delta32 = core.posterior_logit_shift(pq, kq, bq, m32, s32)
                    rows.append(dict(task=task, lr=record['lr'], seed=record['seed'], fold=fold,
                        selected_step=record['epoch'], posterior_gradient_max=float(detail['gradient_max']),
                        long_posterior_gradient_max=float(longer['gradient_max']),
                        residual_max_abs_difference=float((delta8-delta32).abs().max()),
                        diagonal_min=float(diagonal.min())))
                inputs[str(result_path)] = base.digest(result_path)
                print('AUDITED_COPULA_POSTERIOR', task, record['lr'], record['seed'], flush=True)
                del model, xs, graphs
                gc.collect()
    maximum = max(row['residual_max_abs_difference'] for row in rows)
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        audit_protocol_sha256=protocol['sha256'], queue_result_sha256=queue['sha256'],
        result_sha256=inputs, rows=rows, maximum_residual_difference=maximum,
        passed=bool(maximum <= protocol['maximum_residual_difference']),
        evaluation_labels_read=False, parameters_changed=False, fits_repeated=0,
        cpu_inference_only=True, confirmed_final_innovations=0))
    base.write_json(path, result)
    print('COPULA_POSTERIOR_AUDIT', result['passed'], maximum, flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('freeze', 'run'))
    args = parser.parse_args()
    freeze() if args.command == 'freeze' else main()
