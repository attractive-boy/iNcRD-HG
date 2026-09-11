"""Four fitting-only optimizer steps per task; no validation/test scores.

This verifies execution, gradient flow and support boundaries. It does NOT
screen the performance of the proposed architecture.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from improve import pair_exchange as parent
from improve.expression_scale_mechanism import state_digest
from improve.first_round_features import sha256
from improve.pair_exchange_modules import remove_target_pairs, sample_queries
from improve.support_operator_modules import SupportOperatorDecoder


def grad_norm(model, prefix):
    gradients = [p.grad for n, p in model.named_parameters() if n.startswith(prefix) and p.grad is not None]
    if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
        raise AssertionError('Missing/nonfinite gradients: ' + prefix)
    result = sum(float(g.detach().square().sum()) for g in gradients) ** .5
    if result <= 0:
        raise AssertionError('Zero gradient: ' + prefix)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--out', default=str(Path(__file__).with_name('prototype_probe.json')))
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError('Preserve prior probe: ' + str(out))
    torch.set_num_threads(4)
    reference = Path('improve/results/dual_pair_tuning_20260910')
    protocol = json.loads((reference / 'protocol.json').read_text())
    result = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                  scope='Engineering probe only: four optimizer steps per task on fitting queries; no performance inference',
                  seed=666, rank=16, device=args.device, torch_version=torch.__version__,
                  inner_holdout_evaluations=0, outer_evaluations=0, test_evaluations=0,
                  reference_protocol_sha256=protocol['sha256'],
                  source_sha256=parent.own_sources(), tasks={})
    for filename in ('support_operator_modules.py', 'test_support_operator.py'):
        result['source_sha256'][filename] = sha256(Path('improve') / filename)
    result['probe_source_sha256'] = sha256(Path(__file__))
    for task in protocol['tasks']:
        start = time.monotonic()
        input_files = {name: sha256(reference / 'inputs' / task / name)
                       for name in ('inner.npz', 'train_graph.pth')}
        arrays, graph = parent.load_stage(reference, protocol, task, 'inner')
        model, data, pairs, weights = parent.make_model(arrays, graph, task, 'pca', 666, args.device)
        config = protocol['training'][task]
        rna = config['rna']
        model.decoder = SupportOperatorDecoder(config['channels'], rna, rank=16).to(args.device)
        initial = state_digest(model)
        # Target labels enter only the explicit support solver. The original
        # other-task associations remain frozen and are not merged across tasks.
        positive = arrays['pairs_train'][:, arrays['y_train'] == 1]
        edges, feature_weights = remove_target_pairs(data.edge_index_dict, weights, task, rna,
                                                      data['drug'].num_nodes, positive)
        key, reverse = (rna, task, 'drug'), ('drug', 'rev_' + task, rna)
        assert edges[key].numel() == edges[reverse].numel() == 0
        labels = torch.from_numpy(arrays['y_train']).to(args.device)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=config['lr'], alpha=.99)
        trace = []
        for step in range(1, 5):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            ids = sample_queries(arrays['y_train'], .5, 667 * 1000003 + step)
            query = pairs['train'][:, ids]
            z = model.encoder(data.x_dict, edges, feature_weights)
            logits, detail = model.decoder(z, query, pairs['train'], labels, return_details=True)
            assert not set(map(tuple, detail['support_pairs'].T.cpu().numpy())).intersection(
                map(tuple, query.T.cpu().numpy()))
            loss = F.binary_cross_entropy_with_logits(logits, labels[ids])
            assert bool(torch.isfinite(loss))
            loss.backward()
            gradients = {prefix: grad_norm(model, prefix) for prefix in
                         ('encoder', 'decoder.rna_basis', 'decoder.drug_prior',
                          'decoder.drug_key', 'decoder.ridge_head', 'decoder.raw_temperature')}
            optimizer.step()
            trace.append(dict(step=step, fitting_query_rows=len(ids),
                              support_rows=detail['support_pairs'].shape[1],
                              training_bce=float(loss.detach()), gradient_norms=gradients))
        model.eval()
        with torch.no_grad():
            z = model.encoder(data.x_dict, edges, feature_weights)
            base, details = model.decoder(z, query, pairs['train'], labels, return_details=True)
            flipped = model.decoder(z, query, pairs['train'], 1 - labels)
            prior = model.decoder(z, query, pairs['train'], labels, mode='prior')
            prior_flipped = model.decoder(z, query, pairs['train'], 1 - labels, mode='prior')
            empty = model.decoder(z, query, pairs['train'][:, :0], labels[:0])
            torch.testing.assert_close(prior, prior_flipped, rtol=0, atol=0)
            torch.testing.assert_close(prior, empty, rtol=0, atol=0)
            effect = float((base - flipped).abs().mean())
            assert effect > 0
        for name, digest in input_files.items():
            assert sha256(reference / 'inputs' / task / name) == digest
        report = dict(fitting_rows=len(labels), inner_holdout_rows_not_scored=len(arrays['y_val']),
                      target_encoder_edges=0, reverse_target_encoder_edges=0,
                      fitting_query_support_overlap=0, steps=trace,
                      finite_nonzero_gradients=True, empty_support_equals_prior=True,
                      prior_invariant_to_memory_labels=True,
                      support_label_flip_mean_absolute_logit_change=effect,
                      operator_prior_mean_absolute_difference=float((details['operator'] - details['prior']).abs().mean()),
                      initial_state_sha256=initial, final_state_sha256=state_digest(model),
                      input_sha256=input_files, elapsed_seconds=time.monotonic() - start)
        result['tasks'][task] = report
        print(task, json.dumps(report, ensure_ascii=False), flush=True)
    result['passed'] = True
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
