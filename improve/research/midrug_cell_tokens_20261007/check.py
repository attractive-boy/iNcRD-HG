"""Single-cell-loop oracle, invariances, full training and original graph deletion."""
import json
import math
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import HERE, DEGREE, read_module, load_inner, seed_all, sha256
from model import CONFIGS, CellTokens, make_adapter, make_optimizer, mask_generator

QUERY_CHECK_SOURCE = DEGREE/'check.py'
_query = read_module('cell_tokens_full_query_check', QUERY_CHECK_SOURCE)
_query.make_adapter = make_adapter
deletion_check = _query.deletion_check


def sequential_reference(layers, value):
    for layer in layers:
        if isinstance(layer, nn.Linear):
            value = F.linear(value, layer.weight, layer.bias)
        elif isinstance(layer, nn.LayerNorm):
            value = F.layer_norm(value, layer.normalized_shape, layer.weight, layer.bias, layer.eps)
        elif isinstance(layer, nn.LeakyReLU):
            value = torch.where(value >= 0, value, layer.negative_slope*value)
        else:
            raise TypeError(type(layer))
    return value


def independent_pair(m, xr, xd, r, d):
    query = sequential_reference(m.query, xd[d])
    if not m.conditioned:
        query = torch.stack([sequential_reference(m.query, row) for row in xd]).mean(0)
    scores, values = [], []
    for j in range(xr.shape[1]):
        cell = sequential_reference(m.cell_encoder, xr[:, j])
        token = cell+F.linear(xr[r, j:j+1], m.expression.weight)
        token = torch.where(token >= 0, token, .1*token)
        key = F.linear(token, m.key.weight)
        values.append(F.linear(token, m.value.weight))
        scores.append((key*query).sum()/math.sqrt(m.hidden))
    alpha = torch.stack(scores).softmax(0)
    pooled = sum(alpha[j]*values[j] for j in range(len(values)))
    return sequential_reference(m.readout, pooled)


def toy_checks():
    rows = []
    for conditioned in [True, False]:
        seed_all(41)
        m = CellTokens(5, 6, hidden=4, conditioned=conditioned, chunk_size=2).double()
        xr, xd = torch.randn(5, 7, dtype=torch.double), torch.randn(4, 6, dtype=torch.double)
        r, d = torch.tensor([0, 2, 4]), torch.tensor([1, 3, 0])
        z = m(xr, xd, r, d)
        ref = torch.stack([independent_pair(m, xr, xd, int(i), int(j)) for i, j in zip(r, d)])
        torch.testing.assert_close(z, ref, atol=1e-10, rtol=1e-10)
        coefficients = torch.linspace(-.8, .7, z.numel(), dtype=torch.double).view_as(z)
        params = list(m.parameters())
        g1 = torch.autograd.grad((z*coefficients).sum(), params)
        g2 = torch.autograd.grad((ref*coefficients).sum(), params)
        errors = []
        for a, b in zip(g1, g2):
            torch.testing.assert_close(a, b, atol=2e-9, rtol=2e-9)
            errors.append(float(abs(a-b).max()))
        permutation = torch.tensor([4, 1, 6, 0, 3, 5, 2])
        changed = m(xr[:, permutation], xd, r, d)
        torch.testing.assert_close(z, changed, atol=1e-10, rtol=1e-10)
        gp = torch.autograd.grad((changed*coefficients).sum(), params)
        for a, b in zip(g1, gp):
            torch.testing.assert_close(a, b, atol=2e-9, rtol=2e-9)
        m.chunk_size = 1
        split = m(xr, xd, r, d)
        torch.testing.assert_close(z, split, atol=1e-10, rtol=1e-10)
        keys, values, queries = m.encode(xr, xd)
        _, attention = m.attend(keys, values, queries, r, d)
        torch.testing.assert_close(attention.sum(-1), torch.ones(3, dtype=torch.double), atol=1e-12, rtol=1e-12)
        rows.append(dict(conditioned=conditioned, output_error=float(abs(z-ref).max()),
                         parameter_gradient_error=max(errors), column_permutation_error=float(abs(z-changed).max()),
                         chunk_error=float(abs(z-split).max())))
    return rows


@torch.no_grad()
def real_branch_check(a):
    m = a.model.cell_context
    r, d = a.pairs['val'][:, :3]
    actual = m(a.dense[0], a.dense[1], r, d)
    expected = torch.stack([independent_pair(m, a.dense[0], a.dense[1], int(i), int(j)) for i, j in zip(r, d)])
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=5e-5)
    permutation = torch.arange(a.dense[0].shape[1]-1, -1, -1, device=r.device)
    permuted = m(a.dense[0][:, permutation], a.dense[1], r, d)
    torch.testing.assert_close(actual, permuted, atol=2e-4, rtol=5e-5)
    chunk = m.chunk_size
    try:
        m.chunk_size = 1
        split = m(a.dense[0], a.dense[1], r, d)
    finally:
        m.chunk_size = chunk
    torch.testing.assert_close(actual, split, atol=2e-4, rtol=5e-5)
    return dict(oracle_error=float(abs(actual-expected).max()),
                column_permutation_error=float(abs(actual-permuted).max()), chunk_error=float(abs(actual-split).max()))


def main():
    toy = toy_checks()
    data = load_inner(0); records = []
    for name, cfg in CONFIGS.items():
        seed_all(42); plain = make_adapter(data, 'unmasked10' if cfg['reconstruction_weight'] else 'control', 'cuda:0')
        seed_all(42); a = make_adapter(data, name, 'cuda:0'); m = a.model
        for key, value in plain.model.state_dict().items():
            torch.testing.assert_close(value, m.state_dict()[key], atol=0, rtol=0)
        m.eval(); plain.model.eval()
        with torch.no_grad():
            p0, p1 = plain.predict('val'), a.predict('val')
            m.cell_projection.weight.normal_(std=.005)
        np.testing.assert_allclose(p0, p1, atol=2e-5, rtol=2e-5)
        m.train(); opt = make_optimizer(m, cfg); opt.zero_grad()
        torch.cuda.reset_peak_memory_stats(); start = time.time()
        loss = F.binary_cross_entropy_with_logits(a.logits('train'), a.y)
        if cfg['reconstruction_weight']:
            indices = m.reconstruction.masks(a.graph, .5, mask_generator('cuda:0', 20261006))
            reconstruction, _ = m.reconstruction.loss(m.encoder, a.graph, indices, False)
            loss = loss+10*reconstruction
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        group_gradients = {}
        for group in ['cell_encoder', 'expression', 'key', 'value', 'query', 'readout']:
            norm = sum(float(p.grad.square().sum()) for p in getattr(m.cell_context, group).parameters() if p.grad is not None)**.5
            assert norm > 0
            group_gradients[group] = norm
        opt.step(); torch.cuda.synchronize()
        elapsed = time.time()-start; peak = torch.cuda.max_memory_allocated()/2**30
        assert int(m.eb.num_batches_tracked) == 1 and not m.lin1._forward_hooks
        assert set(a.graph.node_types) == {'miRNA', 'drug'} and all('lnc' not in key.lower() for key in m.state_dict())
        m.eval()
        records.append(dict(config=name, initial_prediction_error=float(abs(p0-p1).max()),
                            finite_nonzero_branch_group_gradients=group_gradients,
                            training_step_seconds=elapsed, peak_cuda_gib=peak,
                            branch=real_branch_check(a), explicit_deletions=deletion_check(a,data,name,count=4)))
        print('PASS', name, elapsed, peak, flush=True)
        del a, m, plain, opt, loss; torch.cuda.empty_cache()
    seed_all(42); a = make_adapter(data, 'drug_rec10', 'cuda:0')
    seed_all(42); b = make_adapter(data, 'mean_rec10', 'cuda:0')
    for key, value in a.model.state_dict().items():
        torch.testing.assert_close(value, b.model.state_dict()[key], atol=0, rtol=0)
    seed_all(42); b = make_adapter({**data, 'val_y':1-data['val_y']}, 'drug_rec10', 'cuda:0')
    with torch.no_grad():
        a.model.cell_projection.weight.normal_(std=.005)
        b.model.load_state_dict(a.model.state_dict())
    a.model.eval(); b.model.eval()
    with torch.no_grad():
        pa, pb = a.predict('val'), b.predict('val')
    np.testing.assert_allclose(pa, pb, atol=2e-5, rtol=2e-5)
    result = dict(status='passed', toy=toy, records=records, matched_initial_parameters_equal=True,
                  heldout_label_invariance=True, heldout_prediction_error=float(abs(pa-pb).max()),
                  query_check_source_sha256=sha256(QUERY_CHECK_SOURCE), no_lncrna=True,
                  outer_validation_evaluated=False, test_evaluated=False)
    (HERE/'initial_checks.json').write_text(json.dumps(result, indent=2))
    print('ALL PASSED', flush=True)


if __name__ == '__main__':
    main()
