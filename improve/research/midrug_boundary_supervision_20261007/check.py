"""Independent NumPy geometry/loss/gradient oracle and real M123 checks."""
import json
import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from common import HERE, DEGREE, read_module, load_inner, seed_all, sha256
from model import CONFIGS, make_adapter, make_optimizer, mask_generator, training_context, prototype_loss, association_loss

QUERY_CHECK_SOURCE = DEGREE/'check.py'
_query = read_module('boundary_supervision_query_checks', QUERY_CHECK_SOURCE)
_query.make_adapter = make_adapter
deletion_check = _query.deletion_check


def numpy_oracle(z, y, cfg, context):
    x, labels = z.detach().numpy(), y.numpy().astype(int)
    mean, scale = x.mean(0), np.maximum(x.std(0), cfg['std_floor'])
    u = (x-mean)/scale
    cov = np.cov(u, rowvar=False, ddof=1)
    ridge = cfg['ridge_fraction']*max(np.diag(cov).mean(), 1e-8)
    cov = cov+ridge*np.eye(x.shape[1])
    c0, c1 = u[labels == 0].mean(0), u[labels == 1].mean(0)
    direction = np.linalg.solve(cov, c1-c0)
    margin = np.abs((u-(c0+c1)/2)@direction)
    indices = []
    for c in [0, 1]:
        pool = np.flatnonzero(labels == c)
        k = math.ceil(cfg['fraction']*len(pool))
        indices.extend(pool[np.argsort(margin[pool], kind='stable')[:k]])
    np.testing.assert_array_equal(indices, context['boundary'].numpy())
    np.testing.assert_allclose(margin, context['margin'].numpy(), atol=2e-10, rtol=2e-10)
    unit = u/np.linalg.norm(u, axis=1, keepdims=True).clip(1e-12)
    centers = np.stack([unit[labels == c].mean(0) for c in [0, 1]])
    centers /= np.linalg.norm(centers, axis=1, keepdims=True).clip(1e-12)
    np.testing.assert_allclose(centers, context['prototypes'].numpy(), atol=1e-12, rtol=1e-12)
    chosen = context['selected'].numpy()
    logits = unit[chosen]@centers.T/cfg['temperature']
    logits -= logits.max(1, keepdims=True)
    probability = np.exp(logits)/np.exp(logits).sum(1, keepdims=True)
    loss = -np.log(probability[np.arange(len(chosen)), labels[chosen]]).mean()
    residual = probability.copy()
    residual[np.arange(len(chosen)), labels[chosen]] -= 1
    gv = residual@centers/(cfg['temperature']*len(chosen))
    v = unit[chosen]
    gu = (gv-v*(gv*v).sum(1, keepdims=True))/np.linalg.norm(u[chosen], axis=1, keepdims=True).clip(1e-12)
    gradient = np.zeros_like(x)
    gradient[chosen] = gu/scale
    return loss, gradient


def toy_checks():
    seed_all(17)
    z = torch.randn(19, 7, dtype=torch.double, requires_grad=True)
    y = torch.tensor([0]*12+[1]*7, dtype=torch.double)
    records = []
    for name in ['boundary', 'random']:
        cfg = CONFIGS[name]
        ctx = training_context(z, y, cfg, mask_generator('cpu', 88))
        loss = prototype_loss(z, y, ctx, cfg['temperature'])
        gradient = torch.autograd.grad(loss, z)[0]
        expected, expected_gradient = numpy_oracle(z, y, cfg, ctx)
        assert abs(float(loss)-expected) < 1e-12
        np.testing.assert_allclose(gradient.numpy(), expected_gradient, atol=2e-11, rtol=2e-10)
        assert torch.autograd.gradcheck(lambda value: prototype_loss(value, y, ctx, cfg['temperature']), (z,), eps=1e-6, atol=1e-6, rtol=1e-4)
        outside = torch.ones(len(z), dtype=torch.bool)
        outside[ctx['selected']] = False
        assert torch.count_nonzero(gradient[outside]) == 0
        assert gradient[~outside].abs().max() > 0
        assert ctx['counts'] == [3, 2]
        assert len(ctx['selected'].unique()) == 5
        # Translation and positive coordinate rescaling above the std floor.
        scaled = z.detach()*torch.linspace(.5, 2., 7)+torch.linspace(-1., 2., 7)
        alternate = training_context(scaled, y, cfg, mask_generator('cpu', 88))
        torch.testing.assert_close(ctx['selected'], alternate['selected'])
        torch.testing.assert_close(ctx['margin'], alternate['margin'], atol=2e-10, rtol=2e-10)
        other_loss = prototype_loss(scaled, y, alternate, cfg['temperature'])
        assert abs(float(other_loss)-float(loss)) < 1e-10
        records.append(dict(config=name,loss_error=abs(float(loss)-expected),
                            gradient_error=float(abs(gradient.numpy()-expected_gradient).max()),
                            finite_difference_passed=True, outside_anchor_gradient_zero=True))
    # Constant/collinear features leave finite geometry and finite loss.
    flat = torch.ones(19, 7, dtype=torch.double, requires_grad=True)
    ctx = training_context(flat, y, CONFIGS['boundary'], mask_generator('cpu', 88))
    loss = prototype_loss(flat, y, ctx, .2)
    assert torch.isfinite(loss)
    assert torch.isfinite(torch.autograd.grad(loss, flat)[0]).all()
    return records


def main():
    toy = toy_checks()
    data = load_inner(0)
    records = []
    for name, cfg in CONFIGS.items():
        seed_all(42)
        plain = make_adapter(data, 'unmasked10' if cfg['reconstruction_weight'] else 'control', 'cuda:0')
        seed_all(42)
        a = make_adapter(data, name, 'cuda:0')
        for key, value in plain.model.state_dict().items():
            torch.testing.assert_close(value, a.model.state_dict()[key], atol=0, rtol=0)
        plain.model.eval(); a.model.eval()
        with torch.no_grad():
            p0, p1 = plain.predict('val'), a.predict('val')
        np.testing.assert_allclose(p0, p1, atol=2e-5, rtol=2e-5)
        # During the first30 epochs the loss and feature gradients are ordinary BCE.
        a.model.train()
        total, stats, info = association_loss(a, cfg, 30, mask_generator('cuda:0', 99))
        expected = F.binary_cross_entropy_with_logits(info['logits'], a.y)
        assert stats['auxiliary_weight'] == 0 and info['context'] is None
        torch.testing.assert_close(total, expected, atol=0, rtol=0)
        g1 = torch.autograd.grad(total, info['features'], retain_graph=True)[0]
        g2 = torch.autograd.grad(expected, info['features'])[0]
        torch.testing.assert_close(g1, g2, atol=0, rtol=0)
        before = int(a.model.eb.num_batches_tracked)
        opt = make_optimizer(a.model); opt.zero_grad()
        torch.cuda.reset_peak_memory_stats(); start = time.time()
        total, stats, info = association_loss(a, cfg, 60, mask_generator('cuda:0', 20261007))
        assert int(a.model.eb.num_batches_tracked) == before+1
        ctx = info['context']
        auxiliary_gradient = torch.autograd.grad(info['auxiliary'], info['features'], retain_graph=True)[0]
        mask = torch.zeros(len(a.y), dtype=torch.bool, device=a.y.device); mask[ctx['selected']] = True
        assert torch.count_nonzero(auxiliary_gradient[~mask]) == 0
        assert auxiliary_gradient[mask].abs().max() > 0 and torch.isfinite(auxiliary_gradient).all()
        if cfg['reconstruction_weight']:
            ids = a.model.reconstruction.masks(a.graph, .5, mask_generator('cuda:0', 20261006))
            reconstruction, _ = a.model.reconstruction.loss(a.model.encoder, a.graph, ids, False)
            total = total+10*reconstruction
        total.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.model.parameters())
        opt.step(); torch.cuda.synchronize()
        elapsed = time.time()-start; peak = torch.cuda.max_memory_allocated()/2**30
        assert not a.model.lin3._forward_pre_hooks and set(a.graph.node_types) == {'miRNA', 'drug'}
        assert all('lnc' not in key.lower() for key in a.model.state_dict())
        records.append(dict(config=name,initial_prediction_error=float(abs(p0-p1).max()),
                            zero_weight_original_loss_and_gradient=True, auxiliary_anchor_gradients_checked=True,
                            selected_counts=ctx['counts'], training_step_seconds=elapsed, peak_cuda_gib=peak,
                            explicit_deletions=deletion_check(a,data,name,count=4)))
        print('PASS',name,elapsed,peak,flush=True)
        del a, plain, info, total, g1, g2; torch.cuda.empty_cache()
    seed_all(42); a = make_adapter(data, 'boundary', 'cuda:0')
    seed_all(42); b = make_adapter({**data,'val_y':1-data['val_y']}, 'boundary', 'cuda:0')
    a.model.train(); b.model.train()
    la, sa, ia = association_loss(a, CONFIGS['boundary'], 60, mask_generator('cuda:0', 77))
    lb, sb, ib = association_loss(b, CONFIGS['boundary'], 60, mask_generator('cuda:0', 77))
    torch.testing.assert_close(la, lb, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(ia['context']['selected'], ib['context']['selected'])
    torch.testing.assert_close(ia['context']['prototypes'], ib['context']['prototypes'], atol=2e-5, rtol=2e-5)
    result = dict(status='passed',toy=toy,records=records,heldout_label_invariance=True,
                  query_check_source_sha256=sha256(QUERY_CHECK_SOURCE),no_lncrna=True,
                  outer_validation_evaluated=False,test_evaluated=False)
    (HERE/'initial_checks.json').write_text(json.dumps(result,indent=2))
    print('ALL PASSED',flush=True)


if __name__ == '__main__':
    main()
