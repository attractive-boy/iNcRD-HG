"""Mean oracle, singleton identity, and noninterference with the training state."""
import copy
import json
import numpy as np
import torch
from common import HERE, DEGREE, read_module, load_inner, seed_all, sha256
from model import make_adapter, make_optimizer, ParameterMean, clone_state, predict_mean, step, START_EPOCH

QUERY_CHECK_SOURCE = DEGREE / 'check.py'
query = read_module('weight_average_deletions', QUERY_CHECK_SOURCE); query.make_adapter = make_adapter
deletion_check = query.deletion_check


def equal_nested(a, b):
    if torch.is_tensor(a): torch.testing.assert_close(a, b, atol=0, rtol=0)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a: equal_nested(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b): equal_nested(x, y)
    else: assert a == b


def main():
    seed_all(42)
    toy = torch.nn.Sequential(torch.nn.Linear(5, 3), torch.nn.BatchNorm1d(3)).double()
    average = ParameterMean(); history = []
    for ep in range(1, START_EPOCH): average.update(toy, ep)
    assert average.count == 0 and not average.sums
    for i in range(7):
        with torch.no_grad():
            for p in toy.parameters(): p.copy_(torch.randn_like(p))
            toy[1].running_mean.fill_(i); toy[1].num_batches_tracked.fill_(i + 100)
        history.append({n: p.detach().clone() for n, p in toy.named_parameters()})
        average.update(toy, START_EPOCH + i); actual = average.state(toy)
        for n, p in toy.named_parameters():
            torch.testing.assert_close(actual[n], torch.stack([s[n] for s in history]).mean(0), atol=1e-14, rtol=1e-14)
        assert int(actual['1.num_batches_tracked']) == i + 100
        torch.testing.assert_close(actual['1.running_mean'], toy[1].running_mean, atol=0, rtol=0)
    data = load_inner(0); seed_all(42); a = make_adapter(data, 'raw', 'cuda:0')
    optimizer = make_optimizer(a.model); average = ParameterMean()
    generator = torch.Generator(device='cuda:0').manual_seed(20261006); records = []
    for ep in range(1, 102):
        losses = step(a, optimizer, ep, generator); average.update(a.model, ep)
        if ep < START_EPOCH: continue
        before = clone_state(a.model); grad = {n: None if p.grad is None else p.grad.detach().clone() for n, p in a.model.named_parameters()}
        opt = copy.deepcopy(optimizer.state_dict()); rng = generator.get_state().clone()
        with torch.no_grad(): raw = a.predict('val')
        predicted = predict_mean(a, average)
        equal_nested(before, a.model.state_dict()); equal_nested(opt, optimizer.state_dict())
        equal_nested(grad, {n: p.grad for n, p in a.model.named_parameters()}); equal_nested(rng, generator.get_state())
        if ep == START_EPOCH:
            equal_nested(before, average.state(a.model))
            np.testing.assert_allclose(raw, predicted, atol=2e-5, rtol=2e-5)
        records.append(dict(epoch=ep, count=average.count, training_state_unchanged=True, max_raw_mean_difference=float(abs(raw - predicted).max())))
    averaged = average.state(a.model); a.model.load_state_dict(averaged)
    deleted = deletion_check(a, data, 'mean', count=4)
    seed_all(42); flipped = make_adapter({**data, 'val_y': 1-data['val_y']}, 'mean', 'cuda:0')
    flipped.model.load_state_dict(averaged); flipped.model.eval(); a.model.eval()
    with torch.no_grad(): p, q = a.predict('val'), flipped.predict('val')
    np.testing.assert_allclose(p, q, atol=2e-5, rtol=2e-5)
    assert int(a.model.eb.num_batches_tracked) == 101
    (HERE / 'initial_checks.json').write_text(json.dumps(dict(status='passed', mean_oracle_snapshots=7,
        records=records, explicit_deletions=deleted, heldout_label_flip_error=float(abs(p - q).max()),
        query_check_source_sha256=sha256(QUERY_CHECK_SOURCE), no_lncrna=True, outer_validation_evaluated=False, test_evaluated=False), indent=2))
    print('ALL PASSED', flush=True)


if __name__ == '__main__': main()
