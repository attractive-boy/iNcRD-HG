"""One original reconstruction trajectory, with a fixed arithmetic parameter mean."""
import torch
from common import Adapter, make_graph, Reconstruction

START_EPOCH = 100
CONFIGS = {'raw': {}, 'mean': {}}


def mask_generator(device, seed):
    return torch.Generator(device=device).manual_seed(seed)


def make_adapter(data, name, device):
    a = Adapter(data, make_graph(data), device)
    if name != 'control':
        a.model.add_module('reconstruction', Reconstruction(a.graph, a.cfg.hidden).to(device))
    return a


def make_optimizer(model):
    return torch.optim.RMSprop([
        dict(params=[p for n, p in model.named_parameters() if not n.startswith('reconstruction.')], lr=1e-4),
        dict(params=model.reconstruction.parameters(), lr=1e-4, weight_decay=1e-4)], alpha=.99)


def clone_state(model, cpu=False):
    return {k: (v.detach().cpu().clone() if cpu else v.detach().clone()) for k, v in model.state_dict().items()}


class ParameterMean:
    def __init__(self):
        self.count = 0
        self.sums = {}

    @torch.no_grad()
    def update(self, model, epoch):
        if epoch < START_EPOCH:
            assert self.count == 0
            return
        assert self.count == epoch - START_EPOCH
        if not self.count:
            self.sums = {n: p.detach().double().clone() for n, p in model.named_parameters()}
        else:
            for n, p in model.named_parameters(): self.sums[n].add_(p.detach().double())
        self.count += 1

    @torch.no_grad()
    def state(self, model):
        assert self.count > 0
        result = clone_state(model)
        for n, p in model.named_parameters(): result[n] = (self.sums[n] / self.count).to(p.dtype)
        # BN sees fixed training EASE inputs, independent of trainable parameters.
        # Keep current trajectory buffers, including integer update count.
        return result


@torch.no_grad()
def predict_mean(a, average):
    before = clone_state(a.model)
    try:
        a.model.load_state_dict(average.state(a.model)); a.model.eval()
        probability = a.predict('val')
    finally:
        a.model.load_state_dict(before)
    return probability


def step(a, optimizer, epoch, generator):
    from torch.nn import functional as F
    m = a.model; m.train(); optimizer.zero_grad(set_to_none=True)
    bce = F.binary_cross_entropy_with_logits(a.logits('train'), a.y)
    indices = m.reconstruction.masks(a.graph, .5, generator)
    auxiliary, separate = m.reconstruction.loss(m.encoder, a.graph, indices, False)
    weight = 10 * min(1., epoch / 50.)
    loss = bce + weight * auxiliary
    assert torch.isfinite(loss); loss.backward(); optimizer.step(); m.eval()
    return dict(bce=float(bce.detach()), reconstruction=float(auxiliary.detach()),
                reconstruction_rna=float(separate['miRNA'].detach()), reconstruction_drug=float(separate['drug'].detach()),
                weight=weight, loss=float(loss.detach()))
