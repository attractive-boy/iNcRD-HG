"""Recover original affine covariates; same topology, decoder and auxiliary target."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from common import HERE, Adapter, Reconstruction, make_graph, sha256

CONFIGS = {name: dict(covariates=name, reconstruction_weight=10.)
           for name in ('aligned', 'permuted', 'zero')}

class ShapeReconstruction(Reconstruction):
    def loss(self, encoder, graph, indices, masked):
        assert not masked
        inputs = graph.x_dict
        h = encoder.layer(encoder.original.conv1, inputs)
        h = {name: F.leaky_relu(value, .1) for name, value in h.items()}
        h = encoder.layer(encoder.original.conv2, h)
        losses = {}
        for name, ix in indices.items():
            mean, scale = getattr(self, name + '_mean'), getattr(self, name + '_scale')
            target = (inputs[name][ix, :len(mean)] - mean) / scale
            losses[name] = F.mse_loss(self.decoder[name](h[name][ix]), target)
        return .5 * (losses['miRNA'] + losses['drug']), losses

def make_adapter(data, name, device):
    original = make_graph(data)
    graph = original.clone()
    manifest = json.loads((HERE / 'input_manifest.json').read_text())
    assert sha256(HERE / 'covariates.npz') == manifest['covariates_sha256']
    with np.load(HERE / 'covariates.npz') as z:
        for kind in ('miRNA', 'drug'):
            stats = z[kind + '_stats'].copy()
            if name == 'permuted': stats = stats[z[kind + '_permutation']]
            elif name == 'zero': stats[:] = 0
            else: assert name == 'aligned'
            graph[kind].x = torch.cat([graph[kind].x, torch.from_numpy(stats)], dim=1)
    a = Adapter(data, graph, device)
    a.model.add_module('reconstruction', ShapeReconstruction(original, a.cfg.hidden).to(device))
    return a

def mask_generator(device, fold):
    return torch.Generator(device=device).manual_seed(20261006 + fold)
