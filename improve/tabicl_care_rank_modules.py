"""Support-rate normalization and a training-only teacher-order constraint."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from improve import tabicl_care_modules as care


class ScaleStableAdapter(care.ConditionalRelationAdapter):
    """A shared CARE controller whose support statistics use a fixed reference size.

    Repeating a support population leaves the controller's rate statistics
    unchanged. This does not assert that the underlying TabICL itself is
    invariant to context size, or that subsampling leaves those rates exact.
    """
    def __init__(self, conditional=True, width=16, normalize_support=True, gate_power=.5):
        super().__init__(conditional=conditional, width=width)
        self.normalize_support = bool(normalize_support)
        self.gate_power = float(gate_power)
        if self.gate_power not in (.5, 1.):
            raise ValueError('Unsupported uncertainty exponent')

    def normalized_evidence(self, relation, evidence):
        if relation.ndim != 3 or relation.shape[-1] != 5 or relation.shape[1] < 1:
            raise ValueError('Expected nonempty query/support/relation features')
        if evidence.shape != (relation.shape[0], 6):
            raise ValueError('Expected six query evidence statistics')
        if not self.normalize_support:
            return evidence
        # The final two relation channels are exact shared-endpoint indicators.
        # Normalize to 512 support cases before applying the original bounded
        # transforms, rather than feeding the raw support counts to the MLP.
        effective_count = 512. * relation[..., 3:].mean(1)
        normalized = effective_count / (effective_count + effective_count.new_tensor([4., 16.]))
        return torch.cat((normalized, evidence[:, 2:]), -1)

    def forward(self, relation, evidence, native_logits, strength=1.):
        if not np.isfinite(strength) or not 0. <= strength <= 4.:
            raise ValueError('Strength must lie in [0, 4]')
        evidence = self.normalized_evidence(relation, evidence)
        if native_logits.ndim != 3 or native_logits.shape[1:] != (len(evidence), 2):
            raise ValueError('Native query logits differ')
        margin = (native_logits[..., 1] - native_logits[..., 0]).abs().detach()
        uncertainty = 4. * torch.sigmoid(margin) * torch.sigmoid(-margin)
        query = torch.cat((evidence[None].expand(len(margin), -1, -1),
                           uncertainty[..., None], (margin / (1. + margin))[..., None]), -1)
        coefficients = self.global_coefficients[None, None]
        if self.conditional:
            coefficients = coefficients + self.controller(query)
        coefficients = torch.tanh(coefficients).expand(len(margin), len(evidence), -1)
        raw = torch.einsum('bqf,qsf->bqs', coefficients, relation)
        return strength * uncertainty.pow(self.gate_power)[..., None] * 2. * torch.tanh(raw / (2. * 5. ** .5))

    def config(self):
        return dict(super().config(), normalize_support=self.normalize_support, gate_power=self.gate_power)


def teacher_order_loss(logits, native_logits, targets, permutations):
    """Protect close, correctly ordered training positive/negative pairs.

    This is a regularizer, not a finite- or future-data metric guarantee. Query
    targets here are training pseudo-query targets, never validation labels.
    """
    value = care.aligned_logits(logits, permutations)
    teacher = care.aligned_logits(native_logits, permutations).detach()
    positive, negative = targets == 1, targets == 0
    if not positive.any() or not negative.any():
        return value.sum() * 0., 0
    margin = value[..., 1] - value[..., 0]
    old_margin = teacher[..., 1] - teacher[..., 0]
    difference = margin[:, positive, None] - margin[:, None, negative]
    old_difference = old_margin[:, positive, None] - old_margin[:, None, negative]
    protected = (old_difference > 0.) & (old_difference < 2.)
    if not protected.any():
        return value.sum() * 0., 0
    penalty = F.relu(.9 * old_difference - difference) / old_difference.clamp_min(.1)
    return penalty[protected].mean(), int(protected.sum().detach())


def load_adapter(path, device='cpu'):
    state = torch.load(path, weights_only=True, map_location='cpu')
    model = ScaleStableAdapter(**state['config'])
    model.load_state_dict(state['state_dict'])
    return model.to(device).eval(), state
