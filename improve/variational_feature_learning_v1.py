"""A narrower VBLL hypothesis: variance gradients shape graph features.

Full and MAP retain their previous numerical implementations. The sole new
control stops only feature gradients of marginal predictive variance, keeping
the same forward objective, mean path, head gradients and predictive integral.
"""
import torch

from improve import variational_lastlayer_graph_v1 as parent

KINDS = ('full', 'no_feature_variance', 'map')
parameters = parent.parameters
identity_task = parent.identity_task
objective = parent.objective
residual_prediction = parent.residual_prediction


class DetachedFeatureVariance(parent.GaussianClassifier):
    def forward(self, features):
        if features.ndim != 2 or features.shape[1] != self.dimension-1:
            raise ValueError('Unexpected last-layer feature shape')
        phi = torch.cat((features.double(),features.new_ones((len(features),1)).double()),-1)
        mean = phi @ self.mean.double().T
        variance = phi.detach().square() @ (2*self.log_scale.double()).exp().T
        return mean,variance


def setup(a,p,kind,device):
    if kind not in KINDS:
        raise ValueError('Unregistered variational feature-learning arm')
    model,xs,graphs,folds = parent.setup(a,p,'map' if kind == 'map' else 'diagonal',device)
    if kind == 'no_feature_variance' and xs:
        previous = model.last
        model.last = DetachedFeatureVariance(p['width'],'diagonal',p['prior_scale']).to(device)
        model.last.load_state_dict(previous.state_dict())
    return model,xs,graphs,folds
