"""Stable MLP matrix arithmetic, retaining the original sigmoid dtype.

Keeping the original probability quantization matters: converting a saturated
float32 probability of 1 to a nonsaturated float64 value can materially change
the historical 1e-12-clipped logit reference. Only the linear algebra is
promoted, and the final affine output returns to the fitted dtype before
the original sigmoid is applied. Fitted parameters are never mutated.
"""
import numpy as np
from scipy.special import expit

from improve.anchor_shrinkage import logit


def probability(name,model,x):
    if name!='mlp':
        return model.predict_proba(x)[:,1].astype(np.float64)
    scaler,network=model.steps[0][1],model.steps[1][1]
    if network.activation!='relu' or network.out_activation_!='logistic' or network.n_outputs_!=1:
        raise ValueError('This audited readout supports only the frozen binary ReLU MLP recipe')
    np.testing.assert_array_equal(network.classes_,[0,1])
    hidden=scaler.transform(np.asarray(x,dtype=np.float64))
    for i,(weight,bias) in enumerate(zip(network.coefs_,network.intercepts_)):
        hidden=hidden@weight.astype(np.float64)+bias.astype(np.float64)
        if i+1<len(network.coefs_):
            hidden=np.maximum(hidden,0.)
    return expit(hidden[:,0].astype(network.coefs_[0].dtype)).astype(np.float64)


def member_mean(models,names,x):
    return np.stack([logit(probability(name,models[name],x)) for name in names]).mean(0)
