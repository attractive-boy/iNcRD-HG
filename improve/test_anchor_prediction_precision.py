import unittest
import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from improve.anchor_prediction_precision import probability


class AnchorPrecisionTest(unittest.TestCase):
    def test_fitted_parameters_unchanged_and_query_order_stable(self):
        rng=np.random.default_rng(73)
        x=rng.normal(size=(100,7)).astype(np.float32);y=(x[:,0]+x[:,1]>0).astype(int)
        model=make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(16,8),random_state=2,max_iter=100))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',ConvergenceWarning)
            model.fit(x,y)
        original=[w.copy() for w in model.steps[-1][1].coefs_]
        expected=model.predict_proba(x)[:,1]
        value=probability('mlp',model,x)
        reverse=probability('mlp',model,x[::-1])[::-1]
        chunks=np.concatenate([probability('mlp',model,x[i:i+13]) for i in range(0,len(x),13)])
        np.testing.assert_allclose(value,expected,atol=1e-6,rtol=1e-6)
        np.testing.assert_allclose(value,reverse,atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(value,chunks,atol=1e-12,rtol=1e-12)
        for before,after in zip(original,model.steps[-1][1].coefs_):
            np.testing.assert_array_equal(before,after)
            self.assertEqual(before.dtype,after.dtype)

    def test_preserve_native_sigmoid_saturation(self):
        x=np.array([[0.],[1.],[2.],[3.]],np.float32);y=np.array([0,0,1,1])
        model=make_pipeline(StandardScaler(),MLPClassifier(hidden_layer_sizes=(2,),random_state=8,max_iter=2))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',ConvergenceWarning)
            model.fit(x,y)
        network=model.steps[-1][1]
        for weight in network.coefs_:weight.fill(0.)
        for bias in network.intercepts_:bias.fill(0.)
        for logit in (17.,25.,-25.):
            network.intercepts_[-1].fill(logit)
            np.testing.assert_array_equal(probability('mlp',model,x),model.predict_proba(x)[:,1])


if __name__=='__main__':unittest.main()
