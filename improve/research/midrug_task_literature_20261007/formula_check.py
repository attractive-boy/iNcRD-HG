"""Literal-equation boundary checks, not a test of unpublished author code."""
import hashlib,json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
def softmax(x):
    v=np.exp(x-x.max());return v/v.sum()
scores=np.array([2.,1.,4.]);adj=np.array([1.,1.,0.])
literal=softmax(scores*adj);masked=softmax(np.where(adj>0,scores,-np.inf))
assert literal[2]>0 and masked[2]==0
prior=np.array([.5,.5]);independent_probs=np.array([.8,.8])
literal_rcc=float(np.sum(prior*np.log(prior/independent_probs)))
assert literal_rcc<0
binary_kl=float(np.sum(prior*np.log(prior/independent_probs)+(1-prior)*np.log((1-prior)/(1-independent_probs))))
assert binary_kl>=0
r=dict(status='passed',interpretation='Boundary of equations read literally; does not establish behavior of unavailable author code',
    dmagt_equations_4_5=dict(multiplication_then_softmax=literal.tolist(),excluded_node_weight=float(literal[2]),negative_infinity_mask=masked.tolist()),
    prior_rdrgse_equation_14=dict(prior=prior.tolist(),independent_probabilities=independent_probs.tolist(),literal_expression=literal_rcc,bernoulli_kl_with_complement=binary_kl),
    model_inputs_used=False,model_training_performed=False,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(HERE/'formula_check.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
