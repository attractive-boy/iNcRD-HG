"""Randomly initialized TabM and newly fitted CatBoost on fixed pair inputs."""
import hashlib
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.nn import functional as F

from improve.graph_tabm_modules import GraphTabM


PAIR_DIM=62


def tree_inputs(a,view):
    if view not in ('graph36','tiered62','expanded62'):
        raise ValueError('Unregistered feature condition')
    output=[]
    for part in ('train','val'):
        raw=a['raw_'+part]
        if raw.shape!=(len(a['y_'+part]),36):
            raise ValueError('Expected 36 frozen graph columns')
        extra=(np.zeros((len(raw),26),np.float32) if view=='graph36'
               else a[view+'_raw_extra_'+part])
        x=np.concatenate([raw,extra],axis=1).astype(np.float32)
        if x.shape!=(len(raw),PAIR_DIM) or not np.isfinite(x).all():
            raise ValueError('Expected 62 finite tree input columns')
        output.append(x)
    return tuple(output)


def make_tabm(cfg,seed,device):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    return GraphTabM(PAIR_DIM,**{k:cfg[k] for k in ('width','depth','members','dropout')}).to(device)


def state_digest(model):
    value=hashlib.sha256()
    for name,tensor in model.state_dict().items():
        a=tensor.detach().cpu().contiguous().numpy()
        value.update(name.encode());value.update(str(a.shape).encode())
        value.update(str(a.dtype).encode());value.update(a.tobytes())
    return value.hexdigest()


def ensemble_log_odds(logits):
    """Log odds of mean member probability, retaining ranks near 0 and 1."""
    logits=logits.double()
    return (torch.logsumexp(F.logsigmoid(logits),dim=1)
            -torch.logsumexp(F.logsigmoid(-logits),dim=1))


@torch.no_grad()
def predict_tabm(model,x,batch_size=1024):
    model.eval()
    result=np.concatenate([ensemble_log_odds(model(x[start:start+batch_size])).cpu().numpy()
                           for start in range(0,len(x),batch_size)])
    if result.shape!=(len(x),) or not np.isfinite(result).all():
        raise ValueError('Invalid TabM ranking scores')
    return result


def train_tabm_epoch(model,optimizer,x,y,batch_size):
    """One full-support optimizer update, matching historical TabM training."""
    model.train();optimizer.zero_grad(set_to_none=True);value=0.
    for start in range(0,len(x),batch_size):
        logits=model(x[start:start+batch_size])
        target=y[start:start+batch_size,None].expand_as(logits)
        loss=F.binary_cross_entropy_with_logits(logits,target,reduction='sum')/(len(x)*logits.shape[1])
        if not bool(torch.isfinite(loss)):
            raise ValueError('Nonfinite TabM training loss')
        loss.backward();value+=float(loss.detach())
    if not all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None):
        raise ValueError('Nonfinite TabM gradient')
    optimizer.step()
    return value


def catboost_class(dependency_dir):
    directory=Path(dependency_dir).resolve()
    if str(directory) not in sys.path:
        sys.path.insert(0,str(directory))
    import catboost
    if catboost.__version__!='1.2.8' or directory not in Path(catboost.__file__).resolve().parents:
        raise ValueError('Expected the isolated, pinned CatBoost package')
    return catboost.CatBoostClassifier


def fresh_catboost(cfg,seed,dependency_dir):
    model=catboost_class(dependency_dir)(**cfg,random_seed=seed)
    if model.is_fitted():
        raise ValueError('CatBoost must start unfitted')
    return model


def fit_catboost(model,x,y):
    """No evaluation set, external checkpoint or init_model is accepted."""
    if model.is_fitted():
        raise ValueError('Refusing to initialize from a fitted model')
    return model.fit(x,y,verbose=False)


def predict_catboost(model,x):
    score=np.asarray(model.predict(x,prediction_type='RawFormulaVal',thread_count=4),dtype=np.float64)
    if score.shape!=(len(x),) or not np.isfinite(score).all():
        raise ValueError('Invalid CatBoost ranking scores')
    return score
