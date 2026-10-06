import argparse
import fcntl
import json
from full_adapter import FullAdapter, HERE, load
from engine import fit, seed_all
import torch

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['MiDrug','MDR','MDS'], required=True)
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output', default=str(HERE/'results'))
    p.add_argument('--epochs', type=int, default=300)
    a = p.parse_args()
    a.model='incrd_m123'; a.lr=1e-4; a.wd=0.; a.eval_every=1
    a.min_epochs=a.epochs; a.patience=a.epochs
    from pathlib import Path
    out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    tag=f'{a.model}_{a.dataset}_f{a.fold}_s{a.seed}'
    with (out/f'{tag}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out/f'{tag}.json').exists():
            print('Already complete:',tag); return
        torch.cuda.set_device(a.device); seed_all(a.seed)
        data=load(a.dataset,a.fold)
        adapter=FullAdapter(data,a)
        fit(adapter,data,a)

if __name__=='__main__': main()
