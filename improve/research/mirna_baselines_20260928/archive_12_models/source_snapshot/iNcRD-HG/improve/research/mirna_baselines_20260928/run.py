import argparse, os

DEFAULTS={
 'gcmdr':(500,1e-3,5e-4,1,100,80), 'ammgc':(300,1e-2,0.,1,80,60),
 'lagcn':(500,1e-3,5e-4,1,100,80), 'jmssmma':(1000,1e-3,5e-5,5,200,150),
 'graphdta':(150,5e-4,0.,1,40,30), 'submdta':(150,5e-4,0.,1,40,30),
 'mldti':(150,1e-4,0.,1,40,30), 'dlstmda':(100,5e-4,0.,1,40,25),
 'dlmvf':(80,5e-4,0.,1,30,20), 'gcnnmma':(100,1e-3,1e-6,1,40,25),
 'plmf':(80,5e-4,5e-4,1,30,20), 'subknet':(60,1e-2,5e-4,1,20,15),
}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',choices=list(DEFAULTS),required=True)
    ap.add_argument('--dataset',choices=['MiDrug','MDR','MDS'],required=True)
    ap.add_argument('--fold',type=int,default=0); ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--device',default='cuda:1'); ap.add_argument('--output',default='results')
    for name,typ in [('epochs',int),('lr',float),('wd',float),('eval-every',int),('min-epochs',int),('patience',int)]:
        ap.add_argument('--'+name,type=typ,default=None)
    ap.add_argument('--batch',type=int,default=128)
    ap.add_argument('--compact-subknet',action='store_true')
    args=ap.parse_args()
    # Prevent duplicate workers from training/writing the same model/fold/seed.
    import fcntl,json
    from pathlib import Path
    out=Path(__file__).resolve().parent/args.output
    (out/'.locks').mkdir(parents=True,exist_ok=True)
    tag=f'{args.model}_{args.dataset}_f{args.fold}_s{args.seed}'
    lock=(out/'.locks'/f'{tag}.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print('Already running:',tag,flush=True);return
    if (out/f'{tag}.json').exists():
        existing=json.loads((out/f'{tag}.json').read_text())
        if existing.get('selection')=='validation_auc' and (out/f'{tag}_predictions.npz').exists():
            print('Already complete:',tag,flush=True);return
    if args.model=='jmssmma':
        import sys
        from pathlib import Path
        sys.path.insert(0,str(Path(__file__).resolve().parent/'vendor'))
    for k,v in zip(('epochs','lr','wd','eval_every','min_epochs','patience'),DEFAULTS[args.model]):
        if getattr(args,k) is None: setattr(args,k,v)
    # Author models use .cuda() internally: set their current device explicitly.
    import torch
    if args.device.startswith('cuda'): torch.cuda.set_device(args.device)
    from bench_data import load
    from engine import fit,seed_all
    seed_all(args.seed)
    data=load(args.dataset,args.fold)
    if args.model in ('gcmdr','ammgc','lagcn','jmssmma'):
        from graph_adapters import GraphAdapter
        adapter=GraphAdapter(data,args)
    elif args.model=='plmf':
        from plmf_adapter import PLMFAdapter
        adapter=PLMFAdapter(data,args)
    elif args.model=='subknet':
        from subknet_adapter import SubKNetAdapter
        adapter=SubKNetAdapter(data,args)
    else:
        from sequence_adapters import SequenceAdapter
        adapter=SequenceAdapter(data,args)
    fit(adapter,data,args)
    lock.close()

if __name__=='__main__': main()
