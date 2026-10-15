"""Unmasked10 with fixed 300 updates, never prediction-fold checkpoint selection."""
import argparse,json,time
import numpy as np
import torch
from torch.nn import functional as F
from common import HERE,CLEAN,seed_all,sha256,nested_data,make_base,mask_generator

def main():
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,choices=range(3),required=True)
    p.add_argument('--nested',type=int,choices=range(-1,5),required=True);p.add_argument('--device',default='cuda:0');args=p.parse_args()
    plan=json.loads((HERE/'bases/plan.json').read_text())
    for path,h in {**plan['source_hashes'],**plan['evidence_hashes']}.items():assert sha256(path)==h
    torch.cuda.set_device(args.device);seed_all(42);data,rows=nested_data(args.fold,args.nested);a=make_base(data,args.device);m=a.model
    original=[p for n,p in m.named_parameters() if not n.startswith('reconstruction.')]
    opt=torch.optim.RMSprop([dict(params=original,lr=1e-4),dict(params=m.reconstruction.parameters(),lr=1e-4,weight_decay=1e-4)],alpha=.99)
    g=mask_generator(args.device,20261006+args.fold);out=HERE/'bases';tag=f'base_f{args.fold}_k{args.nested}';assert not (out/f'{tag}.json').exists()
    started=time.time();curves=[]
    for epoch in range(1,301):
        m.train();opt.zero_grad(set_to_none=True);bce=F.binary_cross_entropy_with_logits(a.logits('train'),a.y)
        idx=m.reconstruction.masks(a.graph,.5,g);rec,_=m.reconstruction.loss(m.encoder,a.graph,idx,False)
        weight=10*min(1.,epoch/50.);loss=bce+weight*rec;assert torch.isfinite(loss);loss.backward();opt.step()
        curves.append(dict(epoch=epoch,bce=float(bce),reconstruction=float(rec),weight=weight,loss=float(loss)))
        if epoch%100==0:print(tag,epoch,time.time()-started,flush=True)
    m.eval()
    with torch.no_grad():tr=a.logits('train').cpu().numpy();va=a.logits('val').cpu().numpy()
    torch.save(m.state_dict(),out/f'{tag}.pt')
    np.savez_compressed(out/f'{tag}.npz',train_logit=tr,val_logit=va,train_pairs=data['train_pairs'],val_pairs=data['val_pairs'],
        train_indices=data['train_indices'],val_indices=data['val_indices'],prediction_rows=rows)
    r=dict(tag=tag,fold=args.fold,nested=args.nested,seed=42,epochs=300,curves=curves,
        checkpoint_selection='fixed final epoch; no prediction-fold metrics',prediction_fold_labels_used_for_selection=False,
        checkpoint_sha256=sha256(out/f'{tag}.pt'),cache_sha256=sha256(out/f'{tag}.npz'),plan_sha256=sha256(out/'plan.json'),source_hashes=plan['source_hashes'],
        graph_sha256=sha256(CLEAN/'graph.pt'),data_sha256=sha256(CLEAN/'inner_data.npz'),
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False,elapsed_seconds=time.time()-started)
    (out/f'{tag}.json').write_text(json.dumps(r,indent=2));print('DONE',tag,flush=True)

if __name__=='__main__':main()
