"""Fixed 300-epoch M123 base training; prediction-fold labels NEVER select epochs."""
import argparse,json,time
import numpy as np
import torch
from common import HERE,CLEAN,Adapter,seed_all,sha256,nested_data,make_graph,Extractor,SG

def main():
    p=argparse.ArgumentParser();p.add_argument('--fold',type=int,required=True);p.add_argument('--nested',type=int,required=True)
    p.add_argument('--device',default='cuda:0');args=p.parse_args();torch.cuda.set_device(args.device);seed_all(42)
    data,pred_rows=nested_data(args.fold,args.nested);tag=f'base_f{args.fold}_k{args.nested}'
    out=HERE/'bases';out.mkdir(exist_ok=True)
    if (out/f'{tag}.json').exists():raise FileExistsError(tag)
    a=Adapter(data,make_graph(data),args.device);opt=torch.optim.RMSprop(a.model.parameters(),lr=1e-4,alpha=.99)
    start=time.time();losses=[]
    for epoch in range(1,301):
        a.model.train();loss=a.train_epoch(opt,epoch);assert np.isfinite(loss);losses.append(loss)
        if epoch%100==0:print(tag,epoch,loss,time.time()-start,flush=True)
    a.model.eval()
    with torch.no_grad():logit=a.logits('val').cpu().numpy()
    torch.save(a.model.state_dict(),out/f'{tag}.pt')
    graph=Extractor(data).many(data['val_pairs'],SG['k'],SG['hops'])
    np.savez_compressed(out/f'{tag}.npz',logit=logit,pairs=data['val_pairs'],prediction_indices=data['val_indices'],train_indices=data['train_indices'],prediction_rows=pred_rows,**graph)
    record=dict(tag=tag,fold=args.fold,nested=args.nested,seed=42,epochs=300,checkpoint_selection='fixed final epoch, no prediction-fold label selection',
        prediction_fold_labels_used_for_selection=False,n_train=len(data['train_y']),n_predict=len(logit),train_losses=losses,
        checkpoint_sha256=sha256(out/f'{tag}.pt'),cache_sha256=sha256(out/f'{tag}.npz'),data_sha256=sha256(CLEAN/'inner_data.npz'),
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False,elapsed_seconds=time.time()-start)
    (out/f'{tag}.json').write_text(json.dumps(record,indent=2));print('DONE',tag,flush=True)

if __name__=='__main__':main()

