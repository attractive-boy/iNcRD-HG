"""Validation-only selection and auditable test predictions shared by all adapters."""
import json, random, time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                            precision_recall_fscore_support, matthews_corrcoef, confusion_matrix)
from bench_data import HERE, sha

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    torch.set_num_threads(2)

def metrics(y,p):
    assert len(y)==len(p) and np.isfinite(p).all()
    yp=(p>=0.5).astype(int)
    pr,re,f1,_=precision_recall_fscore_support(y,yp,average='binary',zero_division=0)
    tn,fp,fn,tp=confusion_matrix(y,yp,labels=[0,1]).ravel()
    return dict(auc=float(roc_auc_score(y,p)),aupr=float(average_precision_score(y,p)),
                acc=float(accuracy_score(y,yp)),precision=float(pr),recall=float(re),
                specificity=float(tn/max(1,tn+fp)),f1=float(f1),mcc=float(matthews_corrcoef(y,yp)))

def fit(adapter,data,args):
    tag=f'{args.model}_{args.dataset}_f{args.fold}_s{args.seed}'
    out=HERE/args.output; out.mkdir(exist_ok=True,parents=True)
    model=adapter.model
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.wd)
    best=-float('inf'); best_ep=0; state=None; curves=[]; started=time.time()
    for epoch in range(1,args.epochs+1):
        model.train(); loss=adapter.train_epoch(optimizer,epoch)
        if not np.isfinite(loss): raise RuntimeError(f'Nonfinite training loss: {tag} epoch {epoch}')
        if epoch % args.eval_every and epoch!=1 and epoch!=args.epochs: continue
        model.eval()
        with torch.no_grad(): val_p=adapter.predict('val')
        val=metrics(data['val_y'],val_p)
        curves.append({'epoch':epoch,'loss':loss,'val_auc':val['auc'],'val_aupr':val['aupr']})
        if val['auc']>best:
            best=val['auc']; best_ep=epoch
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch==1 or epoch%20==0:
            print(f'{tag} ep={epoch} loss={loss:.5f} val_auc={val["auc"]:.5f} best={best:.5f} elapsed={time.time()-started:.0f}s',flush=True)
        if epoch>=args.min_epochs and epoch-best_ep>=args.patience: break
    assert state is not None
    model.load_state_dict(state); model.eval()
    with torch.no_grad():
        vp=adapter.predict('val'); tp=adapter.predict('test')
    result=dict(model=args.model,dataset=args.dataset,fold=args.fold,seed=args.seed,
                selection='validation_auc',graph='train_only',threshold=0.5,
                best_epoch=best_ep,epochs_run=epoch,elapsed_seconds=time.time()-started,
                val_auc=metrics(data['val_y'],vp)['auc'],**metrics(data['test_y'],tp),
                split_sha256=sha(HERE/f'data/{args.dataset}_fold{args.fold}.npz'),
                adapter_note=adapter.note,config=vars(args))
    np.savez_compressed(out/f'{tag}_predictions.npz',test_pairs=data['test_pairs'],test_y=data['test_y'],
                        test_probability=tp,val_pairs=data['val_pairs'],val_y=data['val_y'],val_probability=vp)
    (out/f'{tag}_curves.json').write_text(json.dumps(curves,indent=2))
    torch.save(state,out/f'{tag}.pt')
    result['checkpoint_sha256']=sha(out/f'{tag}.pt')
    (out/f'{tag}.json').write_text(json.dumps(result,indent=2))
    print('DONE '+json.dumps({k:result[k] for k in ('model','dataset','fold','seed','auc','aupr','best_epoch','elapsed_seconds')}),flush=True)
    return result
