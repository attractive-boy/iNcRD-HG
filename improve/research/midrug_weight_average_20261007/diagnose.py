"""Post-selection train/validation and reconstruction diagnostics, no reselection."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from common import HERE, load_inner, seed_all, sha256
from model import make_adapter, clone_state


def main():
    selection_hash=sha256(HERE/'selection.json');plan=json.loads((HERE/'results/plan.json').read_text())
    assert json.loads((HERE/'verification.json').read_text())['status']=='passed'
    records=[]
    for fold in range(3):
        data=load_inner(fold)
        for name in plan['configs']:
            path=HERE/'results'/f'{name}_f{fold}_s42.pt';meta=json.loads(path.with_suffix('.json').read_text())
            assert sha256(path)==meta['checkpoint_sha256'];seed_all(42);a=make_adapter(data,name,'cuda:0')
            a.model.load_state_dict(torch.load(path,map_location='cuda:0',weights_only=True));a.model.eval();before=clone_state(a.model)
            row=dict(config=name,fold=fold,best_epoch=meta['best_epoch'],epochs_run=meta['epochs_run'],average_count=meta.get('average_count',1))
            with torch.no_grad():
                for part in ['train','val']:
                    logits=a.logits(part);prob=logits.sigmoid().cpu().numpy();y=data[part+'_y']
                    row[part]=dict(auc=float(roc_auc_score(y,prob)),ap=float(average_precision_score(y,prob)),
                                   bce=float(F.binary_cross_entropy_with_logits(logits,torch.as_tensor(y,device=logits.device,dtype=logits.dtype))))
                ix={n:torch.arange(a.graph[n].num_nodes,device=a.y.device) for n in ['miRNA','drug']}
                aux,separate=a.model.reconstruction.loss(a.model.encoder,a.graph,ix,False)
                row['full_node_reconstruction']=dict(mean=float(aux),**{k:float(v) for k,v in separate.items()})
            row['train_minus_val']={m:row['train'][m]-row['val'][m] for m in ['auc','ap']}
            for key,value in before.items():torch.testing.assert_close(value,a.model.state_dict()[key],atol=0,rtol=0)
            for m in ['auc','ap']:assert abs(row['val'][m]-meta['inner_'+m])<2e-6
            curves=json.loads(path.with_name(path.stem+'_curves.json').read_text());last=curves[-1]
            row['last_curve']={m:last[m] for m in ['epoch','inner_auc','inner_ap']}
            records.append(row);print('PASS',name,fold,flush=True);del a,before;torch.cuda.empty_cache()
    means=[]
    for name in plan['configs']:
        rows=[r for r in records if r['config']==name]
        means.append(dict(config=name,**{part:{m:float(np.mean([r[part][m] for r in rows])) for m in ['auc','ap','bce']} for part in ['train','val']},
            reconstruction=float(np.mean([r['full_node_reconstruction']['mean'] for r in rows])),
            train_minus_val={m:float(np.mean([r['train_minus_val'][m] for r in rows])) for m in ['auc','ap']}))
    assert sha256(HERE/'selection.json')==selection_hash
    result=dict(status='passed',records=records,means=means,selection_sha256=selection_hash,selection_changed=False,
        no_parameter_or_buffer_updates=True,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    (HERE/'component_diagnostics.json').write_text(json.dumps(result,indent=2))
    table='\n'.join(f"| {r['config']} | {r['train']['auc']:.6f}/{r['train']['ap']:.6f} | {r['val']['auc']:.6f}/{r['val']['ap']:.6f} | {r['train']['bce']:.6f} | {r['reconstruction']:.6f} |" for r in means)
    (HERE/'COMPONENT_DIAGNOSTICS.md').write_text(f'''# 固定参数平均：冻结选择后的诊断

| 模型 | 训练 AUC/AP | 内部留出 AUC/AP | 训练 BCE | 全节点重构 MSE |
|---|---:|---:|---:|---:|
{table}

各折平均快照数、训练/留出差距和最后曲线点见JSON。两种模型均按预定内部规则
独立选轮；其所选epoch可能不同，所以该表不是同一epoch的参数干预实验。
所有参数/缓冲和选型哈希保持不变，未按诊断调整平均起点、区间或成员。
''')


if __name__=='__main__':main()
