"""Frozen-model residual and routing diagnostics; no changes to selected candidates."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,load_inner,seed_all,sha256
from model import CONFIGS,make_adapter


def metrics(y,p):return dict(auc=float(roc_auc_score(y,p)),ap=float(average_precision_score(y,p)))


@torch.no_grad()
def examine(a,data,part):
    base,residual,experts,gate=a.model.components(a.pairs[part],a.ps,a.qis[part],a.ease[part],a.dense)
    y=torch.as_tensor(data[part+'_y'],device=base.device,dtype=base.dtype)
    full=(base+residual).sigmoid().cpu().numpy();off=base.sigmoid().cpu().numpy()
    uniform=(base+experts.mean(-1)).sigmoid().cpu().numpy()
    return dict(full=metrics(data[part+'_y'],full),without_experts=metrics(data[part+'_y'],off),
        uniform_gate_intervention=metrics(data[part+'_y'],uniform),
        bce=float(F.binary_cross_entropy_with_logits(base+residual,y)),
        mean_absolute_residual=float(residual.abs().mean()),rms_residual=float(residual.square().mean().sqrt()),
        mean_absolute_probability_change=float(np.abs(full-off).mean()),
        mean_gate_weights=gate.mean(0).cpu().tolist(),
        expert_logit_rms=experts.square().mean(0).sqrt().cpu().tolist())


def main():
    selection_hash=sha256(HERE/'selection.json');assert json.loads((HERE/'verification.json').read_text())['status']=='passed'
    records=[]
    for fold in range(3):
        data=load_inner(fold)
        for name in CONFIGS:
            path=HERE/'results'/f'{name}_f{fold}_s42.pt';r=json.loads(path.with_suffix('.json').read_text());assert sha256(path)==r['checkpoint_sha256']
            seed_all(42);a=make_adapter(data,name,'cuda:0');a.model.load_state_dict(torch.load(path,map_location='cuda:0',weights_only=True));a.model.eval()
            before={k:v.detach().clone() for k,v in a.model.state_dict().items()}
            tr,va=examine(a,data,'train'),examine(a,data,'val')
            for m in ['auc','ap']:assert abs(va['full'][m]-r['inner_'+m])<2e-6
            for k,v in before.items():torch.testing.assert_close(v,a.model.state_dict()[k],atol=0,rtol=0)
            records.append(dict(config=name,fold=fold,train=tr,val=va));print('PASS',name,fold,flush=True)
            del a,before;torch.cuda.empty_cache()
    means=[]
    for name in CONFIGS:
        rows=[r for r in records if r['config']==name];item=dict(config=name)
        for part in ['train','val']:
            item[part]={k:{m:float(np.mean([r[part][k][m] for r in rows])) for m in ['auc','ap']} for k in ['full','without_experts','uniform_gate_intervention']}
            for k in ['bce','mean_absolute_residual','rms_residual','mean_absolute_probability_change']:
                item[part][k]=float(np.mean([r[part][k] for r in rows]))
        means.append(item)
    assert sha256(HERE/'selection.json')==selection_hash
    (HERE/'component_diagnostics.json').write_text(json.dumps(dict(status='passed',records=records,means=means,
        selection_sha256=selection_hash,selection_changed=False,no_parameter_or_buffer_updates=True,
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))
    table='\n'.join(f"| {r['config']} | {r['val']['full']['auc']:.6f}/{r['val']['full']['ap']:.6f} | {r['val']['without_experts']['auc']:.6f}/{r['val']['without_experts']['ap']:.6f} | {r['val']['uniform_gate_intervention']['auc']:.6f}/{r['val']['uniform_gate_intervention']['ap']:.6f} | {r['val']['mean_absolute_residual']:.5f} |" for r in means)
    (HERE/'COMPONENT_DIAGNOSTICS.md').write_text(f'''# 条件专家选型冻结后的干预诊断

| 训练配置 | 完整内部AUC/AP | 关闭专家AUC/AP | 改均匀门控AUC/AP | 平均绝对残差logit |
|---|---:|---:|---:|---:|
{table}

关闭专家保留经过联合训练的共享主干，不等于单独训练的M123消融；改变门控
不撤销训练时路由作用，不是新训练的uniform对照。所有结果只作冻结后诊断，
不据此改变候选、阈值或部署门控。训练配对指标、专家输出RMS及路由占比见JSON。
''')


if __name__=='__main__':main()
