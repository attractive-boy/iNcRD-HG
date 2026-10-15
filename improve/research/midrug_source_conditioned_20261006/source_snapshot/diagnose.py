"""Post-selection branch-use diagnostics; no refitting or candidate selection."""
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,load_inner,seed_all,sha256
from model import CONFIGS,make_adapter

def metric(y,p):return dict(auc=roc_auc_score(y,p),ap=average_precision_score(y,p))

@torch.no_grad()
def examine(a,data):
    m=a.model;m.eval();state={n:p.clone() for n,p in m.state_dict().items()};captured={}
    h1=m.lin1.register_forward_pre_hook(lambda module,args:captured.update(features=args[0]))
    h2=m.source_projection.register_forward_hook(lambda module,args,output:captured.update(addition=output))
    try:full=a.predict('val')
    finally:h1.remove();h2.remove()
    base=m.lin1(captured['features']);extra=captured['addition'];w=m.source_projection.weight.clone()
    try:
        m.source_projection.weight.zero_();zero=a.predict('val')
    finally:m.source_projection.weight.copy_(w)
    mode=m.source_path.conditioned
    try:
        m.source_path.conditioned=not mode;swapped=a.predict('val')
    finally:m.source_path.conditioned=mode
    for n,p in m.state_dict().items():torch.testing.assert_close(p,state[n],atol=0,rtol=0)
    assert not m.lin1._forward_hooks and not m.lin1._forward_pre_hooks and not m.source_projection._forward_hooks
    return dict(full=metric(data['val_y'],full),zero_projection=metric(data['val_y'],zero),swapped_boundary=metric(data['val_y'],swapped),
        projection_weight_norm=float(w.norm()),addition_rms=float(extra.square().mean().sqrt()),base_pre_activation_rms=float(base.square().mean().sqrt()),
        zero_mean_abs_probability_shift=float(abs(full-zero).mean()),swapped_mean_abs_probability_shift=float(abs(full-swapped).mean()))

def main():
    plan=json.loads((HERE/'results/plan.json').read_text());selection=sha256(HERE/'selection.json')
    assert json.loads((HERE/'selection.json').read_text())['status']=='frozen_inner_selection'
    assert json.loads((HERE/'verification.json').read_text())['status']=='passed'
    for p,h in plan['source_hashes'].items():assert sha256(p)==h
    rows=[]
    for fold in range(3):
        data=load_inner(fold)
        for name in CONFIGS:
            p=HERE/'results'/f'{name}_f{fold}_s42.json';r=json.loads(p.read_text());assert sha256(p.with_suffix('.pt'))==r['checkpoint_sha256']
            seed_all(42);a=make_adapter(data,name,'cuda:0');a.model.load_state_dict(torch.load(p.with_suffix('.pt'),map_location='cuda:0',weights_only=True))
            rows.append(dict(config=name,fold=fold,checkpoint_sha256=r['checkpoint_sha256'],**examine(a,data)))
    means=[]
    for name in CONFIGS:
        rs=[r for r in rows if r['config']==name];z=dict(config=name)
        for mode in ['full','zero_projection','swapped_boundary']:
            z[mode]={k:float(np.mean([r[mode][k] for r in rs])) for k in ['auc','ap']}
        for k in ['projection_weight_norm','addition_rms','base_pre_activation_rms','zero_mean_abs_probability_shift','swapped_mean_abs_probability_shift']:
            z[k]=float(np.mean([r[k] for r in rs]))
        means.append(z)
    assert sha256(HERE/'selection.json')==selection
    (HERE/'component_diagnostics.json').write_text(json.dumps(dict(status='passed',records=rows,means=means,selection_sha256=selection,
        selection_changed=False,original_validation_evaluated=False,test_evaluated=False,
        limitations='Post-training interventions do not undo joint training of the original backbone; swapped boundaries are not refitted models. Diagnostics do not select a new configuration.'),indent=2))
    table='\n'.join(f"| {r['config']} | {r['full']['auc']:.9f}/{r['full']['ap']:.9f} | {r['zero_projection']['auc']:.9f}/{r['zero_projection']['ap']:.9f} | {r['swapped_boundary']['auc']:.9f}/{r['swapped_boundary']['ap']:.9f} | {r['addition_rms']:.6f} | {r['base_pre_activation_rms']:.6f} |" for r in means)
    (HERE/'COMPONENT_DIAGNOSTICS.md').write_text(f'''# 冻结选择后的分支使用诊断

仅内部留出集，已经固定检查点与配置选择，不做新训练。三种读出使用同一模型权重：
完整预测、临时置零新投影、临时切换source/uniform边界。操作后参数及缓冲精确恢复。

| 配置 | 完整AUC/AP | 置零投影AUC/AP | 切换边界AUC/AP | 分支增量RMS | 原lin1预激活RMS |
|---|---:|---:|---:|---:|---:|
{table}

置零投影不等于撤销新分支对原主干训练的影响；切换边界也不是重新训练对照。
这些数值只诊断已训练分支是否被使用、对边界是否敏感，不升级为额外候选。
selection.json前后哈希不变，未访问新候选原验证或测试。
''')
    print(json.dumps(means,indent=2))

if __name__=='__main__':main()
