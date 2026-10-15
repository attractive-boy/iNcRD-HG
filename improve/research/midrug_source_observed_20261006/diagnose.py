"""Frozen-model observation-message interventions, internal development only."""
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,load_inner,seed_all,sha256
from model import CONFIGS,make_adapter

def metric(y,p):return dict(auc=roc_auc_score(y,p),ap=average_precision_score(y,p))

@torch.no_grad()
def examine(a,data):
    m=a.model;m.eval();state={n:p.clone() for n,p in m.state_dict().items()}
    full=a.predict('val')
    extra=list(m.source_path.cdm.parameters())+list(m.source_path.cmd.parameters())
    norm=float(torch.cat([p.flatten() for p in extra]).norm())
    try:
        for p in extra:p.zero_()
        no_observation=a.predict('val')
    finally:m.load_state_dict(state)
    try:
        m.source_projection.weight.zero_();no_source=a.predict('val')
    finally:m.load_state_dict(state)
    for n,p in m.state_dict().items():torch.testing.assert_close(p,state[n],atol=0,rtol=0)
    assert not m.lin1._forward_hooks
    return dict(full=metric(data['val_y'],full),zero_observation=metric(data['val_y'],no_observation),zero_projection=metric(data['val_y'],no_source),
        observation_weight_norm=norm,projection_weight_norm=float(m.source_projection.weight.norm()),
        observation_mean_abs_probability_shift=float(abs(full-no_observation).mean()),source_mean_abs_probability_shift=float(abs(full-no_source).mean()))

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
            z=examine(a,data)
            for k in ['auc','ap']:assert abs(z['full'][k]-r['inner_'+k])<1e-10
            rows.append(dict(config=name,fold=fold,checkpoint_sha256=r['checkpoint_sha256'],**z));del a;torch.cuda.empty_cache()
    means=[]
    for name in CONFIGS:
        rs=[r for r in rows if r['config']==name];z=dict(config=name)
        for mode in ['full','zero_observation','zero_projection']:z[mode]={k:float(np.mean([r[mode][k] for r in rs])) for k in ['auc','ap']}
        for k in ['observation_weight_norm','projection_weight_norm','observation_mean_abs_probability_shift','source_mean_abs_probability_shift']:z[k]=float(np.mean([r[k] for r in rs]))
        means.append(z)
    assert sha256(HERE/'selection.json')==selection
    (HERE/'component_diagnostics.json').write_text(json.dumps(dict(status='passed',records=rows,means=means,selection_sha256=selection,
        selection_changed=False,original_validation_evaluated=False,test_evaluated=False,
        limitations='Post-training message and projection interventions do not undo joint training. No intervention is a refitted model or a newly selected candidate.'),indent=2))
    table='\n'.join(f"| {r['config']} | {r['full']['auc']:.9f}/{r['full']['ap']:.9f} | {r['zero_observation']['auc']:.9f}/{r['zero_observation']['ap']:.9f} | {r['zero_projection']['auc']:.9f}/{r['zero_projection']['ap']:.9f} | {r['observation_mean_abs_probability_shift']:.6f} |" for r in means)
    (HERE/'COMPONENT_DIAGNOSTICS.md').write_text(f'''# 冻结选择后的观测消息诊断

仅原训练内部留出集，已冻结检查点与配置选择，不新增拟合。
比较完整预测、临时置零C/Cᵀ线性变换、临时置零整个源分支投影。随后恢复全部参数。

| 配置 | 完整AUC/AP | 置零观测消息AUC/AP | 置零源投影AUC/AP | 置零观测平均概率变化 |
|---|---:|---:|---:|---:|
{table}

置零C消息仍保留已联合训练的A消息、源参数及原主干，不等于重新训练source对照；
置零源投影也不撤销原主干的联合训练变化。干预结果不重新选配置、不推进原验证。
选择文件哈希不变，无新候选原验证或测试评估。
''')
    print(json.dumps(means,indent=2))

if __name__=='__main__':main()
