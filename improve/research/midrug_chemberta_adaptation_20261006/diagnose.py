"""Post-selection removal of learned drug-feature changes, no refitting."""
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,load_inner,seed_all,sha256
from model import CONFIGS,make_adapter

def metric(y,p):return dict(auc=roc_auc_score(y,p),ap=average_precision_score(y,p))

@torch.no_grad()
def examine(a,data):
    m=a.model;m.eval();ad=m.encoder.drug_adaptation;state={n:p.clone() for n,p in m.state_dict().items()}
    full=a.predict('val');delta=ad(a.graph['drug'].x)-a.graph['drug'].x
    norm=float(torch.cat([p.flatten() for p in ad.adaptation_parameters()]).norm())
    try:
        for n,p in ad.named_parameters():
            if p.requires_grad and 'up' in n:p.zero_()
        features=ad(a.graph['drug'].x)
        torch.testing.assert_close(features,a.graph['drug'].x,atol=1e-7,rtol=1e-7)
        zero=a.predict('val')
    finally:m.load_state_dict(state)
    for n,p in m.state_dict().items():torch.testing.assert_close(p,state[n],atol=0,rtol=0)
    assert m.encoder._adapted_drug is None
    return dict(full=metric(data['val_y'],full),zero_adaptation=metric(data['val_y'],zero),
        adaptation_weight_norm=norm,mean_absolute_feature_change=float(delta.abs().mean()),
        mean_absolute_probability_change=float(abs(full-zero).mean()))

def main():
    plan=json.loads((HERE/'results/plan.json').read_text());selection=sha256(HERE/'selection.json')
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
        for mode in ['full','zero_adaptation']:z[mode]={k:float(np.mean([r[mode][k] for r in rs])) for k in ['auc','ap']}
        for k in ['adaptation_weight_norm','mean_absolute_feature_change','mean_absolute_probability_change']:z[k]=float(np.mean([r[k] for r in rs]))
        means.append(z)
    assert sha256(HERE/'selection.json')==selection
    (HERE/'component_diagnostics.json').write_text(json.dumps(dict(status='passed',records=rows,means=means,selection_sha256=selection,
        selection_changed=False,original_validation_evaluated=False,test_evaluated=False,
        limitations='Zeroing learned feature changes does not undo joint backbone training; the intervention is not a retrained baseline or a new selected candidate.'),indent=2))
    table='\n'.join(f"| {r['config']} | {r['full']['auc']:.9f}/{r['full']['ap']:.9f} | {r['zero_adaptation']['auc']:.9f}/{r['zero_adaptation']['ap']:.9f} | {r['mean_absolute_feature_change']:.6f} | {r['mean_absolute_probability_change']:.6f} |" for r in means)
    (HERE/'COMPONENT_DIAGNOSTICS.md').write_text(f'''# 冻结选择后的药物适配诊断

仅内部留出集，无新拟合；临时置零新增低秩B矩阵，验证药物特征精确回到原输入。
随后恢复全部参数，选择文件哈希不变。

| 配置 | 完整AUC/AP | 置零适配AUC/AP | 平均特征变化 | 平均概率变化 |
|---|---:|---:|---:|---:|
{table}

置零适配不撤销原主干的联合训练，因此不能把干预当作重新训练的原M123。
诊断不重新选配置、不推进原验证，无新候选原验证或测试评估。
''');print(json.dumps(means,indent=2))

if __name__=='__main__':main()
