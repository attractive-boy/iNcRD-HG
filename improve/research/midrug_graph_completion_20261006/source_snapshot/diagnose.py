"""Post-selection descriptive component diagnostics; no new candidate selection."""
import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,load_inner,seed_all,sha256
from model import make_adapter
from incrd_hg_m123.model import Model

def metrics(y,p):return dict(auc=float(roc_auc_score(y,p)),ap=float(average_precision_score(y,p)))

def main():
    verification=json.loads((HERE/'verification.json').read_text());assert verification['status']=='passed'
    selection=json.loads((HERE/'selection.json').read_text())
    plan=json.loads((HERE/'results/plan.json').read_text())
    for path,h in plan['source_hashes'].items():assert sha256(path)==h
    records=[]
    for fold in range(3):
        data=load_inner(fold)
        for name in plan['configs']:
            path=HERE/'results'/f'{name}_f{fold}_s42.json';r=json.loads(path.read_text());seed_all(42)
            a=make_adapter(data,name,'cuda:0');assert sha256(path.with_suffix('.pt'))==r['checkpoint_sha256']
            a.model.load_state_dict(torch.load(path.with_suffix('.pt'),map_location='cuda:0',weights_only=True));a.model.eval()
            with torch.no_grad():
                full=a.predict('val');aux=a.model.soft_paths.last_query_logits.sigmoid().cpu().numpy()
                backbone=Model.forward(a.model,a.pairs['val'],a.ps,a.qis['val'],a.ease['val'],a.dense).sigmoid().cpu().numpy()
            with np.load(path.with_name(path.stem+'_inner.npz')) as z:
                np.testing.assert_array_equal(data['val_y'],z['y']);np.testing.assert_allclose(full,z['probability'],atol=2e-5,rtol=2e-5)
            y=data['val_y'];records.append(dict(config=name,fold=fold,full=metrics(y,full),scorer_only=metrics(y,aux),
                backbone_without_residual=metrics(y,backbone),checkpoint_sha256=r['checkpoint_sha256']))
            print('PASS diagnostic',name,fold,flush=True)
    means={}
    for name in plan['configs']:
        rs=[r for r in records if r['config']==name]
        means[name]={component:{metric:float(np.mean([r[component][metric] for r in rs])) for metric in ['auc','ap']}
                     for component in ['full','scorer_only','backbone_without_residual']}
    out=dict(status='completed',selection_unchanged_sha256=sha256(HERE/'selection.json'),records=records,means=means,
        interpretation='Post-selection internal description. Removing the residual does not undo auxiliary training of the shared backbone. Standalone auxiliary scores were not used to select checkpoints or configs.',
        used_for_selection=False,outer_validation_evaluated=False,test_evaluated=False)
    (HERE/'component_diagnostics.json').write_text(json.dumps(out,indent=2))
    rows=[]
    for name,components in means.items():
        for component,v in components.items():rows.append(f"| {name} | {component} | {v['auc']:.9f} | {v['ap']:.9f} |")
    report='''# 冻结选型后的分量诊断

仅用于解释原训练内部结果，不改变selection.json或据此改选配置/轮次。
backbone_without_residual仅去掉预测残差，不能撤销已经发生的辅助损失训练。
scorer_only是第一层图预测器的直接查询输出；full是已审计模型。

| 配置 | 分量 | 内部AUC | 内部AP |
|---|---|---:|---:|
'''+ '\n'.join(rows)+'\n'
    (HERE/'COMPONENT_DIAGNOSTICS.md').write_text(report)
    archive=json.loads((HERE/'archive_manifest.json').read_text())
    for name in ['component_diagnostics.json','COMPONENT_DIAGNOSTICS.md']:
        archive['artifact_hashes'][name]=sha256(HERE/name)
    (HERE/'archive_manifest.json').write_text(json.dumps(archive,indent=2))
    print(json.dumps(means,indent=2))

if __name__=='__main__':main()
