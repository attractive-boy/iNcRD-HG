"""Freeze retrieval selection before component diagnostics and archive."""
import csv,json,shutil
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,RELEASE,sha256,load_inner,seed_all
from model import make_adapter

def main():
    plan=json.loads((HERE/'results/plan.json').read_text());v=json.loads((HERE/'verification.json').read_text())
    assert v['status']=='passed' and len(v['records'])==15 and v['new_checkpoint_replays']==9 and v['immutable_reused_control_checks']==6
    for p,h in {**plan['source_hashes'],**plan['evidence_hashes']}.items():assert sha256(p)==h
    runs={n:[json.loads(Path(c[str(f)]['path']).read_text()) for f in range(3)] for n,c in plan['controls'].items()}
    runs.update({n:[json.loads((HERE/'results'/f'{n}_f{f}_s42.json').read_text()) for f in range(3)] for n in plan['configs']})
    rows=[dict(config=n,mean_inner_auc=float(np.mean([r['inner_auc'] for r in rs])),mean_inner_ap=float(np.mean([r['inner_ap'] for r in rs]))) for n,rs in runs.items()]
    best=next(r for r in rows if r['config']=='label')
    best['comparisons']={n:{m:dict(mean_delta=float(np.mean([r['inner_'+m]-b['inner_'+m] for r,b in zip(runs['label'],runs[n])])),
        fold_deltas=[r['inner_'+m]-b['inner_'+m] for r,b in zip(runs['label'],runs[n])]) for m in ['auc','ap']} for n in plan['comparators']}
    best['selection_score']=min(c[m]['mean_delta'] for c in best['comparisons'].values() for m in ['auc','ap'])
    promote=best['selection_score']>=plan['promotion_minimum_both_gains_over_each_control']
    selection=dict(status='frozen_inner_selection',rows=rows,best=best,promote_to_outer_validation=promote,
        plan_sha256=sha256(HERE/'results/plan.json'),verification_sha256=sha256(HERE/'verification.json'),
        result_hashes={p.name:sha256(p) for p in sorted((HERE/'results').glob('*_s42.json'))},
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    sp=HERE/'selection.json'
    if sp.exists():assert json.loads(sp.read_text())==selection
    else:sp.write_text(json.dumps(selection,indent=2))
    diagnostics=[]
    for name in plan['configs']:
        for fold in range(3):
            r=runs[name][fold];tag=r['tag'];p=HERE/'results'/f'{tag}.pt'
            seed_all(42);data=load_inner(fold);a=make_adapter(data,name,'cuda:0')
            a.model.load_state_dict(torch.load(p,map_location='cuda:0',weights_only=True));a.model.eval();memory=a.model.case_memory
            with np.load(HERE/'replay_predictions'/f'{tag}.npz') as z:train=z['train'];val=z['val']
            with torch.no_grad():
                query=a.pairs['val'];_,_,idx,weight=memory.context(query);delta=memory(query)
            d=dict(config=name,fold=fold,train_auc=roc_auc_score(data['train_y'],train),train_ap=average_precision_score(data['train_y'],train),
                inner_auc=r['inner_auc'],inner_ap=r['inner_ap'],mean_abs_residual_logit=float(delta.abs().mean()),rms_residual_logit=float(delta.square().mean().sqrt()))
            if idx is not None:
                with torch.no_grad():
                    contexts=memory.support_pairs[idx];qy=query.T[:,None,:]
                    d.update(mean_effective_neighbors=float((1/weight.square().sum(1)).mean()),
                        same_miRNA_mass=float((weight*(contexts[:,:,0]==qy[:,:,0])).sum(1).mean()),
                        same_drug_mass=float((weight*(contexts[:,:,1]==qy[:,:,1])).sum(1).mean()),
                        positive_label_mass=float((weight*memory.support_y[idx]).sum(1).mean()))
            diagnostics.append(d);del a,memory;torch.cuda.empty_cache()
    diag_means={n:{k:float(np.mean([r[k] for r in diagnostics if r['config']==n])) for k in diagnostics[next(i for i,r in enumerate(diagnostics) if r['config']==n)] if k not in ['config','fold']} for n in plan['configs']}
    (HERE/'component_diagnostics.json').write_text(json.dumps(dict(status='passed',selection_sha256=sha256(sp),records=diagnostics,means=diag_means,no_further_selection=True),indent=2))
    flat=[dict(config=n,fold=f,**{k:r[k] for k in ['inner_auc','inner_ap','best_epoch','epochs_run']}) for n,rs in runs.items() for f,r in enumerate(rs)]
    with (HERE/'inner_summary.csv').open('w') as out:
        w=csv.DictWriter(out,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    original=HERE.parent/'midrug_internal_20261006/plmf_protection.json';protected=json.loads(original.read_text())
    for p,h in protected['source_hashes'].items():assert sha256(RELEASE/p)==h
    frozen=RELEASE/'data/frozen';manifest=json.loads((frozen/'manifest.json').read_text());refs=HERE.parent/'baselines_m123_20261004/results';protection=[]
    for r in protected['records']:
        ds,f=r['dataset'],r['fold'];assert sha256(refs/f'incrd_m123_{ds}_f{f}_s42.pt')==r['checkpoint_sha256']
        assert sha256(frozen/f'{ds}_fold{f}.npz')==r['split_sha256'];assert sha256(frozen/f'{ds}_graph.pt')==manifest['datasets'][ds]['graph_sha256']
        protection.append(dict(dataset=ds,fold=f,checkpoint_sha256=r['checkpoint_sha256'],split_sha256=r['split_sha256'],graph_sha256=manifest['datasets'][ds]['graph_sha256']))
    (HERE/'plmf_hash_recheck.json').write_text(json.dumps(dict(status='passed',records=protection,source_unchanged=True,prior_full_replay_sha256=sha256(original)),indent=2))
    table='\n'.join(f"| {r['config']} | {r['mean_inner_auc']:.9f} | {r['mean_inner_ap']:.9f} |" for r in rows)
    paired='\n'.join(f"| {ref} | {c['auc']['mean_delta']:+.9f} | {c['ap']['mean_delta']:+.9f} |" for ref,c in best['comparisons'].items())
    diagtable='\n'.join(f"| {n} | {r['train_auc']:.6f} | {r['train_ap']:.6f} | {r['rms_residual_logit']:.6f} |" for n,r in diag_means.items())
    new=[r for r in v['records'] if not r['reused_control']];deleted=max(x['errors']['full_logit'] for r in new for x in r['explicit_deletions'])
    report=f'''# 原属性检索与非线性标签值：内部检验

**原测试AUC/AP同时绝对+.02尚未证实。** 九次新训练及检查点重放完成，另核验六个
既有纯M123/unmasked10完整重放记录。仅原训练三内折seed42，无新原验证/测试评估。

| 配置 | 内部 AUC | 内部 AP |
|---|---:|---:|
{table}

| label的规定对照 | ΔAUC | ΔAP |
|---|---:|---:|
{paired}

唯一候选label的最小规定平均增益{best['selection_score']:+.9f}；推进原验证：{promote}。
门槛是相对全部规定对照的三内折平均AUC/AP均≥+.002。不事后改选对照或调整邻居数。
最终仍需原测试seeds42/43/44均值AUC≥.9733929274、AP≥.9336833737，PLMF两集不降。

## 方法和边界

按原属性学习配对key，以负平方L2从当前训练案例中取固定32近邻，排除正负查询自身。
非线性value为分类标签embedding加key差值MLP，softmax加权后加到查询hidden，再输出
原M123 logit的残差。features移除标签value，direct没有案例检索。原M123和unmasked10
重构保留；各臂有效参数和计算量不同，不声称等容量对照。案例输入作为非持久buffer
由冻结当折训练数据重建；标签不参与key和无监督原属性标准化。

此为TabR较早方法的本地适配，论文/固定作者代码与近期同领域方法的适用性见
[文献笔记](../midrug_task_literature_20261007/READING_NOTES.md)。历史local_support已有
配对检索加Newton残差；本轮不同在于直接学习标签及key差值value，不是首次尝试检索。
不使用外部数据、序列、注释、新权重或lncRNA，不将采样零标签解释为药物敏感性。

预检零残差恢复原模型、真实更新梯度、正负自身标签翻转和物理移除案例、独立NumPy
距离及value聚合通过。九模型重放核验概率/指标/最早最大均值选轮、损失组成、BN计数、
案例边界和查询批次不变；完整M1/M2/M3物理删查询最大logit误差{deleted:.3g}。

## 冻结选择后诊断

| 配置 | 训练 AUC | 训练 AP | 残差logit RMS |
|---|---:|---:|---:|
{diagtable}

这些是各自所选检查点的训练集拟合与分支幅度，不能用训练拟合改善代替留出增益。
组件诊断还记录有效近邻数和相同miRNA/药物的注意力质量，未据此重新选型。
PLMF十个已完整重放模型、冻结图/划分和发布源码哈希保持一致，本轮未重新训练或
重放PLMF，不能把保护性哈希核对称为新候选跨数据集验证。
'''
    (HERE/'REPORT.md').write_text(report)
    state=dict(goal_status='active',completion_proven=False,stage='completed_and_audited',best=best,promote_to_outer_validation=promote,
        new_fits=9,new_checkpoint_replays=9,reused_control_checks=6,no_lncrna=True,new_candidate_test_evaluations=0,processes_expected_live=[],
        component_diagnostics=diag_means,next_action='Review frozen retrieval component evidence and prior experiments to choose a distinct next hypothesis; no retrospective neighbor/width/weight sweep.')
    (HERE/'CONTINUATION.json').write_text(json.dumps(state,indent=2))
    progress=HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md';prefix=progress.read_text().split('最新完成')[0]
    progress.write_text(prefix+f'''最新完成[原属性案例检索](midrug_label_retrieval_20261007/REPORT.md)九次新训练及模型重放。
label内部AUC/AP为{best['mean_inner_auc']:.9f}/{best['mean_inner_ap']:.9f}，推进原验证：{promote}。
前轮[八成员集成](midrug_independent_ensemble_20261007/REPORT.md)21次训练/重放也已封存。
没有新原验证/测试成绩；PLMF十个模型及冻结输入/发布源码哈希未变。
近期[任务文献与TabR核读](midrug_task_literature_20261007/READING_NOTES.md)已归档。
''')
    scope_path=HERE.parent/'ACTIVE_SCOPE.json';scope=json.loads(scope_path.read_text())
    scope.update(active_directory=str(HERE),latest_continuation=str(HERE/'CONTINUATION.json'),next_action=state['next_action'])
    snap=HERE/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,snap/p.name)
    paths=['verification.json','selection.json','REPORT.md','CONTINUATION.json','plmf_hash_recheck.json','results/plan.json',
        'initial_checks.json','README.md','inner_summary.csv','check.log','launch.log','audit.log','component_diagnostics.json']
    paths+=sorted(str(p.relative_to(HERE)) for p in (HERE/'results').iterdir() if p.suffix in {'.pt','.npz','.json'} and p.name!='plan.json')
    paths+=sorted(str(p.relative_to(HERE)) for p in (HERE/'replay_predictions').glob('*.npz'))
    archive=dict(status='archived',source_hashes={p.name:sha256(p) for p in sorted(HERE.glob('*.py'))},artifact_hashes={p:sha256(HERE/p) for p in paths},frozen_dependencies={**plan['source_hashes'],**plan['evidence_hashes']})
    (HERE/'archive_manifest.json').write_text(json.dumps(archive,indent=2));count=0
    for p,h in archive['source_hashes'].items():assert sha256(HERE/p)==h and sha256(snap/p)==h;count+=2
    for p,h in archive['artifact_hashes'].items():assert sha256(HERE/p)==h;count+=1
    for p,h in archive['frozen_dependencies'].items():assert sha256(p)==h;count+=1
    (HERE/'archive_recheck.json').write_text(json.dumps(dict(status='passed',hash_checks=count,new_checkpoint_replays=9,
        reused_control_checks=6,protected_plmf_records=10,selection_unchanged=True,archive_sha256=sha256(HERE/'archive_manifest.json'),
        selection_sha256=sha256(sp),no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))
    scope['latest_archive_check']=str(HERE/'archive_recheck.json');scope_path.write_text(json.dumps(scope,indent=2))
    print(json.dumps(dict(best=best,promote=promote,diagnostics=diag_means),indent=2))

if __name__=='__main__':main()
