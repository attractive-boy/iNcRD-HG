"""Freeze sole-candidate selection before diagnostics, then protect and archive."""
import csv,json,shutil
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,RELEASE,load_inner,seed_all,sha256
from model import make_adapter

def write(p,x):p.write_text(json.dumps(x,indent=2))

def main():
    plan=json.loads((HERE/'results/plan.json').read_text());v=json.loads((HERE/'verification.json').read_text())
    assert v['status']=='passed' and v['new_checkpoint_replays']==9 and v['final_state_inspections']==9
    assert len(v['records'])==15 and v['immutable_reused_control_checks']==6
    dependencies={**plan['source_hashes'],**plan['evidence_hashes']}
    for p,h in dependencies.items():assert sha256(p)==h
    runs={n:[json.loads(Path(c[str(f)]['path']).read_text()) for f in range(3)] for n,c in plan['controls'].items()}
    runs.update({n:[json.loads((HERE/'results'/f'{n}_f{f}_s42.json').read_text()) for f in range(3)] for n in plan['configs']})
    rows=[dict(config=n,mean_inner_auc=float(np.mean([r['inner_auc'] for r in rs])),mean_inner_ap=float(np.mean([r['inner_ap'] for r in rs]))) for n,rs in runs.items()]
    assert plan['eligible_candidates']==['full'];best=next(r for r in rows if r['config']=='full')
    best['comparisons']={n:{m:dict(mean_delta=float(np.mean([r['inner_'+m]-b['inner_'+m] for r,b in zip(runs['full'],runs[n])])),
        fold_deltas=[r['inner_'+m]-b['inner_'+m] for r,b in zip(runs['full'],runs[n])]) for m in ['auc','ap']} for n in plan['comparators']}
    best['selection_score']=min(c[m]['mean_delta'] for c in best['comparisons'].values() for m in ['auc','ap'])
    promote=best['selection_score']>=plan['promotion_minimum_both_gains_over_each_control']
    selection=dict(status='frozen_inner_selection',rows=rows,best=best,promote_to_outer_validation=promote,
        plan_sha256=sha256(HERE/'results/plan.json'),verification_sha256=sha256(HERE/'verification.json'),
        result_hashes={p.name:sha256(p) for p in sorted((HERE/'results').glob('*_s42.json'))},
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    sp=HERE/'selection.json'
    if sp.exists():assert json.loads(sp.read_text())==selection
    else:write(sp,selection)
    diagnostics=[]
    for name in plan['configs']:
        for fold in range(3):
            r=runs[name][fold];seed_all(42);data=load_inner(fold);a=make_adapter(data,name,'cuda:0');m=a.model
            m.load_state_dict(torch.load(HERE/'results'/f"{r['tag']}.pt",map_location='cuda:0',weights_only=True));m.eval();f=m.feedback
            with np.load(HERE/'replay_predictions'/f"{r['tag']}.npz") as z:train=z['train']
            with torch.no_grad():
                x0,z0=f.initial(*a.dense[:2])
                x1,z1,detail=f.step(x0,z0,x0[f.neighbors],z0[f.neighbors],f.valid,f.types,True)
                x2,z2=f.full_step(x1,z1);L=f.frame(z2);singular=torch.linalg.svdvals(L)
                w=detail['weights'];C=detail['moment'];diag=torch.diag_embed(C.diagonal(dim1=-2,dim2=-1))
                output=f(*a.dense[:2],a.pairs['val']);addition=m.feedback_projection(output)
                offclip=(z2[...,:f.off].abs()>=2).float().mean();logclip=(z2[...,f.off:].abs()>=5).float().mean()
            d=dict(config=name,fold=fold,best_epoch=r['best_epoch'],epochs_run=r['epochs_run'],
                train_auc=roc_auc_score(data['train_y'],train),train_ap=average_precision_score(data['train_y'],train),
                inner_auc=r['inner_auc'],inner_ap=r['inner_ap'],
                mean_effective_incoming_events=float((1/w.square().sum(1)).mean()),
                rms_first_step_coordinates=float(z1.square().mean().sqrt()),rms_second_step_coordinate_change=float((z2-z1).square().mean().sqrt()),
                moment_offdiagonal_energy_fraction=float((C-diag).square().sum()/C.square().sum().clamp_min(1e-12)),
                max_second_step_frame_condition=float((singular[...,0]/singular[...,-1]).max()),
                mean_second_step_frame_condition=float((singular[...,0]/singular[...,-1]).mean()),
                minimum_frame_singular_value=float(singular.min()),offdiagonal_clip_fraction=float(offclip),logdiagonal_clip_fraction=float(logclip),
                val_rms_added_lin1_activation=float(addition.square().mean().sqrt()))
            assert all(np.isfinite(value) for key,value in d.items() if key not in ['config'])
            diagnostics.append(d);del a,m,f;torch.cuda.empty_cache()
    means={n:{k:float(np.mean([r[k] for r in diagnostics if r['config']==n])) for k in diagnostics[0] if k not in ['config','fold','best_epoch','epochs_run']} for n in plan['configs']}
    write(HERE/'component_diagnostics.json',dict(status='passed',selection_sha256=sha256(sp),records=diagnostics,means=means,
        graph_state_scope='Full training graph geometry; validation addition uses actual query-deleted inference. Descriptive diagnostics, not new fits or further selection.',no_further_selection=True))
    flat=[dict(config=n,fold=f,**{k:r[k] for k in ['inner_auc','inner_ap','best_epoch','epochs_run']}) for n,rs in runs.items() for f,r in enumerate(rs)]
    with (HERE/'inner_summary.csv').open('w') as out:
        writer=csv.DictWriter(out,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    original=HERE.parent/'midrug_internal_20261006/plmf_protection.json';protected=json.loads(original.read_text())
    for p,h in protected['source_hashes'].items():assert sha256(RELEASE/p)==h
    frozen=RELEASE/'data/frozen';manifest=json.loads((frozen/'manifest.json').read_text());refs=HERE.parent/'baselines_m123_20261004/results';protection=[]
    for r in protected['records']:
        ds,fold=r['dataset'],r['fold']
        assert sha256(refs/f'incrd_m123_{ds}_f{fold}_s42.pt')==r['checkpoint_sha256']
        assert sha256(frozen/f'{ds}_fold{fold}.npz')==r['split_sha256']
        assert sha256(frozen/f'{ds}_graph.pt')==manifest['datasets'][ds]['graph_sha256']
        protection.append(dict(dataset=ds,fold=fold,checkpoint_sha256=r['checkpoint_sha256'],split_sha256=r['split_sha256'],graph_sha256=manifest['datasets'][ds]['graph_sha256']))
    write(HERE/'plmf_hash_recheck.json',dict(status='passed',records=protection,source_unchanged=True,prior_full_replay_sha256=sha256(original),new_gpu_replays=0))
    table='\n'.join(f"| {r['config']} | {r['mean_inner_auc']:.9f} | {r['mean_inner_ap']:.9f} |" for r in rows)
    paired='\n'.join(f"| {ref} | {c['auc']['mean_delta']:+.9f} | {c['ap']['mean_delta']:+.9f} |" for ref,c in best['comparisons'].items())
    stages={n:[r['best_epoch'] for r in runs[n]] for n in plan['configs']}
    new=[r for r in v['records'] if not r['reused_control']]
    deleted=max(x['errors']['full_logit'] for r in new for x in r['explicit_deletions'])
    branch_error=max(r['branch']['explicit_two_step_error'] for r in new)
    grad_error=max(r['chunk']['gradient_error'] for r in new)
    diagtable='\n'.join(f"| {n} | {r['train_auc']:.6f} | {r['train_ap']:.6f} | {r['val_rms_added_lin1_activation']:.6f} | {r['mean_second_step_frame_condition']:.3f} |" for n,r in means.items())
    report=f'''# 消息残差二阶矩反馈：内部实验结果

**原测试AUC/AP同时绝对+.02尚未证实。** 本轮完成9次新拟合、9个所选检查点预测重放，
另检查9个最终状态及所选/最终优化器；六个旧强对照引用不可变完整重放。
仅原训练内部三折seed42，没有新原验证/测试评估。

| 配置 | 内部AUC | 内部AP |
|---|---:|---:|
{table}

| full的规定对照 | ΔAUC | ΔAP |
|---|---:|---:|
{paired}

唯一候选full最小规定平均增益{best['selection_score']:+.9f}；推进原验证：{promote}。
只有相对全部四个对照的三折平均AUC/AP均≥+.002才推进，不事后改选diagonal/mean。
最终仍须原测试seeds42/43/44均值AUC≥.9733929274189016、AP≥.933683373732677，PLMF两集不降。

## 方法与核验

依据[GeoF，2026-09预印本](https://arxiv.org/abs/2609.34711v1)的两轮块三角传输与反馈机制。
本地4个4维块、图无关单位frame初值、类型标识控制器，无结构atlas/度/RW初值；
这些与论文的区别见[冻结方案](README.md)。保留完整原相似关系和训练正关联的双向边。
二阶矩进入下一轮坐标、注意力和消息，区别于旧路径矩只进入读出头。

三个臂均从每折原unmasked10所选检查点恢复，冻结原M123/重构的全部参数和缓冲。
新增投影初始为零，stage0恢复较强源模型；所选阶段为{stages}。
只训练新反馈及投影，BCE、RMSprop lr1e-4/alpha.99/wd1e-4、set_to_none=True。
最高300步、至少100步、patience100，所选和最终原参数/BN逐位与源一致。

独立NumPy逐邻居计算、有限差分和所有参数物理删图梯度通过；同均值同对角合成例区分完整二阶矩。
两轮精确删边重新计算两端点的第一轮状态和frame、第二轮全部入消息及反馈；图中不存在的查询
只共享等价完整图计算，正查询256分块，没有查询批次归一化或邻居截断。
真实预检的分块梯度检查曾使用Torch1.13不支持的checkpoint+autograd.grad接口，
仅检查器改用backward后通过；原失败源码/日志保留在precheck_failures。训练一直使用backward。

九模型重放中完整M1/M2/M3最终logit物理删查询最大误差{deleted:.3g}；
反馈分支与显式两轮删图最大误差{branch_error:.3g}；不同分块参数梯度最大误差{grad_error:.3g}。
选轮、曲线、预测、优化器各参数实际步数、正负查询及移除重构推理等价均核验。

## 冻结选择后的诊断

| 配置 | 训练AUC | 训练AP | 留出lin1新增激活RMS | 完整训练图frame平均条件数 |
|---|---:|---:|---:|---:|
{diagtable}

完整训练图的坐标变化、有效入邻居数、矩非对角能量与clip比例见component_diagnostics.json。
这些是所选检查点诊断，不是原验证结果或新的训练消融，也不用于重新选择温度/混合权重/轮数。
正定frame不意味着训练达成论文条件收缩保证，非零表示变化也不保证排序提高。

没有lncRNA、外部数据、注释、额外序列或新权重。PLMF十个受保护旧检查点、冻结输入和
发布源码哈希保持不变；本轮没有新增PLMF训练或GPU重放。
'''
    (HERE/'REPORT.md').write_text(report)
    state=dict(goal_status='active',completion_proven=False,stage='completed_and_audited',best=best,promote_to_outer_validation=promote,
        new_fits=9,new_checkpoint_replays=9,final_state_inspections=9,reused_control_checks=6,no_lncrna=True,new_candidate_test_evaluations=0,
        processes_expected_live=[],component_diagnostics=means,next_action='Use the frozen full/diagonal/mean results and geometry diagnostics to decide the next distinct hypothesis; do not retrospectively sweep feedback weights, temperature, block size or depth.')
    write(HERE/'CONTINUATION.json',state)
    progress=HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md';prefix=progress.read_text().split('最新完成')[0]
    progress.write_text(prefix+f'''最新完成[消息残差二阶矩反馈](midrug_feedback_geometry_20261007/REPORT.md)：9次新训练与9个所选检查点预测重放，最终状态和优化器另行检查。
full内部AUC/AP为{best['mean_inner_auc']:.9f}/{best['mean_inner_ap']:.9f}，推进原验证：{promote}。
[前轮折外案例对照](midrug_retrieval_crossfit_20261007/REPORT.md)和[近期文献核查](midrug_post_crossfit_literature_20261007/READING_NOTES.md)保留。
没有新原验证/测试成绩；PLMF十个旧模型及冻结输入/发布源码哈希未变。
''')
    snapshot=HERE/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,snapshot/p.name)
    files=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'source_snapshot' not in p.parts
        and p.name not in ['archive_manifest.json','archive_recheck.json','summarize.log']]
    archive=dict(status='archived',source_hashes={p.name:sha256(p) for p in sorted(HERE.glob('*.py'))},
        artifact_hashes={str(p.relative_to(HERE)):sha256(p) for p in sorted(files)},frozen_dependencies=dependencies)
    write(HERE/'archive_manifest.json',archive);count=0
    for p,h in archive['source_hashes'].items():assert sha256(HERE/p)==h and sha256(snapshot/p)==h;count+=2
    for p,h in archive['artifact_hashes'].items():assert sha256(HERE/p)==h;count+=1
    for p,h in archive['frozen_dependencies'].items():assert sha256(p)==h;count+=1
    write(HERE/'archive_recheck.json',dict(status='passed',hash_checks=count,new_checkpoint_replays=9,final_state_inspections=9,reused_control_checks=6,
        protected_plmf_records=10,selection_unchanged=True,archive_sha256=sha256(HERE/'archive_manifest.json'),selection_sha256=sha256(sp),
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False))
    scopepath=HERE.parent/'ACTIVE_SCOPE.json';scope=json.loads(scopepath.read_text());scope.update(active_directory=str(HERE),
        latest_continuation=str(HERE/'CONTINUATION.json'),latest_archive_check=str(HERE/'archive_recheck.json'),next_action=state['next_action']);write(scopepath,scope)
    print(json.dumps(dict(best=best,promote=promote,diagnostics=means),indent=2))

if __name__=='__main__':main()
