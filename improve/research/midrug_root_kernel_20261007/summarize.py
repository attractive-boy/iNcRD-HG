"""Freeze the only eligible candidate before diagnostics and archival."""
import csv,json,shutil
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,RELEASE,load_inner,seed_all,sha256,base
from model import make_adapter,source_record,initialize

def write(p,x):p.write_text(json.dumps(x,indent=2))

def main():
    plan=json.loads((HERE/'results/plan.json').read_text());v=json.loads((HERE/'verification.json').read_text())
    assert v['status']=='passed' and v['new_checkpoint_replays']==9 and v['final_state_inspections']==9
    assert len(v['records'])==21 and v['immutable_reused_control_checks']==12
    dependencies={**plan['source_hashes'],**plan['evidence_hashes']}
    for p,h in dependencies.items():assert sha256(p)==h
    runs={n:[json.loads(Path(c[str(f)]['path']).read_text()) for f in range(3)] for n,c in plan['controls'].items()}
    runs.update({n:[json.loads((HERE/'results'/f'{n}_f{f}_s42.json').read_text()) for f in range(3)] for n in plan['configs']})
    rows=[dict(config=n,mean_inner_auc=float(np.mean([r['inner_auc'] for r in rs])),mean_inner_ap=float(np.mean([r['inner_ap'] for r in rs]))) for n,rs in runs.items()]
    assert plan['eligible_candidates']==['root_path'];best=next(r for r in rows if r['config']=='root_path')
    best['comparisons']={n:{m:dict(mean_delta=float(np.mean([r['inner_'+m]-b['inner_'+m] for r,b in zip(runs['root_path'],runs[n])])),
        fold_deltas=[r['inner_'+m]-b['inner_'+m] for r,b in zip(runs['root_path'],runs[n])]) for m in ['auc','ap']} for n in plan['comparators']}
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
            r=runs[name][fold];seed_all(42);data=load_inner(fold);a,source,_=initialize(data,name,fold,'cuda:0');m=a.model
            initial={k:v.clone() for k,v in m.subgraph.state_dict().items()}
            m.load_state_dict(torch.load(HERE/'results'/f"{r['tag']}.pt",map_location='cuda:0',weights_only=True));m.eval()
            with np.load(HERE/'replay_predictions'/f"{r['tag']}.npz") as z:train,val=z['train'],z['val']
            with torch.no_grad():
                batch=m.cache.select(*a.pairs['val'])
                if name=='mpnn':batch={**batch,'neg':torch.zeros_like(batch['pos'])}
                branch=m.subgraph(batch,*a.dense[:2])
                plain=base.Model.forward(m,a.pairs['val'],a.ps,a.qis['val'],a.ease['val'],a.dense)
                combined=(plain+branch).sigmoid().cpu().numpy();plain_p=plain.sigmoid().cpu().numpy()
            np.testing.assert_allclose(combined,val,atol=2e-5,rtol=2e-5)
            with np.load(source.with_name(source.stem+'_inner.npz')) as z:original=z['probability']
            np.testing.assert_allclose(plain_p,original,atol=2e-5,rtol=2e-5)
            curve=json.loads((HERE/'results'/f"{r['tag']}_curves.json").read_text())
            d=dict(config=name,fold=fold,best_epoch=r['best_epoch'],epochs_run=r['epochs_run'],
                train_auc=roc_auc_score(data['train_y'],train),train_ap=average_precision_score(data['train_y'],train),
                final_inner_auc_from_curve=curve[-1]['inner_auc'],final_inner_ap_from_curve=curve[-1]['inner_ap'],
                val_branch_logit_rms=float(branch.square().mean().sqrt()),val_branch_logit_std=float(branch.std(unbiased=False)),val_branch_logit_mean=float(branch.mean()),without_branch_source_error=float(abs(plain_p-original).max()))
            if name!='mpnn':
                with torch.no_grad():
                    all_features=m.subgraph.features(batch,*a.dense[:2]);features=all_features[:,:32].view(-1,8,4)
                    d.update(root_feature_mean=all_features[:,32:].mean(0).cpu().tolist(),root_feature_std=all_features[:,32:].std(0).cpu().tolist(),root_columns_norm=float(m.subgraph.head[0].weight[:,32:].norm()))
                    erased=all_features.clone();erased[:,32:]=0;without_root=m.subgraph.head(erased).view(-1)
                    d['root_erasure_logit_rms']=float((branch-without_root).square().mean().sqrt())
                    final=torch.load(HERE/'results'/f"{r['tag']}_final.pt",map_location='cuda:0',weights_only=True)
                    d.update(kernel_mean_by_step=features.mean((0,1)).cpu().tolist(),kernel_std_by_step=features.std((0,1)).cpu().tolist(),
                        selected_edge_raw_change=float((m.subgraph.edge_raw-initial['edge_raw']).norm()),
                        final_edge_raw_change=float((final['subgraph.edge_raw']-initial['edge_raw']).norm()))
            diagnostics.append(d);del a,m,initial;torch.cuda.empty_cache()
    metrics=['train_auc','train_ap','val_branch_logit_rms','val_branch_logit_std','val_branch_logit_mean','final_inner_auc_from_curve','final_inner_ap_from_curve']
    means={n:{k:float(np.mean([r[k] for r in diagnostics if r['config']==n])) for k in metrics} for n in plan['configs']}
    write(HERE/'component_diagnostics.json',dict(status='passed',selection_sha256=sha256(sp),records=diagnostics,means=means,
        scope='Postselection diagnostics only; final metrics from frozen curves, final weights inspected but not full prediction replays.',no_further_selection=True))
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
    diagtable='\n'.join(f"| {n} | {r['train_auc']:.6f}/{r['train_ap']:.6f} | {r['final_inner_auc_from_curve']:.6f}/{r['final_inner_ap_from_curve']:.6f} | {r['val_branch_logit_rms']:.6f} |" for n,r in means.items())
    (HERE/'REPORT.md').write_text(f'''# 全局图核加双根路径属性均值：内部结果

**原测试AUC/AP同时绝对+.02尚未证实。** 完成9次新拟合及9个所选模型预测重放，
另检查9个最终状态及所选/最终优化器，核对12个旧强对照的不可变完整重放。
只有原训练内部三折seed42，没有新原验证/测试评估。

| 配置 | 内部AUC | 内部AP |
|---|---:|---:|
{table}

| root_path的规定对照 | ΔAUC | ΔAP |
|---|---:|---:|
{paired}

唯一候选root_path最小规定平均增益{best['selection_score']:+.9f}；推进原验证：{promote}。
必须对全部六个对照的三折平均AUC/AP均≥+.002，不事后改选对照或核归一化方式。
最终仍须原测试三种子均值AUC≥.9733929274189016、AP≥.933683373732677，PLMF两集不降。

## 方法及审计

前轮[冻结图核诊断](../midrug_walk_kernel_diagnostics_20261007/REPORT.md)发现新增logit
以整体偏置为主；本轮检验明确的双根读出，不预设它一定有效。保持原32维全局pathwise核，
新增8维双根3步路径的属性匹配均值，除以同路径在属性匹配恒1时的质量。
root_path含两端及中间节点匹配；root_endpoints只匹配两端；global_only追加8维零输入。
无支撑时新通道为零，对分母保留隐藏图梯度；未改原型数、核步数、邻居数或子图抽取。

局部分支只用删查询的正MiDrug双向边，M123主干仍保留完整相似图。
每折从同一unmasked10源模型及新核初始化开始，原全部参数/缓冲冻结。
三个臂第一层共同扩为40维，前32列初始化保留、新列零，末层零；global_only有效容量较小。
BCE，RMSprop lr1e-4/alpha.99/wd1e-4，set_to_none=True；max300/min100/patience100，
含stage0的最早最大双指标均值选轮。所选轮次：{stages}。
源模型已在同一内部留出选轮，本轮不是独立验证。

独立逐路径枚举、商式梯度、端点对照的中间属性零梯度及无路径零值通过。
真实正负查询缓存与物理删记录后重建逐项一致，查询批次/节点置换/留出标签翻转通过。
所有所选和最终原参数及BN逐位等于源，global_only新增列始终为零。
预测、选轮、最终状态、逐参数优化器实际步数以及完整M1/M2/M3加分支物理删图通过，
最大最终logit删除误差{deleted:.3g}。移除冻结重构模块保持预测等价。

## 选型冻结后的诊断

| 配置 | 所选训练AUC/AP | 最后一步内部AUC/AP | 所选分支logit RMS |
|---|---:|---:|---:|
{diagtable}

新增根通道均值/标准差、头新增列范数、分支均值/标准差及固定所选权重下擦除根通道
的logit变化见component_diagnostics.json。均为选型冻结后的描述，无新系数搜索或重选。
末步内部指标来自冻结曲线，不计为额外完整预测重放。原整体偏置不改变精确排序，
但不能据此把本轮所有训练变化归为校准；也不从固定协议的负结果推断方法家族上限。

没有lncRNA、外部数据、注释、额外序列或新权重。PLMF十个旧检查点、原输入和发布源码
哈希保持不变；本轮没有新PLMF训练或GPU重放。
''')
    state=dict(goal_status='active',completion_proven=False,stage='completed_and_audited',best=best,promote_to_outer_validation=promote,
        new_fits=9,new_checkpoint_replays=9,final_state_inspections=9,reused_control_checks=12,no_lncrna=True,new_candidate_test_evaluations=0,
        processes_expected_live=[],component_diagnostics=means,next_action='Use the frozen rooted-path versus endpoint-only/global controls and branch-variation diagnostics to choose the next distinct hypothesis. Do not retrospectively select controls or sweep kernel steps, prototype counts, pooling coefficients or extraction sizes.')
    write(HERE/'CONTINUATION.json',state)
    progress=HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md';prefix=progress.read_text().split('最新完成')[0]
    progress.write_text(prefix+f'''最新完成[全局图核加双根路径属性均值](midrug_root_kernel_20261007/REPORT.md)：9次新拟合与所选模型重放，最终状态和优化器另查。
root_path内部AUC/AP为{best['mean_inner_auc']:.9f}/{best['mean_inner_ap']:.9f}，推进原验证：{promote}。
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
    write(HERE/'archive_recheck.json',dict(status='passed',hash_checks=count,new_checkpoint_replays=9,final_state_inspections=9,reused_control_checks=12,
        protected_plmf_records=10,selection_unchanged=True,archive_sha256=sha256(HERE/'archive_manifest.json'),selection_sha256=sha256(sp),
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False))
    scopepath=HERE.parent/'ACTIVE_SCOPE.json';scope=json.loads(scopepath.read_text());scope.update(active_directory=str(HERE),
        latest_continuation=str(HERE/'CONTINUATION.json'),latest_archive_check=str(HERE/'archive_recheck.json'),
        latest_experiment_archive_check=str(HERE/'archive_recheck.json'),latest_research_review=str(HERE/'README.md'),
        latest_research_continuation=str(HERE/'CONTINUATION.json'),next_action=state['next_action']);write(scopepath,scope)
    print(json.dumps(dict(best=best,promote=promote,diagnostics=means),indent=2))

if __name__=='__main__':main()
