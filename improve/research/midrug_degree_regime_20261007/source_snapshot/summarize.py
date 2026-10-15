"""Freeze the conditional-residual gate and archive with PLMF hash protection."""
import csv
import json
import shutil
from pathlib import Path
import numpy as np
from common import HERE,RELEASE,sha256


def main():
    plan=json.loads((HERE/'results/plan.json').read_text());v=json.loads((HERE/'verification.json').read_text())
    assert v['status']=='passed' and len(v['records'])==15 and v['new_checkpoint_replays']==9
    assert v['immutable_reused_control_checks']==6
    for p,h in {**plan['source_hashes'],**plan['evidence_hashes']}.items():assert sha256(p)==h
    runs={n:[json.loads(Path(c[str(f)]['path']).read_text()) for f in range(3)] for n,c in plan['controls'].items()}
    for name in plan['configs']:runs[name]=[json.loads((HERE/'results'/f'{name}_f{f}_s42.json').read_text()) for f in range(3)]
    rows=[];flat=[]
    for name,rs in runs.items():
        comparisons={ref:{m:dict(mean_delta=float(np.mean([r['inner_'+m]-b['inner_'+m] for r,b in zip(rs,runs[ref])])),
            fold_deltas=[r['inner_'+m]-b['inner_'+m] for r,b in zip(rs,runs[ref])]) for m in ['auc','ap']}
            for ref in plan['comparators'].get(name,['control','unmasked10'])}
        rows.append(dict(config=name,mean_inner_auc=float(np.mean([r['inner_auc'] for r in rs])),mean_inner_ap=float(np.mean([r['inner_ap'] for r in rs])),
            comparisons=comparisons,selection_score=min(c[m]['mean_delta'] for c in comparisons.values() for m in ['auc','ap'])))
        flat.extend(dict(config=name,fold=f,**{k:r[k] for k in ['inner_auc','inner_ap','best_epoch','epochs_run']}) for f,r in enumerate(rs))
    best=next(r for r in rows if r['config']=='degree');promote=best['selection_score']>=plan['promotion_minimum_both_gains_over_each_control']
    selection=dict(status='frozen_inner_selection',rows=rows,best=best,promote_to_outer_validation=promote,
        plan_sha256=sha256(HERE/'results/plan.json'),verification_sha256=sha256(HERE/'verification.json'),
        result_hashes={p.name:sha256(p) for p in sorted((HERE/'results').glob('*_s42.json'))},
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    sp=HERE/'selection.json'
    if sp.exists():assert json.loads(sp.read_text())==selection
    else:sp.write_text(json.dumps(selection,indent=2))
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
    new=[r for r in v['records'] if not r['reused_control']];deleted=max(x['errors']['full_logit'] for r in new for x in r['explicit_deletions'])
    report=f'''# 支持度条件残差专家：内部检验

**原测试AUC/AP同时绝对+.02尚未证实。** 九次新拟合及选中模型重放；六个既有
纯M123/unmasked10对照复用已有完整重放，当前核对元数据、权重和保存概率/标签。
仅原训练内部三折seed42，没有新候选原验证/测试评估。

| 配置 | 内部 AUC | 内部 AP |
|---|---:|---:|
{table}

| degree的规定对照 | ΔAUC | ΔAP |
|---|---:|---:|
{paired}

唯一候选degree的最小规定平均增益为{best['selection_score']:+.9f}；所有对照的
平均AUC/AP增益均须≥.002，推进原验证：{promote}。随机路由与均匀路由仅为对照，
不事后改选。最终仍需原测试seed42/43/44均值AUC≥.9733929274、AP≥.9336833737，
且PLMF MDR/MDS不下降。

## 机制与依据

支持度预检的四组阈值仅来自当前训练查询删边后的度中位数。低/低组每折只有
12–21个训练正例；约60%的AUC逆序出现在跨组比较。预检平分同分AUC逆序，
按同分阈值末端precision分解AP缺口，两者均精确重构全局误差。分组难度差异
只支持进一步检验，不证明可消除错误。原M3已有这些度信息，没有增加新数据。

所有配对仍用同一原M123主干，附加四个D→64→1残差专家，输入是无参数LayerNorm
后的原融合向量，末层零初始化。degree按支持度路由；random用固定ID哈希，
训练组人数逐组匹配degree；uniform对相同四路取等权。random的类别数和留出组
人数不保证相同，不能将其差异单独解释为生物学机制。
统一RMSprop，主干/重构lr1e-4、专家lr1e-3，重构/专家wd1e-4，原重构权重10。
相同主干、重构和专家初始化，原M1/M2/M3和查询删除规则保留；没有按组改标签权重。
专家混合家族已有历史GraphMETRO实现，本轮未调用旧混合RNA入口，非首次使用该家族。

## 核验

三折三配置初始恢复原预测；门控与独立NumPy删边计数/哈希分桶一致，随机与度
训练组人数逐元素相同。两次更新梯度有限、各专家参与训练、BN每步只增加一次；
留出标签翻转不变。训练后九个checkpoint验证概率/选轮/损失/阈值/路由、完整
正负查询物理删除、单查询批次不变性、移除重构模块预测等价。最大完整logit
删边误差{deleted:.3g}。分组阈值是训练后冻结常量，删查询重建图不重新训练阈值。

仅原miRNA/drug输入，无lncRNA、外部数据/注释或新权重。PLMF十个模型、冻结
图/划分、发布源码保持既有完整重放时的哈希。专家属于推理结构，不能等价删除。
'''
    state=dict(goal_status='active',completion_proven=False,stage='completed_and_audited',best=best,
        promote_to_outer_validation=promote,new_fits=9,new_checkpoint_replays=9,reused_control_checks=6,
        no_lncrna=True,new_candidate_test_evaluations=0,processes_expected_live=[],
        next_action='Use frozen residual/routing diagnostics to choose a distinct next hypothesis; no retrospective random-arm promotion or group/width/lr sweep.')
    diag=HERE/'component_diagnostics.json'
    if diag.exists():
        d=json.loads(diag.read_text());assert d['status']=='passed' and d['selection_sha256']==sha256(sp)
        state['component_diagnostics']=dict(status='passed',means=d['means'],selection_unchanged=True)
        report+='\n冻结后的[专家残差干预诊断](COMPONENT_DIAGNOSTICS.md)已经完成，不用于重新选择分支。\n'
    (HERE/'REPORT.md').write_text(report);(HERE/'CONTINUATION.json').write_text(json.dumps(state,indent=2))
    progress=HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md';prefix=progress.read_text().split('最新完成')[0]
    progress.write_text(prefix+f'''最新完成[支持度条件残差专家](midrug_degree_regime_20261007/REPORT.md)九次新训练及模型重放。
degree内部AUC/AP为{best['mean_inner_auc']:.9f}/{best['mean_inner_ap']:.9f}，推进原验证：{promote}。
前轮[固定参数平均](midrug_weight_average_20261007/REPORT.md)、[近期文献核查](midrug_finetune_literature_20261007/READING_NOTES.md)保留。
没有新增原验证/测试成绩；PLMF十个模型、冻结输入和发布源码保持哈希。
''')
    snap=HERE/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,snap/p.name)
    paths=['verification.json','selection.json','REPORT.md','CONTINUATION.json','plmf_hash_recheck.json','results/plan.json',
        'initial_checks.json','README.md','inner_summary.csv','check.log','launch.log','audit.log','PRECHECK_PLAN.md','precheck.json','precheck.log']
    if diag.exists():paths+=['component_diagnostics.json','COMPONENT_DIAGNOSTICS.md','diagnose.log']
    paths+=sorted(str(p.relative_to(HERE)) for p in (HERE/'results').iterdir() if p.suffix in {'.pt','.npz','.json'} and p.name!='plan.json')
    precheck=json.loads((HERE/'precheck.json').read_text())
    (HERE/'archive_manifest.json').write_text(json.dumps(dict(status='archived',source_hashes={p.name:sha256(p) for p in sorted(HERE.glob('*.py'))},
        artifact_hashes={p:sha256(HERE/p) for p in paths},frozen_dependencies={**plan['source_hashes'],**plan['evidence_hashes'],**precheck['source_hashes']}),indent=2))
    print(json.dumps(dict(best=best,promote=promote,new_fits=9,new_checkpoint_replays=9),indent=2))


if __name__=='__main__':main()
