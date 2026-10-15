"""Summarize fixed-holdout learning curves without selecting a new candidate."""
import csv,json,shutil
from pathlib import Path
import numpy as np
from common import HERE,RELEASE,sha256
from model import CONFIGS,SIZES


def main():
    plan=json.loads((HERE/'results/plan.json').read_text());v=json.loads((HERE/'verification.json').read_text())
    assert v['status']=='passed' and len(v['records'])==18 and v['total_checkpoint_replays']==18
    for path,digest in {**plan['source_hashes'],**plan['evidence_hashes']}.items():assert sha256(path)==digest
    rows=v['records'];sizes={**SIZES,'full':1.};means=[];gains=[];increments=[]
    for size,fraction in sizes.items():
        for config in CONFIGS:
            records=sorted([r for r in rows if r['size']==size and r['config']==config],key=lambda r:r['fold'])
            assert len(records)==3
            means.append(dict(config=config,size=size,fraction=fraction,mean_train_count=float(np.mean([r['train_count'] for r in records])),
                train={k:float(np.mean([r['train_metrics'][k] for r in records])) for k in ['auc','ap']},
                val={k:float(np.mean([r['val_metrics'][k] for r in records])) for k in ['auc','ap']},
                train_minus_val={k:float(np.mean([r['train_minus_val'][k] for r in records])) for k in ['auc','ap']},
                mean_train_bce=float(np.mean([r['train_bce'] for r in records]))))
        by_config={c:sorted([r for r in rows if r['size']==size and r['config']==c],key=lambda r:r['fold']) for c in CONFIGS}
        gain={metric:[a['val_metrics'][metric]-b['val_metrics'][metric] for a,b in zip(by_config['unmasked10'],by_config['control'])] for metric in ['auc','ap']}
        gains.append(dict(size=size,fraction=fraction,mean={k:float(np.mean(values)) for k,values in gain.items()},fold_deltas=gain))
    for name in CONFIGS:
        for smaller,larger in [('half','three_quarter'),('three_quarter','full')]:
            left=sorted([r for r in rows if r['size']==smaller and r['config']==name],key=lambda r:r['fold'])
            right=sorted([r for r in rows if r['size']==larger and r['config']==name],key=lambda r:r['fold'])
            change={metric:[a['val_metrics'][metric]-b['val_metrics'][metric] for a,b in zip(right,left)] for metric in ['auc','ap']}
            increments.append(dict(config=name,from_size=smaller,to_size=larger,
                mean={k:float(np.mean(values)) for k,values in change.items()},fold_deltas=change))
    assessment=dict(status='frozen_diagnostic_assessment',means=means,reconstruction_gains=gains,successive_size_increments=increments,
        plan_sha256=sha256(HERE/'results/plan.json'),verification_sha256=sha256(HERE/'verification.json'),
        point_metadata_hashes={r['metadata_path']:r['metadata_sha256'] for r in rows},
        no_candidate_selection=True,no_performance_extrapolation=True,no_automatic_outer_promotion=True,
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False,goal_achieved=False)
    ap=HERE/'assessment.json'
    if ap.exists():assert json.loads(ap.read_text())==assessment
    else:ap.write_text(json.dumps(assessment,indent=2))
    flat=[dict(config=r['config'],size=r['size'],fold=r['fold'],train_count=r['train_count'],best_epoch=r['best_epoch'],
               train_auc=r['train_metrics']['auc'],train_ap=r['train_metrics']['ap'],val_auc=r['val_metrics']['auc'],val_ap=r['val_metrics']['ap']) for r in rows]
    with (HERE/'points.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    original=HERE.parent/'midrug_internal_20261006/plmf_protection.json';protected=json.loads(original.read_text())
    for path,digest in protected['source_hashes'].items():assert sha256(RELEASE/path)==digest
    frozen=RELEASE/'data/frozen';manifest=json.loads((frozen/'manifest.json').read_text());refs=HERE.parent/'baselines_m123_20261004/results';protection=[]
    for r in protected['records']:
        ds,fold=r['dataset'],r['fold'];assert sha256(refs/f'incrd_m123_{ds}_f{fold}_s42.pt')==r['checkpoint_sha256']
        assert sha256(frozen/f'{ds}_fold{fold}.npz')==r['split_sha256']
        assert sha256(frozen/f'{ds}_graph.pt')==manifest['datasets'][ds]['graph_sha256']
        protection.append(dict(dataset=ds,fold=fold,checkpoint_sha256=r['checkpoint_sha256'],split_sha256=r['split_sha256'],graph_sha256=manifest['datasets'][ds]['graph_sha256']))
    (HERE/'plmf_hash_recheck.json').write_text(json.dumps(dict(status='passed',records=protection,source_unchanged=True,prior_full_replay_sha256=sha256(original)),indent=2))
    table='\n'.join(f"| {r['size']} | {r['mean_train_count']:.1f} | {r['config']} | {r['val']['auc']:.9f} | {r['val']['ap']:.9f} | {r['train_minus_val']['auc']:+.6f}/{r['train_minus_val']['ap']:+.6f} |" for r in means)
    gain_table='\n'.join(f"| {r['size']} | {r['mean']['auc']:+.9f} | {r['mean']['ap']:+.9f} |" for r in gains)
    increase_table='\n'.join(f"| {r['config']} | {r['from_size']} → {r['to_size']} | {r['mean']['auc']:+.9f} | {r['mean']['ap']:+.9f} |" for r in increments)
    error=max(x['errors']['full_logit'] for r in rows for x in r['explicit_deletions'])
    report=f'''# 固定内部留出集的训练量诊断

**最终原测试 AUC/AP 均绝对 +.02 尚未证实。** 完成12次新拟合，并重放全部18个
检查点（12个新子集模型、6个既有100%模型）。没有原验证/测试或新候选晋级。

| 内部训练量 | 平均训练配对数 | 模型 | 留出 AUC | 留出 AP | 训练减留出 AUC/AP |
|---|---:|---|---:|---:|---:|
{table}

100%是各内部折的完整训练部分，约6474条，不是全部9712条原训练配对。
所有训练量在同一个内部折上使用完全相同留出配对；50%嵌套于75%，75%嵌套于100%。

| 训练量 | 重建减纯 M123 的 ΔAUC | ΔAP |
|---|---:|---:|
{gain_table}

| 模型 | 训练量变化 | 留出 ΔAUC | ΔAP |
|---|---|---:|---:|
{increase_table}

相同训练量下的模型差值才是这里的方法收益。增加内部子集训练量得到的上升，
不能计为相对原始 iNcRD-HG 的提升，因为原基准也使用完整原训练集。
这些是同一数据池内重叠子集、同一训练种子的诊断，不是独立生物验证；
不从三个点拟合达到最终目标的外推保证，也不据此改写旧模型未通过的门槛。
完整逐折结果见 points.csv 和 assessment.json，保留所有反向变化。

## 实现与核验

每类按固定私有随机流排序，取50%/75%前缀，再恢复原训练顺序。
未选中配对不进入标签监督或关联图，未被改成负例。所有原节点属性和同类图保留，
每个子集重新计算其训练关联图、M2/M3、EASE和配对标准化。
主干、优化器、重建目标、选轮规则与100%旧来源一致；源文件哈希已核对。

12个模型/子集组合均检查了支持图精确成员、留出互斥与嵌套、原初始化、
有限共享编码器梯度、BN一次及完整正/负查询物理删除，留出标签翻转不改变预测。
18个检查点核对保存概率/指标/选轮及对应训练量；新曲线核对损失构成。
全部训练指标由选中检查点在精确删查询表示上重算，不是保留自身正边的训练预测。
最大完整删查询logit误差 {error:.3g}，去除重建模块保持预测等价。

仅原miRNA/drug输入，无lncRNA、外部数据、注释或新权重。PLMF10个受保护检查点、
冻结图/划分及发布源码哈希通过；两个数据集保持已有完整重放的结果。
原测试三种子均值最终仍须AUC≥.9733929274、AP≥.9336833737。
已验证纯miRNA M123+BPR为.9575892678/.9233078960，本轮没有新增测试成绩。
'''
    (HERE/'REPORT.md').write_text(report)
    state=dict(goal_status='active',completion_proven=False,stage='completed_and_audited',
        current_turn_classification='progress:12 nested-size fits and18 checkpoint replays with fixed-heldout learning curves',
        new_fits=12,total_checkpoint_replays=18,means=means,reconstruction_gains=gains,successive_size_increments=increments,
        no_lncrna=True,new_candidate_test_evaluations=0,processes_expected_live=[],
        next_action='Interpret same-size reconstruction gains and data-scale increments; do not infer final-test improvement or relax the prior failed candidate gates.')
    (HERE/'CONTINUATION.json').write_text(json.dumps(state,indent=2))
    progress=HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md';prefix=progress.read_text().split('最新完成')[0]
    progress.write_text(prefix+'''最新完成[固定内部留出集的训练量诊断](midrug_training_size_20261007/REPORT.md)：12次新拟合，18个检查点重放。
比较约3237/4855/6474条内部训练配对的纯M123与固定重建正则化；完整逐折结果已归档。
这是数据量诊断，没有新增原验证/测试成绩或候选晋级；最终双+.02目标未完成。
PLMF10个检查点、冻结输入与发布源码哈希保护通过。
''')
    snapshot=HERE/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for path in HERE.glob('*.py'):shutil.copy2(path,snapshot/path.name)
    artifacts=['README.md','subsets.npz','subset_manifest.json','initial_checks.json','verification.json','assessment.json','points.csv',
        'REPORT.md','CONTINUATION.json','plmf_hash_recheck.json','results/plan.json','prepare.log','check.log','launch.log','audit.log']
    artifacts+=sorted(str(p.relative_to(HERE)) for p in (HERE/'results').iterdir() if p.suffix in {'.pt','.npz','.json'} and p.name!='plan.json')
    artifacts+=sorted(str(p.relative_to(HERE)) for p in (HERE/'replay_predictions').glob('*.npz'))
    (HERE/'archive_manifest.json').write_text(json.dumps(dict(status='archived',source_hashes={p.name:sha256(p) for p in sorted(HERE.glob('*.py'))},
        artifact_hashes={p:sha256(HERE/p) for p in artifacts},frozen_dependencies={**plan['source_hashes'],**plan['evidence_hashes']}),indent=2))
    print(json.dumps(dict(means=means,reconstruction_gains=gains,successive_size_increments=increments,maximum_deletion_logit_error=error),indent=2))

if __name__=='__main__':main()
