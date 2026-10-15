"""Apply the frozen paired-comparator rule and archive the completed experiment."""
import csv,json,shutil
from pathlib import Path
import numpy as np
from common import HERE,RELEASE,sha256


def main():
    plan=json.loads((HERE/'results/plan.json').read_text());audit=json.loads((HERE/'verification.json').read_text())
    assert audit['status']=='passed' and len(audit['records'])==30
    assert all(r['status']=='passed' for r in audit['records'])
    bound=json.loads((HERE/'shared_error_bound.json').read_text())
    assert bound['source_sha256']==sha256(HERE/'shared_errors.py')
    for p,h in bound['prediction_hashes'].items():assert sha256(p)==h
    for p,h in plan['source_hashes'].items():assert sha256(p)==h
    runs={name:[json.loads(Path(c[str(f)]['path']).read_text()) for f in range(3)] for name,c in plan['controls'].items()}
    for name in plan['configs']:runs[name]=[json.loads((HERE/'results'/f'{name}_f{f}_s42.json').read_text()) for f in range(3)]
    rows=[];flat=[]
    for name,rs in runs.items():
        comparisons={}
        for ref in plan['comparators'].get(name,['control','unmasked10']):
            comparisons[ref]={}
            for metric in ['auc','ap']:
                delta=[r['inner_'+metric]-b['inner_'+metric] for r,b in zip(rs,runs[ref])]
                comparisons[ref][metric]=dict(mean_delta=float(np.mean(delta)),fold_deltas=delta)
        rows.append(dict(config=name,mean_inner_auc=float(np.mean([r['inner_auc'] for r in rs])),
            mean_inner_ap=float(np.mean([r['inner_ap'] for r in rs])),comparisons=comparisons,
            selection_score=min(comparisons[c][m]['mean_delta'] for c in comparisons for m in ['auc','ap'])))
        flat += [dict(config=name,fold=f,**{k:r[k] for k in ['inner_auc','inner_ap','best_epoch']}) for f,r in enumerate(rs)]
    best=max((r for r in rows if r['config'] in plan['configs']),key=lambda r:r['selection_score'])
    promote=best['selection_score']>=plan['promotion_minimum_both_gains_over_each_control']
    selection=dict(status='frozen_inner_selection',rows=rows,best=best,
        promote_to_outer_validation=promote,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False,
        plan_sha256=sha256(HERE/'results/plan.json'),verification_sha256=sha256(HERE/'verification.json'),
        result_hashes={p.name:sha256(p) for p in sorted((HERE/'results').glob('*_s42.json'))})
    (HERE/'selection.json').write_text(json.dumps(selection,indent=2))
    with (HERE/'inner_summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    original=HERE.parent/'midrug_internal_20261006/plmf_protection.json';protected=json.loads(original.read_text())
    for p,h in protected['source_hashes'].items():assert sha256(RELEASE/p)==h
    frozen=RELEASE/'data/frozen';manifest=json.loads((frozen/'manifest.json').read_text());refs=HERE.parent/'baselines_m123_20261004/results'
    protection=[]
    for r in protected['records']:
        ds,f=r['dataset'],r['fold']
        assert sha256(refs/f'incrd_m123_{ds}_f{f}_s42.pt')==r['checkpoint_sha256']
        assert sha256(frozen/f'{ds}_fold{f}.npz')==r['split_sha256']
        assert sha256(frozen/f'{ds}_graph.pt')==manifest['datasets'][ds]['graph_sha256']
        protection.append(dict(dataset=ds,fold=f,checkpoint_sha256=r['checkpoint_sha256'],split_sha256=r['split_sha256'],graph_sha256=manifest['datasets'][ds]['graph_sha256']))
    (HERE/'plmf_hash_recheck.json').write_text(json.dumps(dict(status='passed',records=protection,
        prior_full_replay_sha256=sha256(original),source_unchanged=True),indent=2))
    table='\n'.join(f"| {r['config']} | {r['mean_inner_auc']:.9f} | {r['mean_inner_ap']:.9f} | "
        f"{r['comparisons']['control']['auc']['mean_delta']:+.9f} | {r['comparisons']['control']['ap']['mean_delta']:+.9f} | "
        f"{r['comparisons']['unmasked10']['auc']['mean_delta']:+.9f} | {r['comparisons']['unmasked10']['ap']['mean_delta']:+.9f} |" for r in rows)
    matched=[]
    for r in rows:
        for c,v in r['comparisons'].items():
            if c in ['control','unmasked10']:continue
            matched.append(f"| {r['config']} | {c} | {v['auc']['mean_delta']:+.9f} | {v['ap']['mean_delta']:+.9f} |")
    new=[r for r in audit['records'] if not r['reused_control']]
    max_error=max(x['errors']['full_logit'] for r in new for x in r['explicit_deletions'])
    interventions=[]
    for name in plan['configs']:
        rs=[r for r in new if r['tag'].startswith(name+'_f')]
        assert len(rs)==3
        interventions.append(f"| {name} | {np.mean([r['extra_intervention']['auc'] for r in rs]):.9f} | {np.mean([r['extra_intervention']['ap'] for r in rs]):.9f} |")
    btable='\n'.join(f"| {r['fold']} | {r['models']} | {r['auc_upper_bound']:.9f} | {r['uniform_mixture_auc']:.9f} |" for r in bound['records'])
    report=f'''# 原训练观测消息：正边与抽样零标签上下文

**原测试 AUC/AP 均绝对 +0.02 尚未证实。** 本轮 24 次新训练及 6 个复用对照共
30 个 checkpoint 重放通过。只有原训练内部三折、seed42，未评估新候选的原验证/测试。

| 配置 | 内部 AUC | 内部 AP | 对纯 M123 ΔAUC | ΔAP | 对 unmasked10 ΔAUC | ΔAP |
|---|---:|---:|---:|---:|---:|---:|
{table}

| 零标签候选 | 同 alpha/重建权重的正标签对照 | ΔAUC | ΔAP |
|---|---|---:|---:|
{chr(10).join(matched)}

按训练前规则，选中 {best['config']}，对所有规定对照的双指标最小平均增益
{best['selection_score']:+.9f}。内部 +0.002 推进门槛：{promote}。
最终仍需原测试三种子 AUC≥0.9733929274、AP≥0.9336833737；不以内部成绩替代验收。

## 方法与范围

保留原两层编码器参数/关系均值、M123 融合头、M2 路径集合及 M3 原始统计。
每层添加由原训练正标签或零标签邻居产生的无偏线性消息，按接收度的 .5/1 次幂
归一化；所有新增变换零初始化。正标签模式是匹配对照，不把额外参数本身当作
负标签带来的收益。另比较原重建权重 0/10。

当前查询同时从正例图和观察矩阵中删去，并重算相关度分母。第一层两个端点的
变化经过激活，再更新第二层端点和 M2 中间状态。每个零训练查询也执行这一修正。
观察矩阵仅由当次训练记录重建，没有通过 checkpoint buffer 覆盖显式删除后的图。
抽样零标签没有被解释为敏感/抑制；没有补采负例或引入外部数据、权重和 lncRNA。

与既有能量排斥、边状态递归、有符号固定统计的区别及文献核对见 README.md 和
predecessor_review.json。已核对 GegenNet 公开模型代码，但没有导入它的数据、执行器
或多项式公式；这是独立的 M123 观测消息实验，不是原论文复现或原创性证明。

## 核验与描述性干预

初始检查含 20 个 float64 非零消息下的最终输出/参数梯度对照，覆盖正例、零例、
正/零度一变零和未知查询。八配置另做真实图训练、初始主干一致、留出标签隔离及
显式删记录检查。每个训练后模型再次验证 M1/M2/M3 和最终 logit；最大最终 logit
删除差 {max_error:.3g}。重放概率/标签/索引、选轮、精确 float32 损失及重建爬升均一致。
所有留出配对在两个训练支持中均为零。去掉辅助重建解码器不改变推理。

| 所选 checkpoint | 清零额外消息后的内部 AUC | AP |
|---|---:|---:|
{chr(10).join(interventions)}

上表仅为训练后干预，未参与选型，也不能替代独立训练的匹配正标签对照。
参数范数、逐折干预及显式删除误差均保存在 verification.json。

## 现有预测的共同排序错误

将本轮以及注意力、全局特征、度归一化、分子分支和两个参考模型的保存概率对齐，
每折 34 个候选。对某个正负配对，如果所有候选都严格排反，任意固定非负权重混合
也必然排反；所有候选都打平的配对也必然打平。将其他排序约束全部放开，得到乐观
AUC 上界 `1-(共同严格错误数+0.5*共同平局数)/(正例数*负例数)`。

| 内部折 | 候选数 | 固定非负混合的 AUC 乐观上界 | 固定等权混合实际 AUC |
|---|---:|---:|---:|
{btable}

上界均值为 {bound['mean_auc_upper_bound']:.9f}，相对纯 M123 内部均值最多
{bound['mean_max_possible_gain_over_control']:+.9f}。这只约束这些**保存概率的固定混合**，
不约束新训练模型、查询相关路由，更不构成原测试性能上限。上界不保证可以达到。
该分析没有参与 checkpoint/config 选择，也没有读取原验证或测试。

PLMF 10 个 checkpoint、图、划分与发布源码哈希和此前完整重放一致。
既有纯 miRNA M123+BPR 原测试均值仍为 0.9575892678 / 0.9233078960。
'''
    (HERE/'REPORT.md').write_text(report)
    snap=HERE/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,snap/p.name)
    state=dict(goal_status='active',completion_proven=False,stage='completed_and_audited',
        previous_turn='progress: exact attention and global feature rounds completed without promotion',
        current_turn='progress: predecessor/code review, positive/zero exact deletion, 24 fits/30 replays and shared-ranking-error bounds for 102 saved inner predictions',
        best=best,promote_to_outer_validation=promote,no_lncrna=True,new_candidate_test_evaluations=0,processes_expected_live=[],
        next_action='If promoted, freeze original-validation confirmation. Otherwise avoid extending these constant score mixtures or sampled-zero linear propagation; investigate a distinct nonlinear query-dependent mechanism only after checking prior crossfit/subgraph/path-attention coverage.')
    (HERE/'CONTINUATION.json').write_text(json.dumps(state,indent=2))
    p=HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md';progress=p.read_text().split('最新完成')[0]
    progress+=f'''最新完成 [24 次训练观测消息实验](midrug_observed_context_20261006/REPORT.md)，包含正/零支持、两种度归一化、重建消融和精确删记录。
内部选中 {best['config']}，AUC {best['mean_inner_auc']:.9f}、AP {best['mean_inner_ap']:.9f}；
对规定对照的双指标最小增益 {best['selection_score']:+.9f}。推进原验证：{promote}。

34 个候选每折的共同排序错误给出现有保存概率固定非负混合的内部 AUC 上界均值
{bound['mean_auc_upper_bound']:.9f}；这是特定混合族的诊断，不是原测试或新模型上限。

此前 [全局特征路由](midrug_global_features_20261006/REPORT.md)及
[精确编码器注意力](midrug_exact_attention_20261006/REPORT.md)均未通过内部推进门槛。
追加论文、适用边界与 GegenNet 作者代码核对见 [文献笔记](midrug_literature_20261006/FOLLOWUP_NOTES.md)
和 [本轮方法说明](midrug_observed_context_20261006/README.md)。
没有新候选进入原验证或测试；PLMF 的 10 个 checkpoint、冻结输入和发布源码哈希复核通过。
'''
    p.write_text(progress)
    archive=dict(status='archived',source_hashes={p.name:sha256(p) for p in sorted(HERE.glob('*.py'))},
        verification_sha256=sha256(HERE/'verification.json'),selection_sha256=sha256(HERE/'selection.json'),
        report_sha256=sha256(HERE/'REPORT.md'),plmf_hash_recheck_sha256=sha256(HERE/'plmf_hash_recheck.json'),
        shared_error_bound_sha256=sha256(HERE/'shared_error_bound.json'))
    (HERE/'archive_manifest.json').write_text(json.dumps(archive,indent=2))
    print(json.dumps(dict(best=best,promote=promote,new_fits=24,max_deletion_logit_error=max_error),indent=2))


if __name__=='__main__':main()
