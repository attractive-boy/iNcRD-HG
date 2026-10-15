"""Freeze selection and archive the fixed block-order experiment."""
import json,csv
from pathlib import Path
import numpy as np
from common import HERE,RELEASE,CLEAN,sha256

def write(p,x):p.write_text(json.dumps(x,indent=2))

def main():
    out=HERE/'results';plan=json.loads((out/'plan.json').read_text());v=json.loads((HERE/'verification.json').read_text());assert v['status']=='passed'
    deps={**plan['source_hashes'],**plan['evidence_hashes'],str(CLEAN/'graph.pt'):plan['graph_sha256'],str(CLEAN/'inner_data.npz'):plan['data_sha256']}
    for f,h in deps.items():assert sha256(f)==h
    runs={k:[json.loads((out/f'{k}_f{f}_s42.json').read_text()) for f in range(3)] for k in plan['configs']}
    scores={k:[dict(auc=r['inner_auc'],ap=r['inner_ap']) for r in rs] for k,rs in runs.items()}
    for k in plan['controls']:scores[k]=[next(r['metrics'] for r in v['controls'] if r['name']==k and r['fold']==f) for f in range(3)]
    rows=[dict(config=k,auc=float(np.mean([r['auc'] for r in rs])),ap=float(np.mean([r['ap'] for r in rs]))) for k,rs in scores.items()]
    comparisons={k:{m:dict(mean_delta=float(np.mean([a[m]-b[m] for a,b in zip(scores['consumer_first'],scores[k])])),fold_deltas=[a[m]-b[m] for a,b in zip(scores['consumer_first'],scores[k])]) for m in ['auc','ap']} for k in plan['comparators']}
    minimum=min(r[m]['mean_delta'] for r in comparisons.values() for m in ['auc','ap']);promote=minimum>=.002
    selection=dict(status='frozen_inner_selection',candidate='consumer_first',rows=rows,comparisons=comparisons,minimum_gain=minimum,promote_to_outer_validation=promote,
        plan_sha256=sha256(out/'plan.json'),verification_sha256=sha256(HERE/'verification.json'),result_hashes={p.name:sha256(p) for p in out.glob('*_s42.json')},
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    assert not (HERE/'selection.json').exists();write(HERE/'selection.json',selection)
    with (HERE/'inner_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for cs in plan['controls'].values():
        for r in cs.values():
            p=Path(r['path']);deps.update({str(p):r['metadata_sha256'],str(p.with_suffix('.pt')):r['checkpoint_sha256'],str(p.with_name(p.stem+'_inner.npz')):r['probability_sha256']})
    original=HERE.parent/'midrug_internal_20261006/plmf_protection.json';old=json.loads(original.read_text());deps[str(original)]=sha256(original)
    for p,h in old['source_hashes'].items():assert sha256(RELEASE/p)==h;deps[str(RELEASE/p)]=h
    frozen=RELEASE/'data/frozen';manifest=json.loads((frozen/'manifest.json').read_text());refs=HERE.parent/'baselines_m123_20261004/results';protection=[]
    for r in old['records']:
        ds,f=r['dataset'],r['fold'];paths={str(refs/f'incrd_m123_{ds}_f{f}_s42.pt'):r['checkpoint_sha256'],str(frozen/f'{ds}_fold{f}.npz'):r['split_sha256'],str(frozen/f'{ds}_graph.pt'):manifest['datasets'][ds]['graph_sha256']}
        for p,h in paths.items():assert sha256(p)==h
        deps.update(paths);protection.append(dict(dataset=ds,fold=f,hashes=paths))
    write(HERE/'plmf_hash_recheck.json',dict(status='passed',records=protection,new_gpu_replays=0,prior_gpu_replay_sha256=sha256(original)))
    table='\n'.join(f"| {r['config']} | {r['auc']:.9f} | {r['ap']:.9f} |" for r in rows)
    paired='\n'.join(f"| {k} | {r['auc']['mean_delta']:+.9f} | {r['ap']['mean_delta']:+.9f} |" for k,r in comparisons.items())
    selected={k:[r['best_epoch'] for r in rs] for k,rs in runs.items()};deleted=max(x['errors']['full_logit'] for r in v['records'] for x in r['physical_deletions'])
    (HERE/'REPORT.md').write_text(f'''# 表示/分类头更新顺序：内部结果

**原测试 AUC/AP 各绝对 +.02 尚未证实。** 只用 miRNA/drug 原训练内部三折，
完成9次拟合、9个所选检查点重放、9个最终状态及所选/最终优化器核验。
没有新候选原验证或测试评估。

| 配置 | 内部 AUC | 内部 AP |
|---|---:|---:|
{table}

| consumer_first 相对对照 | ΔAUC | ΔAP |
|---|---:|---:|
{paired}

唯一候选的最小规定增益 {minimum:+.9f}，通过推进门槛：{promote}。
要求相对全部四个对照的两个三折均值均≥+.002，不替代原测试双+.02。
所选轮次：{selected}。representation_first和joint不事后改为候选。

## 方法与界限

根据2026年UWSR论文§3.3的更新顺序干预，在每轮两个完整前后向中分别更新
分类头C（lin1/2/3）与表示R（其余原参数及重建解码器）。C-first先更新头，
再在已更新头下重新计算表示梯度；R-first反转顺序；joint两块都使用第一遍
预更新梯度，第二遍仅匹配计算预算。每个有效参数每轮仅一步RMSprop，不是
把同一模型完整更新两次。两遍使用同一重建节点采样。

第二遍BN仍使用当前批统计量，只临时停止running统计追踪，保证每轮BN更新一次。
原M1/M2/M3、图、推理结构、BCE+重构10、初始化、私有采样器、RMSprop学习率
1e-4/alpha.99/eps1e-8及正则均保留。R原参数wd0、重建wd1e-4，C wd0。
max800/min300/patience200，最早最大内层(AUC+AP)/2选轮。
额外前向增加算力，不增加每参数优化步数。第一/第二遍损失对应不同参数点，
完整曲线分别记录，不能把两者相加解释成某个联合优化目标。

这是共享图表示上的分块训练实验，不是UWSR的稀疏嵌入后验正则化复现，
也不能把排序结果唯一归因为自影响或所谓标签泄漏。原模型没有论文假设的
独立稀疏lookup参数；本轮未构造未经依据的不确定性或Fisher权重。

手算三参数RMSprop（含持续二阶矩和解码器衰减）与三种顺序完全一致；实际图
同一梯度的拆分/单优化器更新逐位一致。独立CUDA前后向三步轨迹的最大参数
差约9.66e-6，原过严断言的失败日志保留；之后使用显式同梯度检查区分数值
归约差异与优化规则错误。BN第二遍输出一致/缓冲不变/异常恢复、每块一次
更新、采样一次、留出隔离、完整物理删查询均通过。

九个模型概率/选轮/两遍损失/BN与RMSprop步数、所选/最终mask RNG状态核验通过。
物理删查询最大完整logit差 {deleted:.3g}，去掉辅助重构保持原推理等价。
旧对照引用既有完整重放并核对哈希/预测。PLMF十个原模型、输入及发布代码
哈希保护通过；本轮新PLMF GPU重放为0。

[原文及本地差别](READING_NOTES.md)。这些内层折反复用于开发，不是独立确认。
未使用lncRNA、外部数据、额外注释、序列或新权重。目标仍为原测试三种子均值
AUC≥.9733929274189016、AP≥.933683373732677，且PLMF两数据集不下降。
''')
    state=dict(goal_status='active',completion_proven=False,stage='completed_and_audited',new_fits=9,new_selected_replays=9,final_state_inspections=9,
        promote_to_outer_validation=promote,new_candidate_test_evaluations=0,processes_expected_live=[],
        next_action='Use the frozen block-order intervention to assess shared-representation/consumer coupling; do not tune block definitions, order schedules or learning rates after these metrics, and do not retrospectively promote controls. Original dual +.02 goal remains active.')
    write(HERE/'CONTINUATION.json',state)
    artifacts=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['archive_manifest.json','archive_recheck.json','summarize.log']]
    archive=dict(status='archived',artifact_hashes={str(p.relative_to(HERE)):sha256(p) for p in sorted(artifacts)},frozen_dependencies=deps);write(HERE/'archive_manifest.json',archive)
    count=0
    for p,h in archive['artifact_hashes'].items():assert sha256(HERE/p)==h;count+=1
    for p,h in deps.items():assert sha256(p)==h;count+=1
    write(HERE/'archive_recheck.json',dict(status='passed',hash_checks=count,archive_sha256=sha256(HERE/'archive_manifest.json'),selection_sha256=sha256(HERE/'selection.json')))
    scopefile=HERE.parent/'ACTIVE_SCOPE.json';scope=json.loads(scopefile.read_text());scope.update(active_directory=str(HERE),latest_continuation=str(HERE/'CONTINUATION.json'),latest_archive_check=str(HERE/'archive_recheck.json'),
        latest_experiment_archive_check=str(HERE/'archive_recheck.json'),latest_research_review=str(HERE/'READING_NOTES.md'),latest_research_continuation=str(HERE/'CONTINUATION.json'),
        latest_research_archive_check=str(HERE/'archive_recheck.json'),next_action=state['next_action']);write(scopefile,scope)
    progress=HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md';prefix=progress.read_text().split('最新完成')[0]
    progress.write_text(prefix+f'最新完成[表示/头更新顺序](midrug_block_order_20261007/REPORT.md)：9 次新拟合和所选重放，最终状态及优化器另查。推进原验证：{promote}。原测试双+.02未证实，PLMF冻结产物未变。\n')
    print(json.dumps(selection,indent=2))

if __name__=='__main__':main()
