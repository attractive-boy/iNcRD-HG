"""Freeze selection and archive the complete paired interpolation experiment."""
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
    comparisons={k:{m:dict(mean_delta=float(np.mean([a[m]-b[m] for a,b in zip(scores['drug_mix'],scores[k])])),fold_deltas=[a[m]-b[m] for a,b in zip(scores['drug_mix'],scores[k])]) for m in ['auc','ap']} for k in plan['comparators']}
    minimum=min(r[m]['mean_delta'] for r in comparisons.values() for m in ['auc','ap']);promote=minimum>=.002
    selection=dict(status='frozen_inner_selection',candidate='drug_mix',rows=rows,comparisons=comparisons,minimum_gain=minimum,promote_to_outer_validation=promote,
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
    (HERE/'REPORT.md').write_text(f'''# 同药物配对表示插值：内部结果

**原测试 AUC/AP 各绝对 +.02 尚未证实。** 本轮只用 miRNA/drug 原训练内部三折，
完成 6 次拟合、6 个所选检查点重放、6 个最终状态检查及所选/最终优化器核验。
没有新候选原验证或测试评估。

| 配置 | 内部 AUC | 内部 AP |
|---|---:|---:|
{table}

| drug_mix 相对对照 | ΔAUC | ΔAP |
|---|---:|---:|
{paired}

唯一候选的最小规定增益 {minimum:+.9f}，通过推进原验证门槛：{promote}。
要求相对全部对照的 AUC/AP 三折均值各 ≥+.002；这个开发门槛没有替代最终测试 +.02。
所选轮次：{selected}。Global Mixup 是预先指定对照，不事后改选。

## 方法与核验

两臂从相同 M123 随机初始化训练，保留原重建权重10/50轮爬升。混合位置固定在
lin1 之前的配对表示，特征与软标签使用相同 Uniform[0,1] 系数，梯度回到两个
原配对及整个编码器。drug_mix 在同药物内做无自配对置换，global_mix 在相同
有效支持集上全局置换；单例在两臂均不混合。共享每轮一个系数，保证总体标签
比例不变。没有添加合成图边，推理仍为原 M123；不声称插值是真实生物学样本。

关联损失为 clean/mixed BCE 的凸组合，30轮后爬升至各一半，60轮后固定。
RMSprop lr1e-4/alpha.99，原参数wd0、重建wd1e-4；max800/min300/patience200。
每轮仅一次原 BN 更新；按内层 AUC/AP 均值最早最大值选轮。
这些留出已被前期开发反复使用，属于开发证据，不是独立泛化确认。

独立逐样本损失/梯度、双来源梯度、系数端点、置换/单例/标签比例、无混合权重
的模型梯度等价、解码一致、有限非零梯度和留出标签翻转检查通过。
所有所选预测和曲线、BN 更新次数、所选/最终 RMSprop 步数及数值有限性通过。
完整 M1/M2/M3 物理删查询最大 logit 差 {deleted:.3g}；移除重建模块保持原推理等价。

参考 [近期论文与基础方法核读](../midrug_vicinal_literature_20261007/READING_NOTES.md)。
Pinterest 使用随机配对，其报告也提示限制伙伴可能损失多样性，本轮不预设同药物更优。
没有 lncRNA、新数据、注释、序列或预训练权重。PLMF 十个旧模型及原输入/发布源码
哈希不变，本轮 PLMF GPU 重放为 0。
最终仍需原测试三种子均值 AUC≥.9733929274189016、AP≥.933683373732677，PLMF 两集不降。
''')
    state=dict(goal_status='active',completion_proven=False,stage='completed_and_audited',new_fits=6,new_selected_replays=6,final_state_inspections=6,
        promote_to_outer_validation=promote,new_candidate_test_evaluations=0,processes_expected_live=[],
        next_action='Use the frozen same-drug versus global interpolation comparison to choose a distinct hypothesis; do not sweep mixing weights, Beta shape, mixing layers or retrospectively promote the global control. Original dual +.02 goal remains active.')
    write(HERE/'CONTINUATION.json',state)
    artifacts=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['archive_manifest.json','archive_recheck.json','summarize.log']]
    archive=dict(status='archived',artifact_hashes={str(p.relative_to(HERE)):sha256(p) for p in sorted(artifacts)},frozen_dependencies=deps);write(HERE/'archive_manifest.json',archive)
    count=0
    for p,h in archive['artifact_hashes'].items():assert sha256(HERE/p)==h;count+=1
    for p,h in deps.items():assert sha256(p)==h;count+=1
    write(HERE/'archive_recheck.json',dict(status='passed',hash_checks=count,archive_sha256=sha256(HERE/'archive_manifest.json'),selection_sha256=sha256(HERE/'selection.json')))
    scopefile=HERE.parent/'ACTIVE_SCOPE.json';scope=json.loads(scopefile.read_text());scope.update(active_directory=str(HERE),latest_continuation=str(HERE/'CONTINUATION.json'),latest_archive_check=str(HERE/'archive_recheck.json'),
        latest_experiment_archive_check=str(HERE/'archive_recheck.json'),latest_research_review=str(HERE.parent/'midrug_vicinal_literature_20261007/READING_NOTES.md'),latest_research_continuation=str(HERE/'CONTINUATION.json'),
        latest_research_archive_check=str(HERE.parent/'midrug_vicinal_literature_20261007/review_recheck.json'),next_action=state['next_action']);write(scopefile,scope)
    progress=HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md';prefix=progress.read_text().split('最新完成')[0]
    progress.write_text(prefix+f'最新完成[同药物表示插值](midrug_entity_mixup_20261007/REPORT.md)：6 次新拟合和所选重放，最终状态及优化器另查。推进原验证：{promote}。原测试双+.02未证实，PLMF冻结产物未变。\n')
    print(json.dumps(selection,indent=2))

if __name__=='__main__':main()
