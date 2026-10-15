"""Freeze selection and archive the fixed proper-loss experiment."""
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
    comparisons={k:{m:dict(mean_delta=float(np.mean([a[m]-b[m] for a,b in zip(scores['negative_tail'],scores[k])])),fold_deltas=[a[m]-b[m] for a,b in zip(scores['negative_tail'],scores[k])]) for m in ['auc','ap']} for k in plan['comparators']}
    minimum=min(r[m]['mean_delta'] for r in comparisons.values() for m in ['auc','ap']);promote=minimum>=.002
    selection=dict(status='frozen_inner_selection',candidate='negative_tail',rows=rows,comparisons=comparisons,minimum_gain=minimum,promote_to_outer_validation=promote,
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
    (HERE/'REPORT.md').write_text(f'''# 保持概率最优解的损失形状：内部结果

**原测试 AUC/AP 各绝对 +.02 尚未证实。** 只用 miRNA/drug 原训练内部三折，
本轮完成9次拟合、9个所选检查点重放、9个最终状态及所选/最终优化器核验。
没有新候选原验证或测试评估。

| 配置 | 内部 AUC | 内部 AP |
|---|---:|---:|
{table}

| negative_tail 相对对照 | ΔAUC | ΔAP |
|---|---:|---:|
{paired}

唯一候选的最小规定增益 {minimum:+.9f}，通过推进门槛：{promote}。
要求相对四个规定对照的两个三折均值均≥+.002，不替代用户原测试双+.02。
所选轮次：{selected}。positive_tail/brier是机制对照，不事后改选。

## 方法及限制

使用经典 proper partial loss 的权函数构造。p=sigmoid(z)：
negative_tail的负/正损失分别为2p、2[-log(p)-(1-p)]，其对logit的梯度
是BCE梯度乘2(1-p)；positive_tail是标签反射，乘2p；Brier为2(p-y)^2，
乘4p(1-p)。三个因子在p=.5均为1。训练前30轮BCE，30–60轮凸过渡至指定损失。
它们在不受模型限制的总体问题中以观测条件类别概率为最优解；这不是对错标
不变性的证明，也不是神经网络校准、凸优化或排序收益保证。简单的概率校准
不能替代AUC/AP。negative_tail也改变正例和容易负例梯度，不能说只处理错标。

三臂原M123/重建10同初始化，原图、M1/M2/M3和推理结构完全保留。
原RMSprop lr1e-4/alpha.99，原参数wd0、重建wd1e-4；max800/min300/patience200。
重建前50轮爬升、私有采样流相同；每轮一次原BN更新，按内层AUC/AP均值选轮。
这些内层折反复用于开发，不能视为独立泛化确认。

100位Decimal独立损失、解析梯度、float64 gradcheck、Taylor切换、极端logit、
条件风险最优点、标签反射和p=.5梯度匹配通过。无过渡时全模型梯度与BCE一致；
预检查发现Torch默认softplus在20附近的近似切换影响有限差分，冻结前改用
logaddexp实现同一数学函数并保留失败日志，未改损失形状或训练方案。
真实训练梯度、BN次数和留出标签隔离通过。所选概率、曲线、优化器和最终状态
均复核；完整模型物理删查询最大logit误差 {deleted:.3g}，移除重建不改变推理。

[近期论文与基础理论核读](../midrug_loss_shape_literature_20261007/READING_NOTES.md)
记录了UWSR/RefCal的实际方法及与本轮差异。它们不是本实验的直接复现。
理论依据来自Reid与Williamson的2010年Composite Binary Losses；不声称新发明。
没有lncRNA、外部数据、序列、注释或新权重。PLMF十个原模型/输入/发布代码
哈希一致，本轮PLMF GPU重放为0。仍须原测试三种子均值AUC≥.9733929274189016、
AP≥.933683373732677，且PLMF两数据集不下降。
''')
    state=dict(goal_status='active',completion_proven=False,stage='completed_and_audited',new_fits=9,new_selected_replays=9,final_state_inspections=9,
        promote_to_outer_validation=promote,new_candidate_test_evaluations=0,processes_expected_live=[],
        next_action='Use the frozen proper-loss gradient-shape comparison to choose a distinct hypothesis; do not tune loss exponents, transition weight/schedule or retrospectively promote controls. Original dual +.02 goal remains active.')
    write(HERE/'CONTINUATION.json',state)
    artifacts=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name not in ['archive_manifest.json','archive_recheck.json','summarize.log']]
    archive=dict(status='archived',artifact_hashes={str(p.relative_to(HERE)):sha256(p) for p in sorted(artifacts)},frozen_dependencies=deps);write(HERE/'archive_manifest.json',archive)
    count=0
    for p,h in archive['artifact_hashes'].items():assert sha256(HERE/p)==h;count+=1
    for p,h in deps.items():assert sha256(p)==h;count+=1
    write(HERE/'archive_recheck.json',dict(status='passed',hash_checks=count,archive_sha256=sha256(HERE/'archive_manifest.json'),selection_sha256=sha256(HERE/'selection.json')))
    scopefile=HERE.parent/'ACTIVE_SCOPE.json';scope=json.loads(scopefile.read_text());scope.update(active_directory=str(HERE),latest_continuation=str(HERE/'CONTINUATION.json'),latest_archive_check=str(HERE/'archive_recheck.json'),
        latest_experiment_archive_check=str(HERE/'archive_recheck.json'),latest_research_review=str(HERE.parent/'midrug_loss_shape_literature_20261007/READING_NOTES.md'),latest_research_continuation=str(HERE/'CONTINUATION.json'),
        latest_research_archive_check=str(HERE.parent/'midrug_loss_shape_literature_20261007/review_recheck.json'),next_action=state['next_action']);write(scopefile,scope)
    progress=HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md';prefix=progress.read_text().split('最新完成')[0]
    progress.write_text(prefix+f'最新完成[proper损失形状](midrug_proper_loss_20261007/REPORT.md)：9 次新拟合和所选重放，最终状态及优化器另查。推进原验证：{promote}。原测试双+.02未证实，PLMF冻结产物未变。\n')
    print(json.dumps(selection,indent=2))

if __name__=='__main__':main()
