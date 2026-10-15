"""Freeze the declared two-metric gate; protect PLMF and archive this study."""
import csv
import json
import shutil
from pathlib import Path
import numpy as np
from common import HERE, RELEASE, sha256


def main():
    plan = json.loads((HERE / 'results/plan.json').read_text())
    verification = json.loads((HERE / 'verification.json').read_text())
    assert verification['status'] == 'passed' and len(verification['records']) == 15
    assert verification['new_checkpoint_replays'] == 9 and verification['immutable_reused_control_checks'] == 6
    for path, digest in {**plan['source_hashes'], **plan['evidence_hashes']}.items(): assert sha256(path) == digest
    runs = {n: [json.loads(Path(c[str(f)]['path']).read_text()) for f in range(3)] for n, c in plan['controls'].items()}
    for name in plan['configs']:
        runs[name] = [json.loads((HERE / 'results' / f'{name}_f{f}_s42.json').read_text()) for f in range(3)]
    rows = []; flat = []
    for name, records in runs.items():
        refs = plan['comparators'].get(name, ['control', 'unmasked10'])
        comparisons = {ref: {metric: dict(mean_delta=float(np.mean([r['inner_' + metric] - b['inner_' + metric] for r, b in zip(records, runs[ref])])),
                         fold_deltas=[r['inner_' + metric] - b['inner_' + metric] for r, b in zip(records, runs[ref])])
                         for metric in ['auc', 'ap']} for ref in refs}
        rows.append(dict(config=name, mean_inner_auc=float(np.mean([r['inner_auc'] for r in records])),
                         mean_inner_ap=float(np.mean([r['inner_ap'] for r in records])), comparisons=comparisons,
                         selection_score=min(c[m]['mean_delta'] for c in comparisons.values() for m in ['auc', 'ap'])))
        flat.extend(dict(config=name, fold=f, inner_auc=r['inner_auc'], inner_ap=r['inner_ap'],
                         best_epoch=r['best_epoch'], source_best_epoch=r.get('source_best_epoch'), epochs_run=r['epochs_run'])
                    for f, r in enumerate(records))
    eligible = [r for r in rows if r['config'] in plan['eligible_candidates']]
    best = max(eligible, key=lambda r: r['selection_score'])
    promote = best['selection_score'] >= plan['promotion_minimum_both_gains_over_each_control']
    selection = dict(status='frozen_inner_selection', rows=rows, best=best, promote_to_outer_validation=promote,
                     plan_sha256=sha256(HERE / 'results/plan.json'), verification_sha256=sha256(HERE / 'verification.json'),
                     result_hashes={p.name: sha256(p) for p in sorted((HERE / 'results').glob('*_s42.json'))},
                     no_lncrna=True, outer_validation_evaluated=False, test_evaluated=False)
    sp = HERE / 'selection.json'
    if sp.exists(): assert json.loads(sp.read_text()) == selection
    else: sp.write_text(json.dumps(selection, indent=2))
    with (HERE / 'inner_summary.csv').open('w') as out:
        writer = csv.DictWriter(out, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    original = HERE.parent / 'midrug_internal_20261006/plmf_protection.json'
    protected = json.loads(original.read_text())
    for path, digest in protected['source_hashes'].items(): assert sha256(RELEASE / path) == digest
    frozen = RELEASE / 'data/frozen'; manifest = json.loads((frozen / 'manifest.json').read_text())
    refs = HERE.parent / 'baselines_m123_20261004/results'; protection = []
    for r in protected['records']:
        ds, fold = r['dataset'], r['fold']
        assert sha256(refs / f'incrd_m123_{ds}_f{fold}_s42.pt') == r['checkpoint_sha256']
        assert sha256(frozen / f'{ds}_fold{fold}.npz') == r['split_sha256']
        assert sha256(frozen / f'{ds}_graph.pt') == manifest['datasets'][ds]['graph_sha256']
        protection.append(dict(dataset=ds, fold=fold, checkpoint_sha256=r['checkpoint_sha256'],
                               split_sha256=r['split_sha256'], graph_sha256=manifest['datasets'][ds]['graph_sha256']))
    (HERE / 'plmf_hash_recheck.json').write_text(json.dumps(dict(status='passed', records=protection,
        source_unchanged=True, prior_full_replay_sha256=sha256(original)), indent=2))
    table = '\n'.join(f"| {r['config']} | {r['mean_inner_auc']:.9f} | {r['mean_inner_ap']:.9f} |" for r in rows)
    new = [r for r in verification['records'] if not r['reused_control']]
    zero = sum(r['best_stage_epoch'] == 0 for r in new)
    deletion = max(x['errors']['full_logit'] for r in new for x in r['explicit_deletions'])
    report = f'''# 重构辅助任务关闭/衰减的匹配微调检验

**本轮没有达到推进原验证的门槛，最终双 +0.02 目标仍未证实。**
仅原 MiDrug 训练内部三折，seed42；九次新拟合及检查点重放，六个既有对照
复用前轮完整重放并核对元数据、模型、概率、配对及标签。无原验证/测试评估。

| 配置 | 内部 AUC | 内部 AP |
|---|---:|---:|
{table}

off/decay 对纯 M123、源 unmasked10、本轮 keep 的 AUC/AP 均值增益均须 ≥.002。
规定候选中最优为 {best['config']}（并列保留声明顺序），最小规定增益
{best['selection_score']:+.9f}，推进原验证：{promote}。共有 {zero}/9 个分支选中
stage0；这些模型参数/缓冲与源检查点逐项精确相同，不能视为新方法改进。

## 协议与验证

每折同一源 unmasked10 权重/缓冲，重新建立空 RMSprop，主干和解码器 lr1e-5，
alpha.99，解码器 wd1e-4。keep 始终权重10；off 为0；decay 前100步线性下降，
第100步起为0。每轮同序推进辅助采样器，关闭时仅计算诊断MSE。
旧优化器状态未保存，因此这里是权重初始化后的优化器重启，非精确继续旧训练。
stage0 已通过同一内部留出选轮，本轮不是新的独立验证证据。

九次实际训练均经保存的 selected/final 优化器状态核对：主任务有效参数按
阶段更新次数计数；keep 解码器随每步更新，off 无解码器状态，decay 只更新
前99步；未使用的重构token无状态。零阶段选中优化器为空，BN计数等于源轮次
加选中阶段轮次。完整正负查询物理删除、逐查询批次不变性、去除重构模块后的
预测等价均通过；最大完整logit删边误差 {deletion:.3g}。

第一次优化器读取受 Torch1.13 weights-only unpickler 不支持浮点组参数影响，
日志保留。修正为对本地生成且哈希验证的优化器文件使用完整读取；模型权重
仍使用 weights-only。未更改训练代码、数值容差、选轮或结果文件。

PLMF十个既有检查点、冻结划分/图和发布源码保持既有完整重放时的哈希。
最终验收仍为原测试seed42/43/44均值 AUC≥.9733929274、AP≥.9336833737，
且 PLMF MDR/MDS 不降。仅原 miRNA/drug 数据；无 lncRNA、外部数据或新权重。
'''
    state = dict(goal_status='active', completion_proven=False, stage='completed_and_audited', best=best,
                 promote_to_outer_validation=promote, new_fits=9, new_checkpoint_replays=9,
                 reused_control_checks=6, new_candidate_test_evaluations=0, no_lncrna=True,
                 processes_expected_live=[], next_action='Read the frozen post-selection diagnostics; do not retrospectively promote or sweep this failed fine-tuning schedule.')
    diag = HERE / 'component_diagnostics.json'
    if diag.exists():
        diagnostics = json.loads(diag.read_text()); assert diagnostics['status'] == 'passed'
        assert diagnostics['selection_sha256'] == sha256(sp)
        state['component_diagnostics'] = dict(status='passed', selection_unchanged=True, means=diagnostics['means'])
        report += '\n冻结后的[训练曲线与参数诊断](COMPONENT_DIAGNOSTICS.md)同时记录最后训练步的表现；不按这些诊断重选模型。\n'
    (HERE / 'REPORT.md').write_text(report); (HERE / 'CONTINUATION.json').write_text(json.dumps(state, indent=2))
    progress = HERE.parent / 'MIDRUG_DUAL_METRIC_PROGRESS.md'
    prefix = progress.read_text().split('最新完成')[0]
    progress.write_text(prefix + f'''最新完成[重构辅助任务微调](midrug_reconstruction_finetune_20261007/REPORT.md)九次新拟合及重放。
全部九个分支回选源检查点；关闭/衰减重建权重未通过冻结的双指标内部门槛。
没有新的原验证/测试成绩。PLMF 10 个检查点、冻结输入和发布源码保持相同哈希。
前轮[训练量研究](midrug_training_size_20261007/REPORT.md)及[梯度诊断](midrug_reconstruction_gradient_20261007/PRECHECK_REVIEW.md)保留。
''')
    snap = HERE / 'source_snapshot'; snap.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'): shutil.copy2(p, snap / p.name)
    paths = ['verification.json', 'selection.json', 'REPORT.md', 'CONTINUATION.json', 'plmf_hash_recheck.json',
             'results/plan.json', 'initial_checks.json', 'README.md', 'inner_summary.csv', 'check.log', 'launch.log', 'audit.log', 'audit_initial_failure.log']
    if diag.exists(): paths += ['component_diagnostics.json', 'COMPONENT_DIAGNOSTICS.md', 'diagnose.log']
    paths += sorted(str(p.relative_to(HERE)) for p in (HERE / 'results').iterdir()
                    if p.suffix in {'.pt', '.npz', '.json'} and p.name != 'plan.json')
    (HERE / 'archive_manifest.json').write_text(json.dumps(dict(status='archived',
        source_hashes={p.name: sha256(p) for p in sorted(HERE.glob('*.py'))},
        artifact_hashes={p: sha256(HERE / p) for p in paths},
        frozen_dependencies={**plan['source_hashes'], **plan['evidence_hashes']}), indent=2))
    print(json.dumps(dict(means=[{k: r[k] for k in ('config', 'mean_inner_auc', 'mean_inner_ap')} for r in rows],
                         promote=promote, stage_zero_count=zero, max_deletion_error=deletion), indent=2))


if __name__ == '__main__': main()
