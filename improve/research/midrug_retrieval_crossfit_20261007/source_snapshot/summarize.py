"""Freeze the declared gate, then diagnose and archive all 27 fits/replays."""
import csv
import json
import shutil
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from common import HERE, RELEASE, load_inner, seed_all, sha256
from model import make_adapter


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2))


def main():
    plan = json.loads((HERE/'results/plan.json').read_text())
    base = json.loads((HERE/'base_verification.json').read_text())
    v = json.loads((HERE/'verification.json').read_text())
    assert base['status'] == v['status'] == 'passed'
    assert base['new_checkpoint_replays'] == 18 and len(base['records']) == 18
    assert v['new_checkpoint_replays'] == 9 and len(v['records']) == 15
    assert v['immutable_reused_control_checks'] == 6
    dependencies = {**plan['source_hashes'], **plan['evidence_hashes']}
    for p, h in dependencies.items():
        assert sha256(p) == h
    runs = {n: [json.loads(Path(c[str(f)]['path']).read_text()) for f in range(3)] for n, c in plan['controls'].items()}
    runs['full300'] = [json.loads((HERE/'offsets'/f'full300_f{f}.json').read_text()) for f in range(3)]
    runs.update({n: [json.loads((HERE/'results'/f'{n}_f{f}_s42.json').read_text()) for f in range(3)] for n in plan['configs']})
    rows = [dict(config=n, mean_inner_auc=float(np.mean([r['inner_auc'] for r in rs])),
                 mean_inner_ap=float(np.mean([r['inner_ap'] for r in rs]))) for n, rs in runs.items()]
    assert plan['eligible_candidates'] == ['oof_label']
    best = next(r for r in rows if r['config'] == 'oof_label')
    best['comparisons'] = {n: {m: dict(
        mean_delta=float(np.mean([r['inner_'+m]-b['inner_'+m] for r, b in zip(runs['oof_label'], runs[n])])),
        fold_deltas=[r['inner_'+m]-b['inner_'+m] for r, b in zip(runs['oof_label'], runs[n])])
        for m in ['auc', 'ap']} for n in plan['comparators']}
    best['selection_score'] = min(c[m]['mean_delta'] for c in best['comparisons'].values() for m in ['auc', 'ap'])
    promote = best['selection_score'] >= plan['promotion_minimum_both_gains_over_each_control']
    selection = dict(status='frozen_inner_selection', rows=rows, best=best, promote_to_outer_validation=promote,
        plan_sha256=sha256(HERE/'results/plan.json'), verification_sha256=sha256(HERE/'verification.json'),
        base_verification_sha256=sha256(HERE/'base_verification.json'),
        result_hashes={str(p.relative_to(HERE)): sha256(p) for p in sorted((HERE/'results').glob('*_s42.json'))},
        no_lncrna=True, outer_validation_evaluated=False, test_evaluated=False)
    sp = HERE/'selection.json'
    if sp.exists():
        assert json.loads(sp.read_text()) == selection
    else:
        write(sp, selection)
    diagnostics = []
    for name in plan['configs']:
        for fold in range(3):
            r = runs[name][fold]; tag = r['tag']
            seed_all(42); data = load_inner(fold); a = make_adapter(data, name, 'cuda:0')
            a.model.load_state_dict(torch.load(HERE/'results'/f'{tag}.pt', map_location='cuda:0', weights_only=True))
            a.model.eval(); memory = a.model.case_memory
            with np.load(HERE/'replay_predictions'/f'{tag}.npz') as z:
                train, obj = z['train'], z['objective_probability']
            with torch.no_grad():
                query = a.pairs['val']; _, _, idx, weight = memory.context(query); delta = memory(query)
                contexts = memory.support_pairs[idx]; qy = query.T[:, None, :]
            d = dict(config=name, fold=fold, best_epoch=r['best_epoch'],
                train_inference_auc=roc_auc_score(data['train_y'], train), train_inference_ap=average_precision_score(data['train_y'], train),
                fitted_offset_auc=roc_auc_score(data['train_y'], obj), fitted_offset_ap=average_precision_score(data['train_y'], obj),
                inner_auc=r['inner_auc'], inner_ap=r['inner_ap'],
                mean_abs_residual_logit=float(delta.abs().mean()), rms_residual_logit=float(delta.square().mean().sqrt()),
                mean_effective_neighbors=float((1/weight.square().sum(1)).mean()),
                same_miRNA_mass=float((weight*(contexts[:, :, 0] == qy[:, :, 0])).sum(1).mean()),
                same_drug_mass=float((weight*(contexts[:, :, 1] == qy[:, :, 1])).sum(1).mean()),
                positive_label_mass=float((weight*memory.support_y[idx]).sum(1).mean()))
            diagnostics.append(d); del a, memory; torch.cuda.empty_cache()
    means = {n: {k: float(np.mean([r[k] for r in diagnostics if r['config'] == n]))
        for k in diagnostics[0] if k not in ['config', 'fold', 'best_epoch']} for n in plan['configs']}
    write(HERE/'component_diagnostics.json', dict(status='passed', selection_sha256=sha256(sp), records=diagnostics,
        means=means, no_further_selection=True,
        interpretation='Fitted-offset metrics use the learned branch on its fitting samples and are not out-of-fold evaluation of the whole predictor.'))
    flat = [dict(config=n, fold=f, **{k: r[k] for k in ['inner_auc', 'inner_ap', 'best_epoch', 'epochs_run']})
            for n, rs in runs.items() for f, r in enumerate(rs)]
    with (HERE/'inner_summary.csv').open('w') as out:
        w = csv.DictWriter(out, fieldnames=list(flat[0])); w.writeheader(); w.writerows(flat)
    original = HERE.parent/'midrug_internal_20261006/plmf_protection.json'
    protected = json.loads(original.read_text())
    for p, h in protected['source_hashes'].items():
        assert sha256(RELEASE/p) == h
    frozen = RELEASE/'data/frozen'; manifest = json.loads((frozen/'manifest.json').read_text())
    refs = HERE.parent/'baselines_m123_20261004/results'; protection = []
    for r in protected['records']:
        ds, f = r['dataset'], r['fold']
        assert sha256(refs/f'incrd_m123_{ds}_f{f}_s42.pt') == r['checkpoint_sha256']
        assert sha256(frozen/f'{ds}_fold{f}.npz') == r['split_sha256']
        assert sha256(frozen/f'{ds}_graph.pt') == manifest['datasets'][ds]['graph_sha256']
        protection.append(dict(dataset=ds, fold=f, checkpoint_sha256=r['checkpoint_sha256'],
            split_sha256=r['split_sha256'], graph_sha256=manifest['datasets'][ds]['graph_sha256']))
    write(HERE/'plmf_hash_recheck.json', dict(status='passed', records=protection, source_unchanged=True,
        prior_full_replay_sha256=sha256(original), new_gpu_replays=0))
    table = '\n'.join(f"| {r['config']} | {r['mean_inner_auc']:.9f} | {r['mean_inner_ap']:.9f} |" for r in rows)
    paired = '\n'.join(f"| {ref} | {c['auc']['mean_delta']:+.9f} | {c['ap']['mean_delta']:+.9f} |" for ref, c in best['comparisons'].items())
    diagtable = '\n'.join(f"| {n} | {r['train_inference_auc']:.6f} | {r['train_inference_ap']:.6f} | {r['rms_residual_logit']:.6f} |" for n, r in means.items())
    stages = {n: [r['best_epoch'] for r in runs[n]] for n in plan['configs']}
    new = [r for r in v['records'] if not r['reused_control']]
    deleted = max(x['errors']['full_logit'] for r in new for x in r['explicit_deletions'])
    report = f'''# 折外预测与训练内预测：案例残差内部对照

**原测试 AUC/AP 同时绝对 +.02 尚未证实。** 完成18次基模型训练与9次案例分支训练，
共27次新训练、27个所选/固定检查点预测重放；另检查9个案例最终状态及所选/最终优化器。
六个旧强对照引用不可变重放。只用原训练内部三折 seed42，没有新原验证或测试评估。

| 配置 | 内部 AUC | 内部 AP |
|---|---:|---:|
{table}

| oof_label 的规定对照 | ΔAUC | ΔAP |
|---|---:|---:|
{paired}

唯一候选 oof_label 的最小规定平均增益 {best['selection_score']:+.9f}；推进原验证：{promote}。
须对全部五个对照的三折平均 AUC/AP 均 ≥+.002，不只与固定300轮基线比较，不事后改选对照。
最终仍须原测试 seeds42/43/44 均值 AUC≥.9733929274、AP≥.9336833737，PLMF 两集不降。

## 训练与信息边界

18个基模型是各内折5个嵌套4/5训练模型加1个完整训练模型，均为原 M123+unmasked10、
seed42、固定300轮。被预测折不用于选轮；嵌套索引与旧 crossfit 已审计缓存相同。
每个训练样本恰好一个折外 logit，来自没有该样本标签的基模型。

案例分支使用原属性构建的32近邻、64维 key 与 label/key差值 value，正负自身配对均排除。
三个臂固定同一案例输入、零输出初始化、完整300轮推理模型；只更新案例分支。
oof_label 使用折外 offset，insample_label 使用完整模型训练 offset，oof_features 去掉 label value。
RMSprop lr1e-4/alpha.99/wd1e-4，set_to_none=True；最多300更新、至少100、耐心100，stage0也可选。
所选阶段：{stages}。完整原模型参数与缓冲在所选及最终状态均逐位等于源模型，BN计数保持300。
OOF基模型训练数据量和图密度仍小于推理模型，故不能将差异全归因于自己的标签是否参与拟合。

## 审计与选择后的诊断

18个基模型核验所有训练/预测缓存、固定轮次、输入哈希和物理删查询；9个案例模型核验
预测、最早最优选轮、实际优化器更新次数及最终状态。案例完整 M1/M2/M3 物理删查询最大
logit误差 {deleted:.3g}；正负自身标签排除、查询批次不变、移除重构模块推理等价均通过。

| 配置 | 完整基模型训练推理 AUC | 完整基模型训练推理 AP | 留出残差 logit RMS |
|---|---:|---:|---:|
{diagtable}

拟合 offset 对应概率另记于 component_diagnostics.json；案例分支仍在这些样本上训练，
该指标不能称整个模型的折外泛化成绩。诊断在 selection.json 冻结后生成，不用于再选参数。

本轮延续旧 crossfit 家族，将原属性案例与原重构主干组合；不宣称首次提出折外残差。
[文献核读](../midrug_task_literature_20261007/READING_NOTES.md)与
[前轮依据](../midrug_task_literature_20261007/WARMSTART_FOLLOWUP.md)保留。
没有 lncRNA、额外序列、注释、数据集或新预训练权重；零标签不等于实验确认的药物敏感性。
PLMF十个旧模型、冻结输入和发布源码哈希未变；本轮没有新增PLMF训练或GPU重放。
'''
    (HERE/'REPORT.md').write_text(report)
    state = dict(goal_status='active', completion_proven=False, stage='completed_and_audited', best=best,
        promote_to_outer_validation=promote, new_fits=27, new_base_fits=18, new_case_fits=9,
        new_checkpoint_replays=27, final_case_state_inspections=9, reused_control_checks=6,
        no_lncrna=True, new_candidate_test_evaluations=0, processes_expected_live=[], component_diagnostics=means,
        next_action='Use frozen comparisons and diagnostics to identify a distinct evidence-supported hypothesis; do not retrospectively tune this case branch.')
    write(HERE/'CONTINUATION.json', state)
    progress = HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md'; prefix = progress.read_text().split('最新完成')[0]
    progress.write_text(prefix+f'''最新完成[折外预测案例分支对照](midrug_retrieval_crossfit_20261007/REPORT.md)：27次新训练与27个检查点预测重放。
oof_label 内部 AUC/AP 为 {best['mean_inner_auc']:.9f}/{best['mean_inner_ap']:.9f}；推进原验证：{promote}。
[前轮主干冻结检索](midrug_retrieval_warmstart_20261007/REPORT.md)及[文献核读](midrug_task_literature_20261007/READING_NOTES.md)保留。
没有新原验证/测试成绩；PLMF十个旧模型及冻结输入/发布源码哈希未变。
''')
    snap = HERE/'source_snapshot'; snap.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):
        shutil.copy2(p, snap/p.name)
    paths = [p for p in HERE.iterdir() if p.is_file() and p.suffix in {'.py', '.json', '.md', '.csv', '.log'}
             and p.name not in {'archive_manifest.json', 'archive_recheck.json', 'summarize.log'}]
    for dirname in ['bases', 'offsets', 'results', 'replay_predictions']:
        paths.extend(p for p in (HERE/dirname).rglob('*') if p.is_file())
    archive = dict(status='archived', source_hashes={p.name: sha256(p) for p in sorted(HERE.glob('*.py'))},
        artifact_hashes={str(p.relative_to(HERE)): sha256(p) for p in sorted(paths)}, frozen_dependencies=dependencies)
    write(HERE/'archive_manifest.json', archive); count = 0
    for p, h in archive['source_hashes'].items():
        assert sha256(HERE/p) == h and sha256(snap/p) == h; count += 2
    for p, h in archive['artifact_hashes'].items():
        assert sha256(HERE/p) == h; count += 1
    for p, h in archive['frozen_dependencies'].items():
        assert sha256(p) == h; count += 1
    write(HERE/'archive_recheck.json', dict(status='passed', hash_checks=count, new_checkpoint_replays=27,
        final_case_state_inspections=9, reused_control_checks=6, protected_plmf_records=10, selection_unchanged=True,
        archive_sha256=sha256(HERE/'archive_manifest.json'), selection_sha256=sha256(sp),
        no_lncrna=True, outer_validation_evaluated=False, test_evaluated=False))
    scope_path = HERE.parent/'ACTIVE_SCOPE.json'; scope = json.loads(scope_path.read_text())
    scope.update(active_directory=str(HERE), latest_continuation=str(HERE/'CONTINUATION.json'),
        latest_archive_check=str(HERE/'archive_recheck.json'), next_action=state['next_action'])
    write(scope_path, scope)
    print(json.dumps(dict(best=best, promote=promote, diagnostics=means), indent=2))


if __name__ == '__main__':
    main()
