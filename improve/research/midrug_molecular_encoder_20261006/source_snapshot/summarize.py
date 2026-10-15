"""Freeze paired selection and separate chemistry attribution from added capacity."""
import csv,json,shutil
from pathlib import Path
import numpy as np
from common import HERE,RELEASE,sha256


def main():
    plan=json.loads((HERE/'results/plan.json').read_text())
    verified=json.loads((HERE/'verification.json').read_text())
    assert verified['status']=='passed' and len(verified['records'])==24
    for p,h in plan['source_hashes'].items():assert sha256(p)==h
    runs={name:[json.loads(Path(c[str(f)]['path']).read_text()) for f in range(3)] for name,c in plan['controls'].items()}
    for name in plan['configs']:
        runs[name]=[json.loads((HERE/'results'/f'{name}_f{f}_s42.json').read_text()) for f in range(3)]
    rows=[];flat=[]
    for name,rs in runs.items():
        comparisons={}
        for cname in plan['controls']:
            comparisons[cname]={}
            for metric in ['auc','ap']:
                delta=[r['inner_'+metric]-c['inner_'+metric] for r,c in zip(rs,runs[cname])]
                comparisons[cname][metric]=dict(mean_delta=float(np.mean(delta)),fold_deltas=delta)
        row=dict(config=name,mean_inner_auc=float(np.mean([r['inner_auc'] for r in rs])),
            mean_inner_ap=float(np.mean([r['inner_ap'] for r in rs])),comparisons=comparisons,
            selection_score=min(comparisons[c][m]['mean_delta'] for c in comparisons for m in ['auc','ap']))
        if name in plan['configs']:
            row['branch_parameters']=rs[0]['branch_parameters']
            row['jointly_trained_branch_diagnostics']={component:{m:float(np.mean([r['branch_diagnostics'][component][m] for r in rs]))
                for m in ['auc','ap']} for component in ['branch_probability','base_probability']}
        rows.append(row)
        flat += [dict(config=name,fold=f,**{k:r[k] for k in ['inner_auc','inner_ap','best_epoch']}) for f,r in enumerate(rs)]
    best=max((r for r in rows if r['config'] in plan['configs']),key=lambda r:r['selection_score'])
    promote=best['selection_score']>=plan['promotion_minimum_both_gains_over_each_control']
    paired=[]
    for wd in (4,3):
        for mode in ['bag','chem']:
            name,comparator=f'gnn_wd{wd}',f'{mode}_wd{wd}'
            pair=dict(config=name,comparator=comparator)
            for m in ['auc','ap']:
                delta=[r['inner_'+m]-b['inner_'+m] for r,b in zip(runs[name],runs[comparator])]
                pair[m]=dict(mean_delta=float(np.mean(delta)),fold_deltas=delta)
            paired.append(pair)
    selection=dict(status='frozen_inner_selection',rows=rows,best=best,molecular_attribution_comparisons=paired,
        promote_to_outer_validation=promote,no_lncrna=True,external_data_used=False,
        outer_validation_evaluated=False,test_evaluated=False,
        plan_sha256=sha256(HERE/'results/plan.json'),verification_sha256=sha256(HERE/'verification.json'),
        result_hashes={p.name:sha256(p) for p in sorted((HERE/'results').glob('*_s42.json'))})
    (HERE/'selection.json').write_text(json.dumps(selection,indent=2))
    with (HERE/'inner_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
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
    pairing='\n'.join(f"| {r['config']} | {r['comparator']} | {r['auc']['mean_delta']:+.9f} | {r['ap']['mean_delta']:+.9f} |" for r in paired)
    diag='\n'.join(f"| {r['config']} | {r['branch_parameters']} | {r['jointly_trained_branch_diagnostics']['branch_probability']['auc']:.9f} | "
        f"{r['jointly_trained_branch_diagnostics']['branch_probability']['ap']:.9f} |" for r in rows if 'branch_parameters' in r)
    report=f'''# 原分子结构表示分支：内部三折实验

**原测试 AUC/AP 同时绝对 +0.02 尚未证实。** 完成 18 个新模型训练和全部重放，
6 个原对照也重放通过。仅原训练内部三折、seed42；未读取原验证或测试。

| 配置 | 内部 AUC | 内部 AP | 对纯 M123 ΔAUC | ΔAP | 对 unmasked10 ΔAUC | ΔAP |
|---|---:|---:|---:|---:|---:|---:|
{table}

预先固定的双对照、双指标最小平均增益选择：{best['config']}，分数 {best['selection_score']:+.9f}。
达到内部 +0.002 推进门槛：{promote}。该门槛不替代最终原测试
AUC≥0.9733929274、AP≥0.9336833737 的要求。

六组新配置的两项平均指标都低于单独重建对照。按预声明规则选中的也是
ChemBERTa 投影对照，不是分子消息分支。两项权重衰减下，gnn 的平均 AP 都
低于各自 bag 和 chem 对照；本轮没有支持键消息传播互补性的证据，不进入原验证或测试。

## 消融与解释边界

| 分子消息分支 | 匹配对照 | ΔAUC | ΔAP |
|---|---|---:|---:|
{pairing}

gnn 与 bag 参数及初始值完全相同，仅 gnn 传播键消息。bag 保留原子度、环与芳香性，
因此是消息传播消融，而非删除一切局部结构信息。chem 使用原 ChemBERTa 投影；
三种模式共享相同初始 RNA/配对头和分子大小标量，但 chem 输入投影参数量不同。
不能把任意附加分支的收益都归因于分子连接信息。

| 配置 | 附加分支参数数 | 联合训练的分支自身 AUC | AP |
|---|---:|---:|---:|
{diag}

分支自身结果使用所选联合 checkpoint，不是独立训练模型的对照，也不是最终测试成绩。
全部候选保留 M1/M2/M3 和同样的重建正则。主损失为融合 BCE，加 0.1 分支 BCE；
原重建权重爬升至 10。分支从零输出开始，初始预测与重建单独对照一致。
超参数、学习率、两项权重衰减和选轮规则在训练前固定，见 results/plan.json。

## 输入和核验

仅原有 216 个药物 SMILES、原表达/ChemBERTa、原训练配对。药物结构保留所有组分；
原子与键是已有药物的内部特征，不引入其他关联实体或监督。未使用 lncRNA 或外部数据，
未下载新预训练权重。原药物顺序和输入哈希均核验。

所有 checkpoint 的融合/分支/主干预测、指标、选轮、精确 float32 损失构成已重算。
每个新模型对 6 个训练正例和 2 个负例显式删记录、重建图，核验 M1/M2/M3 与最终 logit。
分子模式另验证原子/键顺序置换；bag 不使用键的性质在 CPU 上精确核对，
CUDA 汇聚采用浮点容差。所有新 checkpoint 内的分子输入缓冲区与冻结描述符精确相同。
移除辅助重建解码器不改变关联推理。

发布实现未修改；PLMF 的 10 个 checkpoint、图、划分及源码哈希与此前完整重放一致。
既有纯 miRNA M123+BPR 原测试均值仍为 0.9575892678/0.9233078960；目标未完成。
后续编码器注意力方案的精确删边推导及独立 NumPy 前向验证，位于
../midrug_exact_attention_20261006/；尚未实现 PyTorch 训练，也没有性能证据。
'''
    (HERE/'REPORT.md').write_text(report)
    snap=HERE/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,snap/p.name)
    state=dict(goal_status='active',completion_proven=False,stage='completed_and_audited',
        previous_turn='progress: degree-normalized encoder comparison and molecular input preparation',
        current_turn='progress: 18 original-molecular-input / atom-bag / ChemBERTa fits with complete audits',
        best=best,promote_to_outer_validation=promote,no_lncrna=True,new_candidate_test_evaluations=0,
        processes_expected_live=[],next_action='No promotion. Implement and independently verify the query-deleted attentive M123 encoder described in ../midrug_exact_attention_20261006/DERIVATION.md; NumPy algebra passed 240 cases, PyTorch/gradient checks remain.')
    (HERE/'CONTINUATION.json').write_text(json.dumps(state,indent=2))
    p=HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md';progress=p.read_text().split('最新完成')[0]
    progress+=f'''最新完成 [18 次原分子表示分支实验](midrug_molecular_encoder_20261006/REPORT.md)，全部重放通过。
内部选中 {best['config']}，AUC {best['mean_inner_auc']:.9f}、AP {best['mean_inner_ap']:.9f}，
相对纯 M123 与 unmasked10 的双指标最小增益 {best['selection_score']:+.9f}。推进原验证：{promote}。

[度归一化编码器](midrug_degree_encoder_20261006/REPORT.md)未超过单独重建；
[低秩补全混合](midrug_completion_20261006/REPORT.md)仅有小幅增益。
没有新候选进入原验证或测试。PLMF 的 10 个 checkpoint、冻结输入和发布源码哈希复核通过。
'''
    p.write_text(progress)
    print(json.dumps(dict(best=best,promote=promote,new_runs=18,attribution=paired),indent=2))


if __name__=='__main__':main()
