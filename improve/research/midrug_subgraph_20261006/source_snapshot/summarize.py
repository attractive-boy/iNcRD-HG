"""Archive completed pure-miRNA subgraph evidence without test claims."""
import json,shutil
import numpy as np
from common import HERE,CLEAN,sha256


def main():
    verification=json.loads((HERE/'verification.json').read_text())
    assert verification['status']=='passed' and len(verification['records'])==15
    assert sum(not r['reused_control'] for r in verification['records'])==12
    selection=json.loads((HERE/'selection.json').read_text())
    assert json.loads((HERE/'plmf_hash_recheck.json').read_text())['status']=='passed'
    gap=json.loads((HERE/'fit_gap.json').read_text())['mean']
    coverage=json.loads((HERE/'context_coverage.json').read_text())['records']
    table='\n'.join(f"| {r['config']} | {r['mean_inner_auc']:.9f} | {r['mean_inner_ap']:.9f} | {r['mean_auc_gain']:+.9f} | {r['mean_ap_gain']:+.9f} |" for r in selection['rows'])
    component=[]
    for name in ('sg1_struct','sg1_cov','sg1_signed','sg2_cov'):
        rows=[r['component_metrics'] for r in verification['records'] if r['tag'].startswith(name+'_')]
        component.append(dict(config=name,**{key:float(np.mean([r[key] for r in rows])) for key in rows[0]}))
    (HERE/'component_summary.json').write_text(json.dumps(component,indent=2))
    ctable='\n'.join(f"| {r['config']} | {r['base_component_auc']:.6f} | {r['base_component_ap']:.6f} | {r['subgraph_component_auc']:.6f} | {r['subgraph_component_ap']:.6f} |" for r in component)
    retention={h:float(np.mean([r['mean_p3_retention'] for r in coverage if r['hops']==h and r['part']=='val'])) for h in (1,2)}
    decision=('达到内部推进门槛；下一步须冻结候选进入原验证集确认。' if selection['promote_to_outer_validation'] else
              '未达到预声明的内部双指标 +0.002 推进门槛，本轮候选不进入原验证集。')
    report='''# 纯 miRNA–drug 查询子图分支

**目标未达到，保持 active。** 最终要求固定原测试集的三种子均值达到
AUC ≥0.9733929274、AP ≥0.9336833737，且 PLMF 的 MDR/MDS 不下降。
本轮只使用纯 miRNA/drug 原始输入，完成 12 个新训练运行与 3 个既有对照重放。
没有使用 lncRNA，也没有读取原验证/测试标签选择或评价新候选。

## 结果：原训练集内部三折

| 配置 | 内部 AUC 均值 | 内部 AP 均值 | 相对对照 ΔAUC | 相对对照 ΔAP |
|---|---:|---:|---:|---:|
'''+table+'\n\n'+decision+'''

纯结构版本有小幅平均改善，但第二折两指标均下降，不能解释为稳定显著提升。
这些内部指标不能与原测试基线相减来计算最终提升；+0.002 也不是最终目标的替代。

## 方法

保留原 M1 查询删边、M2 简单路径汇聚、M3 MC-EASE 和原融合头，增加局部子图 logit。
所有子图先删除当前查询的正/负记录，再选择节点、构造边、距离与度数。
1-hop 最大各 16 个 miRNA/药物；2-hop 最大各 32 个，为一阶和二阶节点分配名额。
候选按查询相关共同邻居数/节点度排序，节点 ID 仅用于索引和稳定破同分。
节点类型、根节点角色、到两根节点的距离、全图/子图度数共 20 个结构特征。

子图 GNN 为 3 层、hidden 32、LayerNorm 与 0.1 dropout，分别比较纯结构、
原表达/ChemBERTa 协变量和原训练抽样负例关系。负关系不表示耐药/敏感方向。
末层读出零初始化，初始结果匹配 M123。所有新分支仅在当折原训练标签上联合 BCE 训练。
主干 RMSprop 1e-4；分支同学习率、weight decay 1e-4。最多 800 epochs，至少 300，
最优轮后 200 轮无改善停止；checkpoint 按内部 (AUC+AP)/2 选择。
四个独立工作进程（每 GPU 两个）；初始单进程完整步约 0.15 s，峰值分配显存约 0.89 GB。

## 核验

- initial_checks.json：两种截取方案各 3 正/3 负，删除监督记录重建后所有缓存数组
  精确一致；初始零分支与主干对照一致、非零读出下 12 对最终 logit 删边与梯度检查通过。
- verification.json：12 个新 checkpoint 预测、指标、选轮重算；每模型 6 个正例的
  M1/M2/M3/完整 logit 显式删边，另各 3 正/3 负独立重建局部输入与子图输出，均通过。
  三个既有纯 miRNA 对照也通过本轮 loader 重新推理，未混用含 lncRNA 的历史模型。
- plan.json 与各记录固定代码、缓存、输入和 checkpoint SHA256；全部图和参数
  只包含 miRNA/drug。common.py 只从独立白名单输入加载，不加载含 lncRNA 的原图。
- plmf_hash_recheck.json：先前完整重推理的 10 个 PLMF checkpoint、原源码和冻结
  输入哈希未变；两套 PLMF 图本身也均为 miRNA/drug。

## 对结果的进一步检查

'''+f'''context_coverage.json 显示，内部留出查询的非零 3-path 平均保留比例约为
1-hop {retention[1]:.4f}，2-hop {retention[2]:.4f}。这是子图分支的截取覆盖率；
原 M123 主干仍使用完整当折训练图及其 M2/M3 统计，不能据此称整个模型丢失了这些路径。

固定 checkpoint 的分量分解如下。子图分量是联合训练的残差输出，并非独立训练
分类器；它单独的 AUC/AP 只能作描述，不能直接证明是否有互补信息或定位因果问题。

| 配置 | 主干分量 AUC | 主干分量 AP | 子图分量 AUC | 子图分量 AP |
|---|---:|---:|---:|---:|
'''+ctable+f'''

fit_gap.json 在既有对照上测得平均训练 AUC/AP 为 {gap['train_auc']:.6f}/{gap['train_ap']:.6f}，
内部留出为 {gap['inner_auc']:.6f}/{gap['inner_ap']:.6f}。存在拟合差距，但不是过拟合原因
或后续方案有效性的证明。下一步检验训练内部嵌套折外预测驱动的残差学习，减少
残差目标对主模型训练内预测的依赖。精确防泄漏要求见 NEXT_EXPERIMENT.md；尚未执行。

既有合格纯 miRNA M123+BPR 测试三种子均值仍为 0.9575892678/0.9233078960，
低于双指标目标；本轮没有新候选测试成绩。PLMF 保持 MDR 0.9483092901/0.9436701562、
MDS 0.9508978837/0.9490119744。保留这些既有结果不等于完成 MiDrug 提升目标。

复现见 README.md；完整精度见 inner_summary.csv、selection.json，脚本快照在 source_snapshot/。
'''
    (HERE/'REPORT.md').write_text(report)
    target=json.loads((HERE/'TARGET.json').read_text())
    state=dict(goal_status='active',completion_proven=False,previous_goal_turn='progress: strict scope, 12 grouping runs and audits',
        current_turn='progress: 12 pure-miRNA subgraph runs, 3 reused controls replayed, query deletion and context/fitting diagnostics',
        current_target=target,inner_selection=selection['best'],promote_to_outer_validation=selection['promote_to_outer_validation'],
        no_lncrna=True,new_candidate_test_evaluations=0,processes_expected_live=[],
        next_action=('Promote selected model to frozen original validation' if selection['promote_to_outer_validation'] else
            'Nested training-only cross-fitted residual learning; never use current inner validation labels to fit residuals; see NEXT_EXPERIMENT.md'))
    (HERE/'CONTINUATION.json').write_text(json.dumps(state,indent=2,ensure_ascii=False))
    scope_path=HERE.parent/'ACTIVE_SCOPE.json';scope=json.loads(scope_path.read_text());scope['active_directory']=str(HERE)
    scope['latest_continuation']=str(HERE/'CONTINUATION.json');scope_path.write_text(json.dumps(scope,indent=2,ensure_ascii=False))
    snap=HERE/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,snap/p.name)
    current='''# MiDrug 专用 miRNA 模型：当前进度

**只使用 miRNA 与药物，不使用 lncRNA。双指标 +0.02 目标仍未完成。**
当前开发：[查询子图实验](midrug_subgraph_20261006/REPORT.md)，范围见 [ACTIVE_SCOPE.json](ACTIVE_SCOPE.json)。

| 项目 | AUC | AP |
|---|---:|---:|
| 此前确认的原始测试基线 | 0.9533929274 | 0.9136833737 |
| 必须达到的测试三种子均值 | 0.9733929274 | 0.9336833737 |
| 既有纯 miRNA M123+BPR 测试均值 | 0.9575892678 | 0.9233078960 |
| 需保住的 MDR 五折均值 | 0.9483092901 | 0.9436701562 |
| 需保住的 MDS 五折均值 | 0.9508978837 | 0.9490119744 |

最近两轮严格纯 miRNA 试验分别完成 12 个分组特征运行、12 个查询子图运行。
新子图最佳内部平均增益为 AUC +0.0001974434、AP +0.0005972455，未达到推进门槛。
这些是在原训练集内部三折得到的结果，不是原测试性能。本轮没有新候选测试。

所有新 checkpoint 已重放验证，并通过显式监督记录删除和参数/输入范围检查。
PLMF 的原源码、数据与 10 个 checkpoint 哈希与此前完整重推理一致，保留原结果。

后续检验[嵌套折外残差学习](midrug_subgraph_20261006/NEXT_EXPERIMENT.md)，仍严格限定
训练内选择，不使用 lncRNA 或外部数据；该方向尚未执行。

历史含 lncRNA 的配置继续排除出候选。其他历史报告保留用于审计，其中完整图最佳值
和 LncDrug 后续计划不适用于当前任务。不得以已保住 PLMF 或小幅内部增益宣称目标完成。
'''
    (HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md').write_text(current)
    print(json.dumps(dict(status='passed',new_runs=12,controls_replayed=3,best=selection['best'],promote=selection['promote_to_outer_validation'],goal_status='active')),flush=True)

if __name__=='__main__':main()
