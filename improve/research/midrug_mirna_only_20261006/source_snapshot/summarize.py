"""Archive the strict miRNA-only results and replace the active scope summary."""
import json,shutil
from model import HERE,sha256


def main():
    check=json.loads((HERE/'verification.json').read_text());assert check['status']=='passed' and len(check['records'])==12
    selection=json.loads((HERE/'selection.json').read_text())
    reference=json.loads((HERE/'reference_replay.json').read_text());assert reference['status']=='passed'
    protected=json.loads((HERE/'plmf_hash_recheck.json').read_text());assert protected['status']=='passed'
    target=json.loads((HERE/'TARGET.json').read_text())
    assert not target['lncrna_features_relations_supervision_allowed']
    table='\n'.join(f"| {r['config']} | {r['mean_inner_auc']:.9f} | {r['mean_inner_ap']:.9f} | {r['mean_auc_gain']:+.9f} | {r['mean_ap_gain']:+.9f} |" for r in selection['rows'])
    decision=('有配置达到预声明的训练内部双指标 +0.002 升级门槛，下一步须在原验证集确认。' if selection['promote_to_outer_validation'] else
              '没有配置达到预声明的训练内部双指标 +0.002 升级门槛，不推进到原验证集测试此组候选。')
    report='''# 仅 miRNA–drug：范围纠正与训练内部实验

**目标未达到，保持 active。** 用户明确要求“不要使用 lncRNA 我做的是专门的mirna的”。
本目录是当前开发入口，只允许 miRNA 与 drug 两种节点。
当前 lncRNA 辅助实验在用户纠正后立即终止；此前含 lncRNA 的历史配置不再作为候选。
原始 AUC/AP 各绝对增加 0.02 的数值目标不变：测试三种子均值须达到
**AUC ≥0.9733929274、AP ≥0.9336833737**。PLMF 两个数据集不得下降。

## 范围落实

- graph.pt 是新建的白名单对象：605 个 miRNA、216 个 drug，只包含节点 x、
  MiMi、DrugDrug 和空的 MiDrug 双向边；没有复制任何 lncRNA store、特征、关系或标签。
- 运行时目标边只从当次训练正例安装。训练进程仅加载独立的 graph.pt，
  不加载原含 lncRNA 的冻结图。模型参数中也不含 lncRNA 分支。
- inner_data.npz 仅保存原 MiDrug 训练对/标签与内部三折索引，没有 LncDrug 辅助监督。
  每折训练约 6,475 对、内部留出约 3,237 对；原验证和测试对不进入本轮候选训练。
- 原 miRNA 表达/ChemBERTa 特征与新图逐元素一致。原名称/SMILES 通过原表达列名、
  CID 和原特征行顺序核对；没有下载序列、注释或额外数据。
- 上一目录 midrug_multitask_20261006 的 launcher 和两个训练进程已终止（exit 143），
  STOPPED_BY_SCOPE.json 标记所有已有结果不合格，launcher 增加拒绝续跑检查。

## 4 配置 × 3 内部折

M1/M2/M3 与融合头保留。候选只为 M3 增加由原名称/分子结构分组产生的查询无关证据。
RNA 分组为去除 3p/5p 后的名称词干与名称编号；它们并非经过序列验证的生物家族。
药物分组为最大分子片段、Murcko scaffold 与 generic scaffold；空 scaffold 的无环分子
各自独立，避免把所有无环药物误归一类。结构相同不等于响应相同。

只用原训练正负标签累计同组其他实体的支持数和比例；所有分组核对角线为零，
当前配对标签不能进入 RNA、药物或交叉分组统计。rna/drug/combined 分别增加
10/15/55 个统计通道。新输入权重从零初始化，以相同 M123 初始参数作对照。

| 配置 | 内部 AUC 均值 | 内部 AP 均值 | 相对对照 ΔAUC | 相对对照 ΔAP |
|---|---:|---:|---:|---:|
'''+table+'\n\n'+decision+'''

所有模型 seed42、RMSprop 1e-4、最多 800 轮、至少 300 轮后最优轮连续 200 轮不改善停止。
Checkpoint 按内部 (AUC+AP)/2 选轮，配置按相对对照的两指标平均增益最小值选择。
这是原训练集内部的三折开发评估，不是最终测试成绩，也不改变 +0.02 的最终验收线。

## 核验

initial_checks.json：严格节点/关系/参数白名单、内部划分隔离、初始对照一致性、
3 正/3 负查询删除全部监督记录后重算 55 通道（误差为零）、12 对最终 logit 显式
删边、新增通道梯度检查通过。verification.json：12 个训练后 checkpoint 全部重放，
指标/选轮重算，每模型 8 对显式删边检查通过。

运行中选择脚本最初名为 select.py，与 Python 标准库重名，导致后续子进程在导入时
退出；已改为 choose.py，确认原 launcher 终止后续跑未完成配置，已完成结果未覆盖。
这次修复不改变训练/模型/特征脚本及既有结果哈希。

reference_replay.json：用独立双类型输入重新推理此前已评测的 6 个纯 miRNA checkpoint，
复现 BCE 与 BPR 的原测试结果；这只是历史参考重放，没有选择或测试新候选。
BCE AUC/AP 均值为 0.9572658546/0.9222631628；既有 BPR 为
0.9575892678/0.9233078960，仍低于目标。不能用此前含 lncRNA 的完整图最佳值代表本任务。

plmf_hash_recheck.json：原 10 个 PLMF checkpoint、发布源码和冻结数据哈希与先前完整
重推理一致；两套 PLMF 图本身都只有 miRNA/drug。保留的 MDR AUC/AP 为
0.9483092901/0.9436701562，MDS 为 0.9508978837/0.9490119744。

## 复现与后续

解释器 /data1/licheng/miniconda3/envs/ncRNAdrug/bin/python。
初次准备：data.py → groups.py；训练：check.py → launch.py；
训练完成：audit.py → choose.py → replay_reference.py → summarize.py。
已有准备文件与训练结果拒绝覆盖；新运行请建独立目录。

后续必须沿用本目录的 miRNA/drug 白名单数据。若当前配置不升级，下一方向是查询
子图分支：只在 miRNA–drug 训练图中构造局部结构，显式删除当前配对后提取，检验
多个路径之间的结构信息；先用同一训练内部协议与纯 miRNA M123 对照。该方向尚未执行。
'''
    (HERE/'REPORT.md').write_text(report)
    state=dict(goal_status='active',completion_proven=False,previous_goal_turn='progress: 33 tested configurations; those using lncRNA now ineligible',
        current_turn='progress: stopped prohibited auxiliary work; strict miRNA-only inputs and 12 inner-fold runs; scope and deletion audits',
        current_target=target,eligible_reference=reference['means']['bpr010_mixed'],inner_selection=selection['best'],
        promote_to_outer_validation=selection['promote_to_outer_validation'],new_candidate_test_evaluations=0,
        no_lncrna=True,processes_expected_live=[],next_action=('Promote inner-selected configuration to frozen original validation' if selection['promote_to_outer_validation'] else
          'Evaluate a query-subgraph branch on strict miRNA/drug training graph with explicit query deletion and inner validation'))
    (HERE/'CONTINUATION.json').write_text(json.dumps(state,indent=2,ensure_ascii=False))
    scope=dict(user_instruction=target['user_scope_clarification'],allowed_node_types=['miRNA','drug'],forbidden=['lncRNA nodes','lncRNA features','lncRNA relations','LncDrug auxiliary supervision'],
        active_directory=str(HERE),target=target,goal_status='active')
    (HERE.parent/'ACTIVE_SCOPE.json').write_text(json.dumps(scope,indent=2,ensure_ascii=False))
    snap=HERE/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,snap/p.name)
    current='''# MiDrug 专用 miRNA 模型：当前进度

**当前范围：只使用 miRNA 和 drug，不使用 lncRNA。目标尚未完成。**
用户已明确禁止 lncRNA。当前开发入口为 [midrug_mirna_only_20261006](midrug_mirna_only_20261006/REPORT.md)，
范围记录见 [ACTIVE_SCOPE.json](ACTIVE_SCOPE.json)。所有含 lncRNA 的历史候选均已排除。

| 验收项 | AUC | AP |
|---|---:|---:|
| 此前确认的原始测试基线 | 0.9533929274 | 0.9136833737 |
| 两项各 +0.02 的测试目标 | 0.9733929274 | 0.9336833737 |
| 既有纯 miRNA M123 + BPR（三种子，已复现） | 0.9575892678 | 0.9233078960 |
| 需保住的 MDR 五折均值 | 0.9483092901 | 0.9436701562 |
| 需保住的 MDS 五折均值 | 0.9508978837 | 0.9490119744 |

当前已经：终止 LncDrug 辅助实验；建立不含 lncRNA 的独立输入；完成 12 个内部三折
对照运行与训练后核验；复现 6 个既有纯 miRNA 参考 checkpoint。没有新候选测试评价。

'''+decision+'\n\n'+'''详细数值、来源与删除检查见[本轮报告](midrug_mirna_only_20261006/REPORT.md)。
PLMF 参考源代码、数据与 checkpoint 哈希未变，两套图也均为 miRNA/drug。

历史含 lncRNA 的完整图结果不再作为本任务最佳候选。历史报告保留用于审计，
其中出现的完整图最佳值、LncDrug 后续计划均不适用于当前范围。
当前已知纯 miRNA 模型距离双指标目标仍有差距；不降低标准，不宣称目标已经达成。
'''
    (HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md').write_text(current)
    print(json.dumps(dict(status='passed',runs=12,promote=selection['promote_to_outer_validation'],best=selection['best'],goal_status='active')),flush=True)

if __name__=='__main__':main()
