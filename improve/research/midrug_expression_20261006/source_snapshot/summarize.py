"""Archive expression controls and consolidate the dual-metric goal state."""
import csv,json,shutil
from model import HERE,RELEASE,CONFIGS,sha256
from incrd_hg_m123.data import load_graph,split_path


def main():
    audit=json.loads((HERE/'verification.json').read_text());assert audit['status']=='passed' and len(audit['records'])==8
    rows=[]
    for name in CONFIGS:
        r=json.loads((HERE/'results'/f'{name}_s42.json').read_text())
        rows.append(dict(tag=r['tag'],val_auc=r['val_auc'],val_ap=r['val_ap'],best_epoch=r['best_epoch'],test_evaluated=r['test_evaluated']))
    with (HERE/'validation_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    previous=HERE.parent/'midrug_internal_20261006';protection=json.loads((previous/'plmf_protection.json').read_text())
    for rel,digest in protection['source_hashes'].items():assert sha256(RELEASE/rel)==digest
    for ds in ('MDR','MDS'):load_graph(ds)
    for r in protection['records']:
        assert sha256(split_path(r['dataset'],r['fold']))==r['split_sha256']
        path=HERE.parent/'baselines_m123_20261004/results'/f"incrd_m123_{r['dataset']}_f{r['fold']}_s42.pt"
        assert sha256(path)==r['checkpoint_sha256']
    (HERE/'plmf_hash_recheck.json').write_text(json.dumps(dict(status='passed',records=10,
        scope='Prior full replay: protected sources, frozen data and checkpoint hashes still identical',
        prior_verification_sha256=sha256(previous/'plmf_protection.json')),indent=2))
    target=json.loads((HERE.parent/'midrug_signedpaths_20261006/TARGET.json').read_text())
    (HERE/'TARGET.json').write_text(json.dumps(target,indent=2,ensure_ascii=False))
    table='\n'.join(f"| {r['tag']} | {r['val_auc']:.9f} | {r['val_ap']:.9f} | {r['best_epoch']} |" for r in rows)
    best=max(rows,key=lambda r:min(r['val_auc']-.9554707318920146,r['val_ap']-.9181315884673041))
    report='''# MiDrug 原始表达预处理对照

目标未达到。AUC/AP 均需相对原始 iNcRD-HG 绝对提高至少 0.02；
固定测试集三种子均值须达到 0.9733929274/0.9336833737。
本轮 8 个配置只使用原始 RNA 表达，未增加外部数据，未测试新候选，未修改发布模型。

| 配置 | 验证 AUC | 验证 AP | 选中轮次 |
|---|---:|---:|---:|
'''+table+f'''

最佳双指标选择配置为 {best['tag']}，验证 AUC/AP = {best['val_auc']:.9f}/{best['val_ap']:.9f}。
这些单种子开发结果没有支持双指标 +0.02 的证据。

## 来源与方法

原 MiExp.csv / LncExp.csv 按各自 info 文件中的实体名及细胞 ID 对齐。
原代码对 miRNA 的样本列做过排序；两类 RNA 最终均为相同顺序的 938 个细胞。
按原行 min-max 重算后，605 个 miRNA 与 1,322 个 lncRNA 的冻结特征全部精确复现。
provenance.json 记录原文件、info 文件、转换脚本和数组 SHA256。
没有实体错配，也没有发现需要更改冻结输入的错误。

在相同原始表达数值上比较：原按行 min-max；整张表达矩阵的全局 min-max；
按细胞列 min-max；按 RNA 行 z-score；按细胞列 z-score；原特征附加表达均值、
标准差、最小值和最大值。附加统计量在实体间标准化，共增加 4 个输入维度。
mi_* 仅变 miRNA；rna_* 同时变两类 RNA。归一化仅使用原节点协变量，没有读取标签。
原关联图、原药物特征、原训练监督和冻结划分保持不变。

RMSprop 1e-4，最多 800 epochs，最少 300，连续 200 轮无改善停止。
按 min(验证 AUC 增益, 验证 AP 增益) 选轮，锚点 0.9554707319/0.9181315885。
全部为 seed42 开发试点；没有将 0.000x 量级差值解释为稳定增益。

## 核验

verification.json：8 个训练后 checkpoint 验证预测重放、指标与双指标选轮重算，
每模型 12 对最终 logit 显式删边检查，均通过；逐边检查原图不变、原药物特征与
监督不变。plmf_hash_recheck.json 确认此前完整重放的 10 个 PLMF checkpoint、
源代码、冻结数据仍与参考一致。PLMF 当前发布路径没有下降。

复现：prepare.py → launch.py → audit.py → summarize.py。
解释器 /data1/licheng/miniconda3/envs/ncRNAdrug/bin/python。
已有结果拒绝覆盖，新运行请放独立目录。完整精度见 validation_summary.csv。

后续优先验证原始 LncDrug 辅助监督与 MiDrug 共享药物表征的联合训练；先在训练
内部做选择，再提交冻结验证。这个方向尚未执行，不能写成已有改进。
目标保持 active；不降低双指标 +0.02 与 PLMF 保护标准。
'''
    (HERE/'REPORT.md').write_text(report)
    state=dict(goal_status='active',completion_proven=False,current_target=target,
        configurations_this_continuation=33,test_evaluations_this_continuation=0,best_expression=best,
        processes_expected_live=[],
        next_action='Evaluate original LncDrug auxiliary supervised transfer with training-internal selection; no external data')
    (HERE/'CONTINUATION.json').write_text(json.dumps(state,indent=2,ensure_ascii=False))
    snap=HERE/'source_snapshot';snap.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,snap/p.name)
    # The consolidated note distinguishes validation diagnostics from accepted test results.
    overall='''# MiDrug 双指标提升进度（2026-10-06）

**目标仍未达到，状态 active。** 用户明确要求 MiDrug 的 AUC 和 AP 都绝对提高至少 0.02，
不引入外部数据集，PLMF-MDA 的 MDR/MDS 现有 AUC/AP 均不得下降。

| 验收项 | AUC | AP |
|---|---:|---:|
| 原始 iNcRD-HG MiDrug 测试三种子均值 | 0.9533929274 | 0.9136833737 |
| 必须达到的 MiDrug 测试三种子均值 | 0.9733929274 | 0.9336833737 |
| 当前已发布 M123 MiDrug 测试三种子均值 | 0.9577362738 | 0.9225177204 |
| 需保住的 MDR 测试五折均值 | 0.9483092901 | 0.9436701562 |
| 需保住的 MDS 测试五折均值 | 0.9508978837 | 0.9490119744 |

本次续跑完成 **33 个新增验证配置**：8 个注意力/静态权重配置、9 个分子图配置、
8 个关联重建配置、8 个表达归一化配置。全部训练后 checkpoint 均已重放预测，
重算指标与选轮，完成显式查询删除检查。没有新候选进入测试，未替换发布代码。
此前两轮分别完成 46 与 35 个配置；这些轮次累计 114 个开发配置，不能视为
独立泛化证据。反复使用同一验证集会增加选择偏差，仍需冻结候选后做最终测试。

现有最佳实际 M123 验证候选仍是完整原图的 20% 上下文边丢弃：
AUC/AP = 0.9596866492/0.9253975204。此前最佳诊断混合为 0.9601638092/0.9266524107，
它不是通过验收的 M123 模型，也不是测试结果。新增方法均未带来足够幅度的证据。
不能将这些验证数字与上表测试基线相减来宣称提升。

具体报告：

- [注意力与分子图，17 配置](midrug_attention_20261006/REPORT.md)
- [非线性关联重建，8 配置](midrug_profiles_20261006/REPORT.md)
- [原始表达归一化，8 配置](midrug_expression_20261006/REPORT.md)
- [上一轮符号路径与排序目标，35 配置](midrug_signedpaths_20261006/REPORT.md)
- [上一轮内部信息与训练方法，46 配置](midrug_internal_20261006/REPORT.md)

只使用原冻结数据、原 SMILES 和原 RNA 表达；没有引入新增序列或方向/疾病注释。
原始表达看似不一致的问题已排除：按原代码对 miRNA 样本列排序后，两类 RNA
均精确复现冻结特征，938 个细胞列对齐，没有实体错配。

[PLMF 哈希复核](midrug_expression_20261006/plmf_hash_recheck.json) 已确认先前完整
重推理的 10 个参考 checkpoint、源码和冻结数据都未改变；原发布路径继续保留。
这不能替代新方法在 PLMF 上的验证，也不代表双指标目标已经完成。

下一方向：只用原始 LncDrug 辅助监督做共享药物表征的联合训练，并采用训练内部
选择，减少继续对固定验证集调参。该方向尚未执行。目标和验收标准保持不变。
'''
    (HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md').write_text(overall)
    print(json.dumps(dict(status='passed',configurations=8,best=best,goal_status='active')),flush=True)

if __name__=='__main__':main()
