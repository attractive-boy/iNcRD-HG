"""Audit saved validation metrics and summarize the unfinished improvement goal."""
import csv
import json
import shutil
from pathlib import Path
import numpy as np
from probe import HERE, RELEASE, score, sha256


def main():
    rows = []
    for family in ("probe", "kernel"):
        records = json.loads((HERE / family / "results.json").read_text())
        for r in records:
            tag = r["tag"] if family == "probe" else f"{r['kernel']}_l{r['ridge']}"
            pred = np.load(HERE / family / f"{tag}_val.npz")
            actual = score(pred["y"], pred["probability"])
            assert all(abs(actual[k]-r["metrics"][k]) < 1e-12 for k in actual)
            rows.append(dict(family=family, tag=tag, seed=42, **actual,
                             selected_epoch=r.get("best_iteration", ""), status="validation_only"))
    for family in ("fusion", "graphdrop", "fullgraph_drop"):
        for path in sorted((HERE / family).glob("*_s42.json")):
            r = json.loads(path.read_text())
            pred = np.load(path.with_name(path.stem+"_val.npz"))
            actual = score(pred["y"], pred["probability"])
            assert abs(actual["auc"]-r["val_auc"]) < 1e-12
            assert abs(actual["ap"]-r["val_ap"]) < 1e-12
            rows.append(dict(family=family, tag=r["tag"], seed=42, **actual,
                             selected_epoch=r["best_epoch"], status="validation_only"))
    assert len(rows) == 46
    checks = json.loads((HERE / "verification.json").read_text())
    assert checks["status"] == "passed" and len(checks["records"]) == 14
    protection = json.loads((HERE / "plmf_protection.json").read_text())
    for relative, digest in protection["source_hashes"].items():
        assert sha256(RELEASE / relative) == digest
    with (HERE / "validation_summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    probe = json.loads((HERE / "probe/results.json").read_text())
    best_blend = max((dict(tag=r["tag"], **b) for r in probe for b in r["m123_blends"]), key=lambda r:r["auc"])
    best_m123 = max((r for r in rows if r["family"] in ("fusion", "graphdrop", "fullgraph_drop")), key=lambda r:r["auc"])
    table = []
    for family in ("probe", "kernel", "fusion", "graphdrop", "fullgraph_drop"):
        candidate = [r for r in rows if r["family"] == family]
        best = max(candidate, key=lambda r:r["auc"])
        table.append(f"| {family} | {len(candidate)} | {best['tag']} | {best['auc']:.6f} | {best['ap']:.6f} |")
    plmf = []
    for dataset in ("MDR", "MDS"):
        r = [x for x in protection["records"] if x["dataset"] == dataset]
        plmf.append(f"| {dataset} | {len(r)} | {np.mean([x['auc'] for x in r]):.6f} | {np.mean([x['ap'] for x in r]):.6f} |")
    source = HERE / "source_snapshot"
    source.mkdir(exist_ok=True)
    for path in HERE.glob("*.py"):
        shutil.copy2(path, source / path.name)
    report = """# MiDrug 内部信息与训练方法搜索：目标尚未达成

本轮完成 46 个验证配置，没有新候选进入测试集，也没有修改发布模型。
所有方法仅使用冻结原始数据和训练监督；未读取 data_external，未下载数据或预训练模型。

## 验收口径

暂按相对原始 iNcRD-HG 的 **AUC 绝对增加 0.02** 执行：原始测试 AUC 0.9533929274，
目标测试 AUC ≥0.9733929274。AP 同时记录；若 AP 也要求增加 0.02，则目标为 ≥0.9336833737。
PLMF-MDA 的 MDR/MDS 相对当前 M1+M2+M3 的 AUC 和 AP 均不得下降。
当前发布 MiDrug M123 测试 AUC/AP 为 0.9577362738/0.9225177204。
本轮新结果全部是验证集结果，不能与上述测试值直接计算提升。

## 验证结果

| 实验族 | 配置数 | 该族验证 AUC 最优配置 | 验证 AUC | 验证 AP |
|---|---:|---|---:|---:|
""" + "\n".join(table) + f"""

探索性 XGBoost/M123 logit 混合最佳验证 AUC/AP 为
{best_blend['auc']:.6f}/{best_blend['ap']:.6f}（{best_blend['tag']}，混合权重 {best_blend['weight']}）。
这是验证集选出的诊断组合，既未完成多种子确认，也未作为最终 M123 模型。
该值仍不足以支持进入达标测试；不能据此声称测试提升。

## 改动与解释

- probe：12 个 XGBoost 配置。21 维原 M123 统计作为对照；扩展到正/负邻居证据、
  多尺度原始内容相似度、内容约束 MC-EASE（共 131 维），以及端点 PCA（195 维）。
  这些是信息诊断，不是擅自把最终 M123 换成树模型。
- kernel：20 个核岭回归配置。核由原始表达谱/ChemBERTa、节点身份形成，配对乘积核
  显式表示跨端点交互；训练只用 train 标签。没有引入新的外部特征。
- fusion：8 个 M123 配置。原 M1/M2/M3 保留，比较长期训练对照、query-hollow 端点张量
  交互、原始特征双线性交互、训练内拟合的低秩实体交互。该族统一梯度范数裁剪 10，
  最多 800 轮，至少 300 轮后连续 200 轮不改善即停；按验证 AUC 保存 checkpoint。
- graphdrop：两节点图上 0/10/20/40% 训练正边随机隐藏，10 个固定上下文，M1/M2/M3
  每个上下文都按对应图重新构建。最多 600 轮；完整训练图上验证。未隐藏任何监督标签，
  只隐藏消息图的边。20% 的 AUC 优于同族 0%，但 AP 没有同步增加，尚不能采用。
- fullgraph_drop：原始 lncRNA/miRNA/drug 全图上补做 0/20% 隐边配对对照。
  原始辅助关系属于同一数据集，不是外部数据；只有目标训练边被随机隐藏。
  20% 隐边验证 AUC/AP 为 0.959687/0.925398，对照为 0.959122/0.924324；
  单种子差值为 +0.000565/+0.001074，可继续确认，但远不足以证明目标达成。

probe/kernel/fusion/graphdrop 延续此前排序试验的 miRNA/drug 两节点图；fullgraph_drop
覆盖原发布图。不能把两种图的结果当成同图配对差，也没有将“禁止外部数据”扩大为
“禁止使用原始 lncRNA 辅助信息”。所有配置使用冻结 fold0；神经网络试点仅 seed42，
结果不能解释为跨划分泛化证据。反复验证选型也不能证明存在不可突破的性能上限。

## PLMF-MDA 性能保护

重新加载 2026-10-04 的全部 10 个 checkpoint，使用当前发布包和冻结划分重新计算预测：

| 数据集 | 折数 | 测试 AUC 均值 | 测试 AP 均值 |
|---|---:|---:|---:|
""" + "\n".join(plmf) + """

AUC 与参考完全相同，AP 均值的浮点差约 2e-8；最大单个预测误差 3.58e-7。
`plmf_protection.json` 保存逐折结果、checkpoint/划分哈希与发布源码哈希。
这证明现有发布路径得到保留，不代表未经测试的新方法已经在 PLMF 上通过。

## 核验与复现

- probe/checks.json：3 个训练正例和 3 个训练负例，逐一删除其监督记录后重新构建全部
  195 维特征；与解析屏蔽一致，最大绝对误差 3.85e-6。
- verification.json：全部 14 个神经网络 checkpoint 重新计算验证预测，核对选轮和
  预测指标。每模型 6 个训练正例做 M1/M2/M3/最终 logit 显式删边检查；6 个 graphdrop
  模型额外在更稀疏图各检查 6 对。全部通过。
- validation_summary.csv：46 个配置完整精度记录。所有新候选 test_evaluated=false。
- source_snapshot/：本轮脚本快照；各训练结果包含训练源码及 checkpoint SHA256。

环境：`/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python`。
在本目录运行 `probe.py`、`kernel_probe.py`、`launch_fusion.py`；
`graphdrop.py --drop {0.0,0.1,0.2,0.4}` 和 `fullgraph_drop.py --drop {0.0,0.2}`
分开启动。`verify.py`、`protect_plmf.py`、`summarize.py` 可重新核验已有产物。
训练入口拒绝覆盖已有同名神经网络结果；复跑训练需使用新的实验目录。

## 后续方向

本轮已排除所测试的简单内容扩展、低秩融合和隐边强度作为直接达标方案。
下一步优先考察 M2 内部的度校正路径权重、查询删边后的多步归一化游走，以及 M3
对训练正/负证据的联合建模；同时保留原图与两节点图对照。
仍须保持相同划分、训练内构图、验证选型、候选冻结后才测试的协议。
目标继续有效，尚未完成；不降低 0.02 的目标，不把仅保持 PLMF 性能视为完成。
"""
    (HERE / "REPORT.md").write_text(report)
    status = dict(goal_status="active", previous_goal_turn="progress: completed ranking experiment", current_turn="progress",
                  new_validation_configurations=len(rows), new_test_evaluations=0,
                  midrug_target_test_auc=.9533929274189016+.02,
                  midrug_target_proven=False, best_diagnostic_validation=best_blend,
                  best_m123_validation=best_m123,
                  plmf_reference_protection="10/10 checkpoint predictions reproduced",
                  next_action="Degree-corrected M2 paths and masked normalized walks; joint training-positive/negative M3 evidence",
                  release_modified=False)
    (HERE / "CONTINUATION.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
