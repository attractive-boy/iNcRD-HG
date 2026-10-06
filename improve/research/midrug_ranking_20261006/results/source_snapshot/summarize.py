"""Audit saved predictions/checkpoints and write validation/test result tables."""
import csv
import json
from pathlib import Path

import numpy as np

from run_ranking import (CONFIGS, HERE, SEEDS, group_metrics, load_fold, make_adapter,
                         metrics, seed_all, sha256, signature)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(records, config):
    rows = [r for r in records if r["config"] == config]
    out = dict(config=config, runs=len(rows))
    for key in ("auc", "aupr", "macro_drug_auc", "macro_drug_ap", "within_drug_auc"):
        values = [r[key] for r in rows]
        out[key] = float(np.mean(values))
        out[key + "_sd"] = float(np.std(values, ddof=1))
    return out


def main():
    output = HERE / "results"
    selection_file = output / "selection.json"
    selection = json.loads(selection_file.read_text())
    assert selection["signature"] == signature()
    chosen = selection["best_bpr"]
    records, audits = [], []
    data = load_fold("MiDrug", 0)
    for phase, configs, part in (("validation", CONFIGS, "val"), ("test", ("bce", chosen), "test")):
        for config in configs:
            for seed in SEEDS:
                tag = f"{config}_s{seed}"
                file = output / phase / f"{tag}.json"
                r = json.loads(file.read_text())
                assert r["signature"] == signature()
                if phase == "validation":
                    assert sha256(file) == selection["validation_result_hashes"][tag]
                    assert r["test_evaluated"] is False
                    assert set(r["graph_nodes"]) == {"miRNA", "drug"}
                    assert all("lnc" not in k.lower() for k in r["graph_edges"])
                    assert r["graph_edges"]["miRNA|MiDrug|drug"] == 3236
                    curves = json.loads(file.with_name(f"{tag}_curves.json").read_text())
                    assert len(curves) == 300
                    best = max(curves, key=lambda item: item["val_auc"])
                    assert best["epoch"] == r["best_epoch"]
                    assert not any("test" in k for row in curves for k in row)
                    assert sha256(file.with_suffix(".pt")) == r["checkpoint_sha256"]
                else:
                    assert r["selection_sha256"] == sha256(selection_file)
                    val = json.loads((output / "validation" / f"{tag}.json").read_text())
                    assert r["checkpoint_sha256"] == val["checkpoint_sha256"]
                predictions = file.with_name(f"{tag}_predictions.npz")
                if phase == "validation":
                    assert sha256(predictions) == r["predictions_sha256"]
                z = np.load(predictions)
                np.testing.assert_array_equal(z["pairs"], data[part + "_pairs"])
                np.testing.assert_array_equal(z["y"], data[part + "_y"])
                recalculated = metrics(z["y"], z["probability"])
                for key, value in recalculated.items():
                    np.testing.assert_allclose(value, r["metrics"][key], rtol=0, atol=1e-12)
                gm = group_metrics(z["pairs"], z["y"], z["probability"])
                for key in ("eligible_drugs", "macro_drug_auc", "macro_drug_ap", "within_drug_auc"):
                    np.testing.assert_allclose(gm[key], r["group_metrics"][key], rtol=0, atol=1e-12)
                records.append(dict(phase=phase, config=config, seed=seed, best_epoch=r["best_epoch"],
                                    **r["metrics"], **{k: gm[k] for k in gm if k != "per_drug"}))
                audits.append(dict(phase=phase, tag=tag, status="pass", prediction_sha256=sha256(predictions)))
    # Confirm exact query correction after training, for both reported models.
    import torch
    from check_modules import check
    trained = {}
    for config in ("bce", chosen):
        seed_all(42)
        adapter = make_adapter(data, "cuda:0")
        adapter.model.load_state_dict(torch.load(output / "validation" / f"{config}_s42.pt",
                                                map_location="cuda:0"))
        trained[config] = check(adapter, count=12)
    (HERE / "checks_trained.json").write_text(json.dumps(trained, indent=2))
    (output / "verification.json").write_text(json.dumps(dict(status="passed", records=audits), indent=2))
    write_csv(HERE / "per_run.csv", records)
    validation = [r for r in records if r["phase"] == "validation"]
    test = [r for r in records if r["phase"] == "test"]
    val_summary = [aggregate(validation, c) for c in CONFIGS]
    summary = [aggregate(test, c) for c in ("bce", chosen)]
    write_csv(HERE / "validation_summary.csv", val_summary)
    write_csv(HERE / "test_summary.csv", summary)
    base, new = summary
    paired = []
    for seed in SEEDS:
        b = next(r for r in test if r["config"] == "bce" and r["seed"] == seed)
        n = next(r for r in test if r["config"] == chosen and r["seed"] == seed)
        paired.append(dict(seed=seed, delta_auc=n["auc"] - b["auc"], delta_ap=n["aupr"] - b["aupr"],
                           delta_macro_drug_auc=n["macro_drug_auc"] - b["macro_drug_auc"],
                           delta_within_drug_auc=n["within_drug_auc"] - b["within_drug_auc"]))
    write_csv(HERE / "paired_delta.csv", paired)
    auc_delta, ap_delta = new["auc"] - base["auc"], new["aupr"] - base["aupr"]
    if auc_delta > 0 and ap_delta > 0:
        verdict = "本轮排序训练在测试集上同时提高 AUC 与 AP，可保留为候选改进；幅度和逐种子一致性见下表。"
    else:
        verdict = "本轮排序训练没有同时提高测试 AUC 与 AP，未达到原目标，保留纯 BCE 为默认。没有根据测试结果改选其他权重。"
    lines = ["# MiDrug：仅 miRNA–drug 的排序训练实验", "", verdict, "",
             "本轮已完成 15 次训练、验证集冻结选型，以及 6 次测试 checkpoint 评估。未使用 lncRNA 或外部数据。", "",
             "## 测试结果", "",
             "| 配置 | AUC（均值 ± SD） | AP（均值 ± SD） | 逐药物宏 AUC | 药物内配对加权 AUC |",
             "|---|---|---|---|---|"]
    for s in summary:
        name = "M1+M2+M3，BCE" if s["config"] == "bce" else f"M1+M2+M3，{chosen}"
        lines.append(f"| {name} | {s['auc']:.6f} ± {s['auc_sd']:.6f} | {s['aupr']:.6f} ± {s['aupr_sd']:.6f} | {s['macro_drug_auc']:.6f} | {s['within_drug_auc']:.6f} |")
    lines += ["", f"排序相对本轮同图 BCE：ΔAUC = {auc_delta:+.6f}；ΔAP = {ap_delta:+.6f}。", "",
              "| 种子 | ΔAUC | ΔAP | Δ逐药物宏 AUC |", "|---|---|---|---|"]
    for p in paired:
        lines.append(f"| {p['seed']} | {p['delta_auc']:+.6f} | {p['delta_ap']:+.6f} | {p['delta_macro_drug_auc']:+.6f} |")
    lines += ["", "## 验证集选择（测试前冻结）", "",
              "| 配置 | 验证 AUC | 验证 AP | 最佳轮次（42/43/44） |", "|---|---|---|---|"]
    for s in val_summary:
        epochs = selection["validation_summary"][s["config"]]["best_epochs"]
        lines.append(f"| {s['config']} | {s['auc']:.6f} | {s['aupr']:.6f} | {epochs} |")
    lines += ["", f"按三种子平均验证 AUC 选择 `{chosen}`；AP 只用于 AUC 完全相同时破同分。", "",
              "## 数据与方法", "",
              "- 沿用冻结 MiDrug train/val/test 配对与标签；训练 9,712 对（3,236 正例、6,476 负例），验证和测试各 3,234 对（1,078 正例）。",
              "- 图只含 605 个 miRNA 和 216 个 drug。保留 938 维 miRNA 表达谱、768 维发布药物特征，以及原 MiMi/DrugDrug 相似关系；不读取外部扩展特征。",
              "- 新建图只复制 miRNA/drug 特征和相似边，彻底排除 lncRNA 节点、LncMi/LncDrug/LncLnc 及反向关系；目标边始终仅为 3,236 个训练正例及反向边。",
              "- 保留 M1 查询删边修正、M2 co/knn/dknn、M3 两侧 EASE（λ=10/50）和原融合头。",
              "- 300 epochs、RMSprop(lr=1e-4, alpha=.99)、无学习率衰减。前 30 轮仅 BCE，31–60 轮线性增加 BPR 权重，60 轮后固定。",
              "- 损失为 BCE + λ·mean(softplus(s_negative − s_positive))。每个训练正例的两个负例都来自原训练负例，并且药物 ID 相同。",
              "- mixed_middle：每个正例每轮有放回抽 8 个候选，一个均匀负例和一个分数排名中间 50% 的负例；uniform 对照只用两个均匀负例。所有 3,236 个正例均有候选。",
              "- 同药物排序仍使用公开采样负标签，不能把它们全部解释为实验真阴性；未引入新的未标注负例。",
              "- 模型训练阶段不构建测试 pair 特征、不计算测试指标。15 次验证完成后写 selection.json，冻结配置与 checkpoint 哈希，随后仅评估 BCE 和选中的 BPR。", "",
              "## 解释边界", "",
              "- 本轮已移除 lncRNA，不能把此前含 lncRNA 的 0.9577/0.9225 当作排序训练的配对对照。应使用本轮同图 BCE。",
              "- 同一固定划分的 3 个初始化种子，SD 只反映初始化波动；不是 3 个独立数据划分，不据此宣称统计显著。",
              "- 逐药物指标仅包含同时有正负例的药物；宏 AUC 对药物等权，药物内加权 AUC 对同药物正负比较数加权。两者不同于全局 AUC。",
              "- 本次只搜索有限的同药物排序配方，结果不能证明所有排序训练必然有效或无效。", "",
              "## 核验与复现", "",
              "- checks_initial.json：无 lncRNA、仅训练负例、同药物配对、BPR 梯度方向、留出扰动不变性、12 对显式删边等价性。",
              "- checks_trained.json：BCE 与选中 BPR 的种子 42 checkpoint，各 12 对显式删边等价性。",
              "- results/verification.json：15 次验证和 6 次测试的预测重算、checkpoint/划分核验、选轮与冻结选择核验。",
              "- per_run.csv / paired_delta.csv / validation_summary.csv / test_summary.csv：完整精度结果。",
              "- 命令和代码入口见 README.md；results/plan.json 与 results/source_snapshot 保存预声明配置和训练源代码。", ""]
    (HERE / "REPORT.md").write_text("\n".join(lines))
    print(json.dumps(dict(verdict=verdict, chosen=chosen, test_summary=summary, paired=paired), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
