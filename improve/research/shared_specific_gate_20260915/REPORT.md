# 从"门控 / 关系注意力 / 交叉注意力 / 分组 softmax"里选一个能提升的机制（2026-09-15）

## 结论（先给答案）

**选"门控"，但必须是保度数（degree-preserving）+ 可拒绝（abstain）的共享/专属通道融合，
而不是 softmax 归一化的注意力。** 这不是偏好问题，是本仓库已经量出来的：

在作者原始随机划分上，**把所有传入权重归一化成和为 1（即 softmax / 注意力 / 分组 softmax 的共同代数形式）
会让两个任务的测试 AUPR 掉 0.039–0.040**；而**把权重重新缩放到每个目标的入度（Σ_j a_ij = d_j）的门控，
相对"和为 1"的门控把 AUPR 拉回 +0.0245（MiDrug）/ +0.0340（LncDrug）**。
GATv2、HAN 式关系注意力都落在这个"丢掉度数尺度"的家族里，因此稳定为负。

来源：`improve/results/propagation_attention_20260907/REPORT.md`

| 机制（本仓库已实现/已测） | MiDrug Δ test AUC / AUPR | LncDrug Δ test AUC / AUPR | 结论 |
|---|---:|---:|---|
| `edge_uniform_mean`（求和→均值，和为 1） | −0.0257 / −0.0394 | −0.0179 / −0.0404 | 丢度数尺度 = 崩 |
| `edge_gate_degree` − `edge_gate_mean`（保度数门控 vs 和为 1 门控） | +0.0140 / +0.0245 | +0.0137 / +0.0340 | **保度数门控是唯一回升的** |
| `edge_gate_degree − base` | −0.0002 / −0.0001 | −0.0017 / +0.0017 | 保度数门控 ≈ 基线，不崩 |
| `relation_node`（HAN 式关系注意力）− base | −0.0001 / +0.0002 | −0.0046 / −0.0096 | 方案 A：负 |
| `gatv2_attention` − base | −0.0152 / −0.0251 | −0.0160 / −0.0340 | 交叉/边注意力：负 |
| HGT（关系级 K/Q/V + 注意力融合） | — | −0.030 AUC / −0.080 AUPR | 更负 |

复核：`fusion_gate`（`improve/complementary_attention_modules.py` 的 `ExpertRouter`）用的是
`masked_weights`，即 `softmax`（和恒为 1），所以它在 LncDrug 上 −0.0010 / −0.0021 并不意外；
MiDrug 的正号来自通道分支而非门控本身。验证集也没有选它。

## 为什么不是另外三个

- **方案 A 关系类型感知注意力**：本仓库就是 `relattn`（`improve/relattn.py`）/ `relation_node`，
  LncDrug 5/5 种子下降。它按关系做 softmax，正属于被否定的代数形式。
- **方案 C 分组 softmax + 语义级 β**：分组 softmax 仍把每组权重归一化，β 只是再次加权，
  仍然改变度数尺度；`neighbour_softmax` / `path_attention` 等近似形式都未稳定超过基线。
- **方案 B 交叉注意力**：miRNA↔lncRNA 的跨类型桥，本仓库作为先验 / 多任务损失 / 度数
  （`cross-task degree`、`joint multi-task`、GBDT 的 `bridge` 特征）都试过，均为负或 ns。
  representation 级 cross-attention 没直接试过，但它同样会用 softmax 对源端归一化——
  要么按保度数方式做，要么先接受"它大概率还是负"的预期。

一句话：**问题不在"共享 vs 专属"这个思想，而在融合算子。softmax 化 = 丢度数 = 掉 0.04 AUPR。**

## 推荐机制：保度数 + 可拒绝的共享/专属门控

```
专属通道 h_spec : 关系卷积的求和聚合（保留 d_j 的尺度）——现模型已有
共享通道 h_shared: cross-type / gene 中介元路径等残差证据
融合:            h = h_spec + g ⊙ h_shared
                 g = sigmoid(W [LN(h_spec) ‖ LN(h_shared) ‖ evidence]) ,  g 初值→0
边权（如需在传播内做）: a_ij = d_j · u_ij / Σ_{k→j} u_kj   （Σ_j a_ij = d_j）
```

两个关键点，都对应已测失败模式：

1. **不重新归一化 h_spec**。共享通道是残差，gate 只缩放"加多少"，不把专属通道的质量稀释掉。
2. **gate 初值接近 0**（`init_bias=-2`）。融合模型从专属基线出发，共享通道必须自己挣到打开它的梯度；
   这是"可拒绝注意力"的实测动机——没有可选项时注意力被迫分配总权重 1。

实现与契约测试已落盘：
- `degree_preserving_gate.py`：`degree_preserving_weights` + `SharedSpecificGate`
- `test_degree_preserving_gate.py`：7 项通过（入度和 = 度数、零共享恒等、门初值近闭、梯度可回传、门有界）

```bash
/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python -m unittest \
    improve.research.shared_specific_gate_20260915.test_degree_preserving_gate -v   # 7 passed
```

## 和 MPHGNN 元路径结合（这是唯一"有净空"的位置）

本仓库已把 MPHGNN 官方模型端到端接到 incrd-hg 图上，在 incrd-hg 自己的划分上没有增益——
因为 `<RNA→gene1~gene2→drug>` 编码的正是 `gene_cosine / drug_program_cosine / rna_pathway_*`
已经覆盖的"RNA 靶基因 × drug 靶基因"信息，边际≈0
（`improve/research/mirna_sota_20260915/REPORT.md`，B3 消融显示打乱/删除 gene–gene 边完全无影响）。

但**在 MPHGNN 自己的 miRNA-only 基准上我们输 0.017–0.027 AUC / 0.022–0.045 AUPR**，
差距被定位为"我们只用了二部图消息传递 + 表格打分，没有它的 gene 中介元路径归纳偏置"
（`improve/research/specialized_single_rna_20260915/REPORT.md`）。

所以正确的用法是把上面的门控放在**共享通道 = gene 中介元路径实例**上：

- 专属通道 = 现有 `RNA–drug` 关系卷积（保度数）。
- 共享通道 = `RNA→gene1~gene2→drug` 元路径实例池化（MPHGNN 的 `meta_path_instance` + BiTrans），
  作为**残差**进入，gate 由 `[z_RNA, z_drug, 元路径覆盖/计数]` 决定。
- **评估面要选对**：不要在作者随机划分上报增益（99.7% 的度量是比较药物度数，架构动不了，见
  `improve/RESULTS.md`）；应在 (a) MPHGNN 独立测试集、(b) 作者 leave-one-drug-out / 逐药物协议
  这两个"度数捷径失效"的面上评估。

## 判定门槛（避免把噪声当提升）

沿用本仓库的联合门槛：

- LncDrug：每个种子 AUC 与 AP 均不下降。
- MiDrug：两项均值提高且至少 2/3（或 2/5）种子提高。
- 必须跑三个对照并同时给出：`base`、`shared-only`（去掉专属通道）、`uniform`（gate 恒 0.5，不用 softmax），
  以及**保度数 vs 和为 1** 的成对对照——后两者是把"是不是门控在起作用"和"是不是度数尺度在起作用"分开。

## 限制

- 上述数字来自本仓库既有固定划分实验；保度数门控目前只做到"≈基线不崩"，还没有证明能超过基线。
- 真正的增益点（MPHGNN 基准、逐药物协议）本轮未重跑，属于待验证假设，不是已得结果。
- 本文件不是性能声明；`degree_preserving_gate.py` 只有形态与契约测试，没有端到端精度结果。
