# MiDrug 查询相关注意力与原始分子结构实验

目标仍未达到。用户要求 AUC、AP 均相对原始 iNcRD-HG 绝对提高 0.02；
同一固定测试集的三种子均值目标为 AUC ≥0.9733929274、AP ≥0.9336833737。
本轮完成 17 个验证配置，全部 seed42，没有新候选测试评价，也未修改发布代码。

## 实验结果

| 实验族 | 配置 | 验证 AUC | 验证 AP |
|---|---|---:|---:|
| attention | attn16_deg_s42 | 0.956516482 | 0.919108445 |
| attention | attn16_reg_s42 | 0.956011786 | 0.918712345 |
| attention | attn16_s42 | 0.954671951 | 0.918103295 |
| attention | attn16_t3_s42 | 0.955534626 | 0.917692431 |
| attention | attn64_deg_s42 | 0.956684715 | 0.920146712 |
| attention | attn64_t3_s42 | 0.955766967 | 0.918503498 |
| attention | baseline_s42 | 0.959113111 | 0.925014802 |
| attention | degree05_s42 | 0.959499055 | 0.925207882 |
| chemistry | maccs_k10 | 0.958539572 | 0.922012004 |
| chemistry | maccs_k3 | 0.958222469 | 0.921430250 |
| chemistry | morgan2_k10 | 0.958160942 | 0.921115781 |
| chemistry | morgan2_k3 | 0.959311031 | 0.923918705 |
| chemistry | morgan3_k10 | 0.958302928 | 0.921761678 |
| chemistry | morgan3_k3 | 0.958621322 | 0.923431789 |
| chemistry | original_k3 | 0.959138496 | 0.924517806 |
| chemistry | pharm2_k10 | 0.959102784 | 0.924837040 |
| chemistry | pharm2_k3 | 0.958802892 | 0.922558309 |

六个学习注意力配置均低于静态权重对照。八个替代分子图也没有同时优于其原图对照：
Morgan2 k3 的 AUC 稍高、AP 较低；pharm2 k10 的 AP 稍高、AUC 较低。
这些结果不支持将上述变体纳入发布路径。

## 方法与边界

- 注意力：四个 M2 路径头使用原始表达谱/ChemBERTa 查询与键，
  权重为 exp(T*tanh(score))，查询投影从零开始，初始门与固定权重一致。
  只依赖节点原始协变量，不依赖当前查询标签；M1/M3 与原辅助图保留。
- 分子图：原 216 个药物 SMILES，重新计算 Morgan2/Morgan3/MACCS/feature-Morgan2
  Tanimoto，相似度 top-3/top-10 无向并集替换 DrugDrug 边。保留原自环重数，
  其他关系、全部节点特征、全部训练监督不变。没有下载外部数据或预训练模型。
- 来源核验：216 个 CID/SMILES/特征行顺序一致；原 ChemBERTa 行 min-max 与冻结
  药物特征完全相同；重新生成的原 Morgan2 count Tanimoto 与原 drugSimMat 最大误差
  2.956e-8。逐文件 SHA256 和 RDKit 版本见 chemical_provenance.json。
- 训练：RMSprop 1e-4，最多 700 epochs，最少 300；最优轮后 200 轮不改善即停。
  所有候选按 min(验证 AUC 增益, 验证 AP 增益) 选轮，锚点为原始模型同验证集
  三种子均值 0.9554707319/0.9181315885。

## 核验

verification.json：17 个已训练 checkpoint 预测重放、保存指标和双指标选轮重算通过。
每模型 12 对显式删边对照通过。注意力最终 logit 检查包含学习后的非均匀门；
通用中间池检查使用 forward 后已清除的门，不将其误述为学习门下的逐池核验。
初始检查还验证非零注意力梯度。分子图检查确认仅 DrugDrug 关系改变。
第一次审计在最后一个训练结果尚未落盘时停止；训练全部完成后完整重跑通过。

plmf_hash_recheck.json：先前已完整重推理的 10 个 PLMF checkpoint、冻结数据与发布代码
哈希均未改变。MDR/MDS 的既有测试均值仍为 0.9483092901/0.9436701562 与
0.9508978837/0.9490119744；这里是保留原路径的证据，不是新方法在 PLMF 上的验证。

## 继续方向

下一实验位于 ../midrug_profiles_20261006：从训练关联行/列中屏蔽当前配对后，
用非线性去噪重建分支预测目标，与 M1/M2/M3 联合训练。这个方向改变关联模式的
表示方式；不会继续放大本轮失败的注意力强度或分子邻居数。目标保持 active。

复现使用 /data1/licheng/miniconda3/envs/ncRNAdrug/bin/python。
check.py → launch.py；chemistry.py → launch_chem.py；audit.py → summarize.py。
已有训练结果拒绝覆盖；需在独立目录复跑。完整精度见 validation_summary.csv。
