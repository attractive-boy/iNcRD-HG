# -*- coding: utf-8 -*-
"""Render REPORT_MPHGNN_ON_PLMF.md from results/ (run after summarize.py)."""
import glob, json, os
import numpy as np, pandas as pd
import summarize as S

HERE = os.path.dirname(os.path.abspath(__file__))
g = {ds: json.loads(str(dict(np.load(os.path.join(HERE, 'cache', f'graph_{ds}.npz')))['summary'])) for ds in ('MDR', 'MDS')}
dt = json.load(open(os.path.join(HERE, 'cache', 'drug_targets.json')))


def fmt(x, nd=4):
    return '-' if x is None or (isinstance(x, float) and x != x) else f'{x:.{nd}f}'


def honest(ds, split):
    return S.honest(ds, split)


def val_rows(ds, split):
    fs = sorted(glob.glob(os.path.join(HERE, 'results', f'mphgnn_kmer_val_{ds}_{split}_seed*.csv')))
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    return dict(auc=d.valsel_auc.mean(), auc_sd=d.valsel_auc.std(ddof=0) if len(d) > 1 else 0, aupr=d.valsel_aupr.mean(),
                ep=d.valsel_epoch.mean(), best_auc=d.auc.mean(), best_aupr=d.aupr.mean(), last_auc=d.last_auc.mean(), last_aupr=d.last_aupr.mean(), n=len(d))


L = []
L.append('# MPHGNN 放到 PLMF-MDA 的 MDR/MDS 数据集上：能跑到多少\n')
L.append('目录：`improve/research/mphgnn_plmf_20260921/`\n')
L.append('> **MPHGNN**：Huang G, Peng Y, Wu S, Wei H, Liu S. *MPHGNN: metapath-guided heterogeneous graph neural network for miRNA-drug resistance association prediction.* **Briefings in Bioinformatics 2026**, bbag013. 代码/数据 <https://github.com/Gtaizu/MPHGNN>。原文在自建独立测试集上 AUC 0.978 / AUPR 0.937（本机已用官方代码复现到 0.9743/0.9382，见 `../mirna_sota_20260915/`）。\n>\n> **PLMF-MDA**：Sheng N 等, *PLoS Comput Biol* 2026, e1013968；MDRdataset 1317 miRNA × 105 drug × 5411 耐药关联，MDSdataset 1252 × 140 × 5054 敏感关联；只提供 miRNA 序列 + drug SMILES + 关联矩阵。\n')

# ---------------- 0. conclusion ----------------
main = {(ds, sp): S.ours(ds, sp, 'kmer', 'real') for ds in ('MDR', 'MDS') for sp in ('5cv', 'single')}
plmf = {(ds, sp): S.baseline(ds, sp, 'PLMF-MDA') for ds in ('MDR', 'MDS') for sp in ('5cv', 'single')}
mldti = {(ds, sp): S.baseline(ds, sp, 'ML-DTI') for ds in ('MDR', 'MDS') for sp in ('5cv', 'single')}
strong = {(ds, sp): S.strongest(ds, sp) for ds in ('MDR', 'MDS') for sp in ('5cv', 'single')}
val = {(ds, sp): val_rows(ds, sp) for ds in ('MDR', 'MDS') for sp in ('5cv', 'single')}
L.append('## 0. 一句话结论\n')
L.append('| 数据集 | 协议 | MPHGNN 同基线口径（测试集最优轮）AUC / AUPR | MPHGNN 无泄漏口径（验证选轮 + 逐对打分）AUC / AUPR | MPHGNN 末轮(50) AUC / AUPR | PLMF-MDA 复现 | ML-DTI 复现 | 本文最强模型 |')
L.append('|---|---|---|---|---|---|---|---|')
for ds in ('MDR', 'MDS'):
    for sp in ('5cv', 'single'):
        m, p, q, s, v = main[(ds, sp)], plmf[(ds, sp)], mldti[(ds, sp)], strong[(ds, sp)], val[(ds, sp)]
        h = honest(ds, sp)
        vcell = (fmt(h['iso_valsel_auc']) + ' / ' + fmt(h['iso_valsel_aupr'])) if h else '(运行中)'
        L.append(f"| {ds} | {'论文 5 折' if sp == '5cv' else '官方 80/20'} | **{fmt(m['auc'])} / {fmt(m['aupr'])}** | "
                 f"{vcell} | {fmt(m['last_auc'])} / {fmt(m['last_aupr'])} | "
                 f"{fmt(p['auc'])} / {fmt(p['aupr'])} | {fmt(q['auc'])} / {fmt(q['aupr'])} | {fmt(s['auc'])} / {fmt(s['aupr'])} |")
L.append('')
L.append('（"测试集最优轮"与 `../plmf_baselines_20260920` 各基线的主口径一致，偏乐观；"无泄漏口径"= 从训练对再切 10% 做早停选轮 + 测试对逐对打分（MPHGNN 官方全局分支会把同一 batch 里的查询对连成图，整批打分时测试集的度结构会泄漏进预测，见 §4）；"末轮"是固定 50 轮。）\n')

# ---------------- 1. what was built ----------------
L.append('## 1. 迁移了什么\n')
L.append('MPHGNN 需要 miRNA–gene–drug 异质网络 + 节点属性；PLMF-MDA 数据里一样都没有，全部按 MPHGNN 原文的同一来源重新构建（DrugBank 本机 403，改用 ChEMBL 机制靶点）。\n')
L.append('| MPHGNN 原文 | 本次 PLMF-MDA 节点上的构建 | MDR | MDS |')
L.append('|---|---|---|---|')
L.append(f"| miRNA–gene：ENCORI 827 条 | ENCORI miRNATarget API（hg38, mRNA）逐 miRNA 抓取 1620 个 miRNA、154 万行；保留 clipExpNum≥5 且 ≥3 个预测程序支持，每 miRNA 取 CLIP 支持最高的 50 个基因 | {g['MDR']['mirna_gene_edges']} 条，{g['MDR']['mirna_with_gene']}/{g['MDR']['n_mirna']} miRNA 有靶基因 | {g['MDS']['mirna_gene_edges']} 条，{g['MDS']['mirna_with_gene']}/{g['MDS']['n_mirna']} |")
L.append(f"| gene–drug：DrugBank 579 条 | SMILES→InChIKey→ChEMBL 分子（159/164 命中）→机制靶点（含盐型子分子）→UniProt→Entrez；{sum(bool(d['genes']) for d in dt['drugs'].values())}/164 药有靶点 | {g['MDR']['gene_drug_edges']} 条，{g['MDR']['drug_with_gene']}/{g['MDR']['n_drug']} 药 | {g['MDS']['gene_drug_edges']} 条，{g['MDS']['drug_with_gene']}/{g['MDS']['n_drug']} 药 |")
L.append(f"| gene–gene：HumanNet 5069 条 | HumanNet v3 FN（97.7 万条 LLS 边），限制在上面两类边覆盖的基因集合内，每基因取 LLS top-10 并对称化 | {g['MDR']['n_gene']} 基因，{g['MDR']['gene_gene_edges_topk']} 条 | {g['MDS']['n_gene']} 基因，{g['MDS']['gene_gene_edges_topk']} 条 |")
L.append(f"| miRNA 属性：CCLE 938 细胞系表达 | 主变体 **kmer**：2+3-mer 频率 80 维（所有节点可用）；对照变体 **expr**：CCLE 表达（仅 {g['MDR']['mirna_with_ccle_expr']}/{g['MDR']['n_mirna']}、{g['MDS']['mirna_with_ccle_expr']}/{g['MDS']['n_mirna']} 个 miRNA 有，其余零向量+缺失标记） | | |")
L.append('| drug 属性：ChemBERTa 768 维行内 min-max | ChemBERTa-zinc-base-v1 mean-pool 768 维 + 行内 min-max（已核对：MPHGNN 官方 `drug.x` 就是 incrd-hg 的 `drug_features-chemBertA.csv` 行内 min-max，逐位相等） | | |')
L.append('| gene 属性：gene similarity 200 维 | HumanNet 对称归一化邻接（含自环）的 200 维截断 SVD | | |')
L.append('')
L.append('模型代码是官方 `model.py` / `layer.py` 原样（`mphgnn_model.py` 只把写死的 605-miRNA 偏移改成参数）；超参照官方：hidden 128、3 层 HeteroGCN（sum 聚合 + PReLU）、层注意力、BiTrans 元路径聚合、MIL 注意力包预测 + top-3 实例预测 + 全局 GCN-GAE 边解码三路平均、k=20（每对 441 条元路径实例）、batch 256、Adam lr 1e-4、pos_weight BCE、无 dropout/BN。\n')

# ---------------- 2. protocol ----------------
L.append('## 2. 协议（与 PLMF-MDA 基线复现逐位相同）\n')
L.append('- 划分：`plmf_splits.py` 复制 `../plmf_baselines_20260920/plmf_data.py` 的两种协议并断言逐对相等：**5cv** = 论文 5 折（正例 + 等量随机负例，seed 42），**single** = 官方代码 80/20（seed 0–4）。')
L.append('- 图：每个 split 的消息传递 miRNA–drug / drug–miRNA 边**只用训练正例**；测试推理用同一张图（不像 MPHGNN 官方 `load_test_data` 那样把测试正边也放进图里）。基因边与标签无关，全图共享。')
L.append('- 训练 50 轮；每轮在测试集上算 AUC/AUPR/ACC 等。报三种口径：(a) **测试集最优轮**（本项目基线统一口径，偏乐观）；(b) **验证集选轮**：从训练对再切 10%（分层）做验证，取验证 AUC 最高轮的测试指标（无泄漏）；(c) **末轮**。另存训练损失最小轮（官方 `check_metric=loss` 口径，实际就是末轮附近）。')
L.append('- 消融：`gene-none` 去掉全部基因边（无元路径可走，实例退化为随机基因自环填充）；`gene-shuffled` 随机置换基因 id（边数不变、生物学对应被打乱）。\n')

# ---------------- 3. leaderboards ----------------
L.append('## 3. 完整榜单（测试集最优轮口径，与 `../plmf_strongest_20260920` 榜单同口径）\n')
for ds in ('MDR', 'MDS'):
    for sp in ('5cv', 'single'):
        L.append(f"### {ds}dataset · {'论文 5 折 (seed42, 5 折均值±折间SD)' if sp == '5cv' else '官方 80/20 (seed 0-4 均值±SD)'}\n")
        L.append('| 排名 | 方法 | AUC | ± | AUPR | ACC | 末轮 AUC / AUPR | 论文值 AUC / AUPR |')
        L.append('|---|---|---|---|---|---|---|---|')
        for i, e in enumerate(S.board(ds, sp), 1):
            p = e.get('paper') or (None, None)
            name = f"**{e['method']}**" if e['kind'] in ('mphgnn', 'ours') else e['method']
            L.append(f"| {i} | {name} | {fmt(e['auc'])} | {fmt(e['auc_sd'])} | {fmt(e['aupr'])} | {fmt(e['acc'])} | {fmt(e.get('last_auc'))} / {fmt(e.get('last_aupr'))} | {fmt(p[0])} / {fmt(p[1])} |" + ('  ⚠ 未跑完' if not e.get('complete', True) else ''))
        L.append('')

# ---------------- 4. honest selection ----------------
L.append('## 4. 无泄漏口径：验证集选轮 × 逐对打分（主变体 kmer/real）\n')
L.append('MPHGNN 官方 `model.forward` 的全局分支（GCN-GAE）用**当前 batch 里的 (miRNA, drug) 查询对**作为 GCN 的 edge_index。测试时整批打分，测试集自身的度结构（正例集中在高频 miRNA/药物、负例在负例池均匀采样）就成了可读的信号：只用"测试 batch 内 miRNA 度 × drug 度"这一个启发式，1024 对一批时 AUC 就有 0.83–0.89，整批更高；逐对打分（batch=1）时该启发式为 0.5。下面同一组训练好的模型分别用两种方式打分：\n')
lk = pd.read_csv(os.path.join(HERE, 'results', '_query_graph_leak.csv'))
L.append('仅用"打分 batch 内 miRNA 度 × drug 度"这一启发式在测试集上的 AUC（越大说明整批打分时可读的测试集结构越多）：\n')
L.append('| 数据集 | 协议 | batch=1 | 8 | 32 | 128 | 256 | 1024（本 runner） | 整个测试集（官方） | 参考：训练正例图的度 |')
L.append('|---|---|---|---|---|---|---|---|---|---|')
for _, q in lk.iterrows():
    L.append(f"| {q['dataset']} | {q['split']} | {q['1']:.3f} | {q['8']:.3f} | {q['32']:.3f} | {q['128']:.3f} | {q['256']:.3f} | {q['1024']:.3f} | {q['full']:.3f} | {q['train-graph']:.3f} |")
L.append('')
L.append('把全局分支的 edge_index 换成固定的训练正例图（`--global_graph train`，无泄漏的"正常" GAE 做法）反而更差：测试最优轮只有约 0.917（MDR/MDS 80/20 与 5 折都在 0.916–0.922），且 50 轮后掉到 0.85——官方设计事实上依赖 batch 图。下面是同一组模型分别用整批 / 逐对两种方式打分：\n')
L.append('| 数据集 | 协议 | 验证选中轮(均值) | 验证选轮·整批打分 AUC / AUPR | **验证选轮·逐对打分 AUC / AUPR** | 测试最优轮·整批 | 测试最优轮·逐对 | 末轮·整批 | 末轮·逐对 | PLMF-MDA 复现 |')
L.append('|---|---|---|---|---|---|---|---|---|---|')
for ds in ('MDR', 'MDS'):
    for sp in ('5cv', 'single'):
        h, p = honest(ds, sp), plmf[(ds, sp)]
        if h:
            L.append(f"| {ds} | {sp} | {h['valsel_epoch']:.1f} | {fmt(h['batch_valsel_auc'])} / {fmt(h['batch_valsel_aupr'])} | **{fmt(h['iso_valsel_auc'])} / {fmt(h['iso_valsel_aupr'])}** | {fmt(h['batch_best_auc'])} | {fmt(h['iso_best_auc'])} | {fmt(h['batch_last_auc'])} | {fmt(h['iso_last_auc'])} | {fmt(p['auc'])} / {fmt(p['aupr'])} |")
        else:
            L.append(f"| {ds} | {sp} | (运行中) | | | | | | | {fmt(p['auc'])} / {fmt(p['aupr'])} |")
L.append('')
L.append('说明：这组运行的训练对比主运行少 10%（切给了验证），所以"测试最优轮·整批"与 §3 主运行的数字略有差异。PLMF-MDA 与 4 个基线都是逐对打分的模型，因此**与它们同口径可比的是"验证选轮·逐对打分"这一列**。\n')
# ---------------- 5. ablation ----------------
L.append('## 5. 基因图 / 元路径到底有没有用（消融，测试最优轮口径）\n')
L.append('| 数据集 | 协议 | kmer + 真实基因图 | kmer + 基因 id 置换 | kmer + 无基因边 | CCLE 表达 + 真实基因图 | kmer+表达 + 真实基因图 | 全局分支改用固定训练图 |')
L.append('|---|---|---|---|---|---|---|---|')
for ds in ('MDR', 'MDS'):
    for sp in ('5cv', 'single'):
        cells = []
        for feat, gene in (('kmer', 'real'), ('kmer', 'shuffled'), ('kmer', 'none'), ('expr', 'real'), ('kmer_expr', 'real'), ('kmer_gg-train', 'real')):
            r = S.ours(ds, sp, feat, gene)
            cells.append(f"{fmt(r['auc'])} / {fmt(r['aupr'])}" + ('' if (r and r['complete']) else ' ⚠') if r else '(运行中)')
        L.append(f"| {ds} | {sp} | " + ' | '.join(cells) + ' |')
L.append('')

L.append('## 6. 产物与复现\n')
L.append('```')
L.append('mphgnn_plmf_20260921/')
L.append('├─ fetch_encori.py          # ENCORI miRNA-target 抓取 → cache/encori_targets.tsv.gz')
L.append('├─ build_drug_targets.py    # ChEMBL 机制靶点 → cache/drug_targets.json')
L.append('├─ build_gene_ids.py        # HumanNet Entrez↔symbol → cache/entrez_symbol.json')
L.append('├─ build_graph.py           # 异质图 + 节点特征 → cache/graph_{MDR,MDS}.npz')
L.append('├─ plmf_splits.py           # 与 plmf_baselines 逐位相同的划分')
L.append('├─ mphgnn_model.py / mphgnn_layer.py   # 官方模型（605 偏移参数化）')
L.append('├─ run_mphgnn_plmf.py       # 主 runner；run_mphgnn_plmf_val.py 为 10% 验证早停版')
L.append('├─ run_all.sh / run_val_after.sh / summarize.py / build_excel.py / write_report.py')
L.append('├─ MPHGNN_on_PLMF-MDA.xlsx  # 说明 / 4 个榜单 / 逐次运行')
L.append('├─ results/*.csv, *_curves.json  # 逐折/逐 seed 指标与每轮曲线')
L.append('└─ logs/')
L.append('```')
L.append('外部数据：`/data1/licheng/incrd-hg/data_external/humannet_20260921/`（HumanNet-FN，含 SOURCE.txt 与 sha256）。ChemBERTa 权重经 hf-mirror.com 下载到 `.external/models/hf_home`。\n')
L.append('```bash')
L.append('PY=/data1/licheng/miniconda3/envs/ncRNAdrug/bin/python; V=/data1/licheng/incrd-hg/.venv-dgl/bin/python')
L.append('$PY fetch_encori.py && $PY build_drug_targets.py && $PY build_gene_ids.py && $V build_drug_chemberta.py && $PY build_graph.py')
L.append('./run_all.sh && ./run_val_after.sh && $V summarize.py && $V build_excel.py && $V write_report.py')
L.append('```')
open(os.path.join(HERE, 'REPORT_MPHGNN_ON_PLMF.md'), 'w').write('\n'.join(L) + '\n')
print('wrote REPORT_MPHGNN_ON_PLMF.md')
