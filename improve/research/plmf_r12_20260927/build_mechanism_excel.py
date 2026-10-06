# -*- coding: utf-8 -*-
"""Build PLMF-MDA_提升机制表.xlsx -- ablation ladders in the requested format:

    模块 / 变体 | MDR AUC | MDR AUPR | ΔAUC vs 参考 | ΔAUPR vs 参考 | MDS AUC | MDS AUPR | ΔAUC vs 参考 | ΔAUPR vs 参考

Δ is the paired (same seed+fold) difference vs the previous row unless the sheet name says otherwise.
Sources: R12 (this dir), R8 plmf_nolqo_20260923, R9 plmf_enc_20260924, R6 plmf_gnnarch_20260923 results/*.csv.
R9 ladder tags were identified by matching the published REPORT_ENC.md numbers exactly.
"""
import os
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
OUT = os.path.join(HERE, 'PLMF-MDA_提升机制表.xlsx')
COLS = ['模块 / 变体', 'MDR AUC', 'MDR AUPR', 'ΔAUC vs 参考', 'ΔAUPR vs 参考', 'MDS AUC', 'MDS AUPR', 'ΔAUC vs 参考', 'ΔAUPR vs 参考']


def load(root, tag, ds, split, seeds, dirname='results'):
    rows = []
    for sd in seeds:
        f = os.path.join(root, dirname, f'{tag}_{ds}_{split}_seed{sd}.csv')
        if os.path.exists(f):
            x = pd.read_csv(f); x['seed'] = sd; rows.append(x)
    return pd.concat(rows) if rows else None


def fmt(d):
    return f'{d:+.4f}'


def ladder(entries, root, split, seeds, metric, ref='prev'):
    """entries: list of (label, tag). metric: 'best' or 'valsel'. ref: 'prev' or 'first'.
    Rows are lists (COLS contains duplicate delta labels, so dicts would collapse them)."""
    out = []
    prev = None
    for label, tag in entries:
        cur = load(root, tag, 'MDR', split, seeds)
        if cur is None:
            out.append([label] + [''] * 8); prev = tag; continue
        row = [label]
        for ds in ('MDR', 'MDS'):
            x = load(root, tag, ds, split, seeds)
            row += [round(x[f'{metric}_auc'].mean(), 4), round(x[f'{metric}_aupr'].mean(), 4)]
            ref_tag = {'prev': prev, 'first': entries[0][1]}[ref]
            if ref_tag is None or ref_tag == tag:
                row += ['—', '—']
            else:
                b = load(root, ref_tag, ds, split, seeds)
                m = x.merge(b, on=['seed', 'fold'], suffixes=('', '_b'))
                row += [fmt(m[f'{metric}_auc'].mean() - m[f'{metric}_auc_b'].mean()),
                        fmt(m[f'{metric}_aupr'].mean() - m[f'{metric}_aupr_b'].mean())]
        out.append(row)
        prev = tag
    return pd.DataFrame(out, columns=COLS)


def main():
    R12, R8, R9, R6 = HERE, os.path.join(RESEARCH, 'plmf_nolqo_20260923'), os.path.join(RESEARCH, 'plmf_enc_20260924'), os.path.join(RESEARCH, 'plmf_gnnarch_20260923')

    ladders = {}
    r12_entries = [('B（R11 基线复现，90% 消息图）', 'B'),
                   ('+ VG 消息图=全部训练正例（--val_graph）', 'VG'),
                   ('+ HN 难负例+BPR（--hardneg 2 --hardneg_pool 8 both bpr）', 'VGHN'),
                   ('+ UNL 未标注单元（--unl 1）', 'VGHNUNL'),
                   ('+ PPRz 空心扩散（--ppr --ppr_norm）', 'VGHNUNLPPR')]

    ladders['R12阶梯_5cv3_best'] = ladder(r12_entries, R12, '5cv', [42, 43, 44], 'best')
    ladders['R12阶梯_5cv3_valsel'] = ladder(r12_entries, R12, '5cv', [42, 43, 44], 'valsel')
    ladders['R12阶梯_80-20_best'] = ladder(r12_entries, R12, 'single', [0, 1, 2], 'best')
    ladders['R12对B累计_best'] = ladder(r12_entries, R12, '5cv', [42, 43, 44], 'best', ref='first')

    r8_entries = [('L0 plain iNcRD-HG', 'F_plain'),
                  ('L1 + M1 QMP 查询掩蔽消息传递', 'F_qmp'),
                  ('L2 + M2 SP-RNCN 简单路径关系型共同邻居', 'F_sp'),
                  ('L3 + M3 MC-EASE 掩蔽单元闭式协同回归', 'F_ease'),
                  ('L4 + M4 SeqSide 序列来源旁信息（= full）', 'F_full')]
    ladders['R8空心阶梯_5cv3_valsel'] = ladder(r8_entries, R8, '5cv', [42, 43, 44], 'valsel')

    r9_entries = [('L0 plain iNcRD-HG 编码器', 'P_plain'),
                  ('L1 + M1 查询掩蔽消息传递', 'P_mask'),
                  ('L2 + E1 CKA 内容键控消息权重', 'P_cka'),
                  ('L3 + E2 H-NBF 空心源条件消息传递（药物源）', 'P_cka_nbf30d128'),
                  ('L4 + E3 反向条件（miRNA 源）', 'P_E3_bi32'),
                  ('L5 + E4 条件状态逐层读出', 'P_L5')]
    ladders['R9编码器阶梯_5cv3_valsel'] = ladder(r9_entries, R9, '5cv', [42, 43, 44], 'valsel')

    r6_entries = [('plain iNcRD-HG（GeneralConv×2 + MLP）', 'plain'),
                  ('+LQO', 'lqo'),
                  ('+ RT-NCN 关系型神经共同邻居', 'pair'),
                  ('+ 溯源超边配对通道', 'pair_prov'),
                  ('+ 旁信息关系配对通道（= full）', 'full')]
    ladders['R6_LQO阶梯_5cv42_valsel'] = ladder(r6_entries, R6, '5cv', [42], 'valsel')

    off_entries = [('官方协议 B（100% 训练对，无留出）', 'OFF_B'),
                   ('+ HN+UNL', 'OFF_FIN'),
                   ('+ PPRz（= F）', 'OFF_FINPPR')]
    ladders['官方协议_5cv42_best'] = ladder(off_entries, R12, '5cv', [42], 'best')
    ladders['官方协议_80-20_best'] = ladder(off_entries, R12, 'single', [0, 1, 2], 'best')

    mech = pd.DataFrame([
        ('输入特征', 'k-mer + ChemBERTa + kNN(10)（plain 基线）', '默认', 'plain 0.9242/0.9161', 'plain 0.9288/0.9252', '已内置', 'R8 阶梯 L0（5cv×3 valsel）'),
        ('空心编码', 'M1 QMP 查询掩蔽消息传递', '--enc mask --exact', '+0.0072/+0.0063', '+0.0075/+0.0060', '已叠加', 'R8 报告 §0'),
        ('配对通道', 'M2 SP-RNCN 简单路径+神经共同邻居', '--qis --qis_ext --qncn --ncn_rel co,knn,dknn,seed7,seed6,pre,fam', '+0.0117/+0.0132', '+0.0113/+0.0117', '已叠加', 'R8 报告 §0'),
        ('配对通道', 'M3 MC-EASE 掩蔽单元闭式协同回归', '--ease 10,50 --ease_side kmer', '+0.0053/+0.0081', '+0.0028/+0.0046', '已叠加', 'R8 报告 §0'),
        ('配对通道', 'M4 SeqSide 种子/前体/家族关系', '--rel seed7,seed6,pre,fam', '+0.0017/+0.0021', '+0.0021/+0.0024', '已叠加', 'R8 报告 §0'),
        ('协议修复', 'VG 消息图=全部训练正例（修复 valsel 少 10% 边）', '--val_graph', '+0.0037/+0.0044', '+0.0028/+0.0029', '已叠加（R12）', 'R12 报告 §0（官方协议下增量=0）'),
        ('训练分布', 'HN 同药∪度匹配难负例+BPR+重采样', '--hardneg 2 --hardneg_pool 8 --hardneg_mode both --hardneg_loss bpr', '+0.0017/+0.0026', '+0.0011/+0.0020', '已叠加', 'R12（对 VG，15 折）'),
        ('训练分布', 'UNL 未标注单元 BCE', '--unl 1 --unl_w 1', '+0.0000/−0.0001', '+0.0002/+0.0003', '已叠加', 'R12（对 VG+HN）'),
        ('扩散通道', 'PPRz 精确空心 PPR/Katz+掩蔽行 z', '--ppr --ppr_norm', '+0.0001/−0.0000', '+0.0003/+0.0004', '已叠加', 'R12（对 VG+HN+UNL）'),
        ('编码器替代', 'LQO 留出查询边消息传递', '--lqo 5（R6）', '+0.0120/+0.0109', '+0.0074/+0.0051', '被 M1 取代', 'R6 阶梯（5cv42 valsel）'),
        ('编码器替代', 'E1 CKA 内容键控权重', '--cka', '+0.0023/+0.0018', '+0.0013/+0.0011', '独立正、叠加被吸收', 'R9 编码器阶梯（5cv×3）'),
        ('编码器替代', 'E2 H-NBF 空心源条件消息传递', '--nbf --nbf_lr 30 --nbf_dim 128', '+0.0062/+0.0066', '+0.0053/+0.0052', '独立正、叠加被吸收', 'R9 编码器阶梯（5cv×3）'),
        ('编码器替代', 'E3 反向条件（miRNA 源）', '--nbf_src both --nbf_mdim 32', '+0.0017/+0.0013', '+0.0019/+0.0022', '独立正、叠加被吸收', 'R9 编码器阶梯（5cv×3）'),
        ('编码器替代', 'E4 条件状态逐层读出', '--nbf_jk', '+0.0008/+0.0015', '+0.0003/+0.0005', '独立弱正、叠加被吸收', 'R9 编码器阶梯（5cv×3）'),
        ('训练分布', 'listwise 行内 softmax 排序损失', '--lw_d 0.1 --lw_m 0.1', '对 UNL +0.0005/+0.0007', '对 UNL +0.0002/+0.0005', 'VG 基线上负→不用', 'R10/R12'),
        ('特征栈', 'ctx(33)+diff(52) × 逻辑回归（交叉拟合）', 'plmf_innov bank.py', '0.9510/0.9455（5cv42）', '0.9540/0.9516（5cv42）', '未验证集成', 'R3 报告 §3'),
        ('冷启动面', 'CPGF / 内容 kNN×LR 堆叠', 'plmf_xds', '冷药 0.83 vs PLMF 0.65', '—', '不同协议', 'R5 报告'),
        ('无效（勿用）', 'PSA/MoE/AA/多尺度kNN/同-miRNA HN/TPE/ID嵌入/药物EASE核/流行度分支/xfit/gat/纯内容编码器/co-attention/scf/align/cdiff/wide/ease_learn', '—', '≤ +0.0005 或为负', '≤ +0.0005 或为负', '不采用', 'R3/R8/R9/R10/R11/R12 筛选'),
    ], columns=['大类', '机制 / 变体', '开关', '增益（MDR）', '增益（MDS）', '状态', '出处 / 口径'])

    notes = pd.DataFrame([
        ('Δ 参考', '各"阶梯"sheet：Δ = 与上一行同 seed+同 fold 的配对差（与 R6/R8/R9 报告口径一致）；第一行无参考记 "—"。'),
        ('R12对B累计_best', '该 sheet 的 Δ 一律对第一行 B（总增益）。'),
        ('R12 阶梯', 'B→VG→HN→UNL→PPRz，MDR/MDS，5cv×3 seeds(42/43/44)=15 折 / 80-20×3 seeds(0/1/2)；best=测试最优 epoch，valsel=10% 训练对留出选 epoch。'),
        ('官方协议', '无留出、100% 训练对、测试最优 epoch（与 PLMF-MDA 基线表同口径）；5cv 仅 seed42（与 R8 §6.2 同），80-20 为 seeds 0/1/2。'),
        ('R8 阶梯', 'plmf_nolqo_20260923；5cv×3 valsel；M2 行的 ncn_rel=co,knn,dknn（M4 关系在下一级加入）。'),
        ('R9 阶梯', 'plmf_enc_20260924；仅编码器阶梯（不含 M2–M4）；tag 由复现数字精确匹配 REPORT_ENC.md。'),
        ('R6 阶梯', 'plmf_gnnarch_20260923；5cv seed42 valsel；含外部溯源数据（当前约束下不可用），此处仅作格式与历史对照。'),
        ('最终模型', 'F = B + VG + HN + UNL + PPRz：5cv×3 0.9558/0.9534（MDR）、0.9574/0.9560（MDS）best；官方协议 5cv42 0.9551/0.9530、0.9582/0.9577。'),
    ], columns=['项目', '说明'])

    with pd.ExcelWriter(OUT) as xw:
        for name, df in ladders.items():
            df.to_excel(xw, name, index=False)
            ws = xw.sheets[name]
            ws.column_dimensions['A'].width = 46
            for col in 'BCDEFGHI':
                ws.column_dimensions[col].width = 13
        mech.to_excel(xw, '机制总表', index=False)
        for col, w in zip('ABCDEFG', (30, 46, 40, 24, 24, 20, 34)):
            xw.sheets['机制总表'].column_dimensions[col].width = w
        notes.to_excel(xw, '说明', index=False)
        xw.sheets['说明'].column_dimensions['A'].width = 22
        xw.sheets['说明'].column_dimensions['B'].width = 110
    print('wrote', OUT)
    for name, df in ladders.items():
        print('\n==', name)
        print(df.to_string(index=False))


if __name__ == '__main__':
    main()
