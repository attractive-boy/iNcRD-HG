# -*- coding: utf-8 -*-
"""Paired comparison (identical PLMF-MDA splits, identical 10% hold-out, valsel) between this round's
no-LQO modules and the LQO-based GNN of round 6 (../plmf_gnnarch_20260923: lqo = M1 LQO, pair = LQO + RT-NCN,
both internal-only) and round 7's best internal LQO stack (../plmf_internal_20260923, seed 42 only)."""
import os
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); B = os.path.dirname(HERE)
SRC = {'r6': os.path.join(B, 'plmf_gnnarch_20260923', 'results'), 'r7': os.path.join(B, 'plmf_internal_20260923', 'results'), 'r8': os.path.join(HERE, 'results')}
PROT = {'5cv×3': [('5cv', s) for s in (42, 43, 44)], '80/20×5': [('single', s) for s in range(5)], '5cv(42)': [('5cv', 42)]}


def load(src, tag, ds, prot):
    out = []
    for sp, sd in PROT[prot]:
        f = os.path.join(SRC[src], f'{tag}_{ds}_{sp}_seed{sd}.csv')
        if os.path.exists(f):
            d = pd.read_csv(f); d['key'] = f'{sp}_{sd}_' + d['fold'].astype(str); out.append(d)
    return pd.concat(out) if out else None


rows = [('r6', 'lqo', 'r8', 'F_qmp', 'M1 QMP 对 LQO（同为仅编码器级修正）'),
        ('r6', 'pair', 'r8', 'F_sp', 'QMP+SP-RNCN 对 LQO+RT-NCN'),
        ('r6', 'pair', 'r8', 'F_full', 'full（无 LQO）对 LQO+RT-NCN'),
        ('r7', 'c8_nbf', 'r8', 'F_full', 'full（无 LQO）对第七轮内部最强 LQO 堆叠 c8'),
        ('r7', 'r_all_l30', 'r8', 'F_full', 'full（无 LQO）对第七轮 LQO+可学习 ridge')]
print('| 对比 | 协议 | MDR 参照 → 本轮 | Δ AUC / AUPR（胜折） | MDS 参照 → 本轮 | Δ AUC / AUPR（胜折） |')
print('|---|---|---|---|---|---|')
for s0, t0, s1, t1, name in rows:
    for prot in ('5cv×3', '80/20×5', '5cv(42)'):
        line = [name, prot]; ok = False
        for ds in ('MDR', 'MDS'):
            a0, a1 = load(s0, t0, ds, prot), load(s1, t1, ds, prot)
            if a0 is None or a1 is None:
                line += ['n/a', 'n/a']; continue
            j = a1.merge(a0, on='key', suffixes=('', '_r'))
            if not len(j):
                line += ['n/a', 'n/a']; continue
            ok = True
            da = j.valsel_auc - j.valsel_auc_r; dp = j.valsel_aupr - j.valsel_aupr_r
            line += [f'{j.valsel_auc_r.mean():.4f}/{j.valsel_aupr_r.mean():.4f} → {j.valsel_auc.mean():.4f}/{j.valsel_aupr.mean():.4f}',
                     f'{da.mean():+.4f} / {dp.mean():+.4f} ({int((da > 0).sum())}/{len(j)} · {int((dp > 0).sum())}/{len(j)})']
        if ok:
            print('| ' + ' | '.join(line) + ' |')
