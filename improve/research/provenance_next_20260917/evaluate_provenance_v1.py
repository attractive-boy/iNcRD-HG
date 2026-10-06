"""Gate evaluation for the provenance channels (reads cached member logits only).

For every recipe label L and weight w:
    score = sigmoid((1-w) * logit(reference) + w * mean_m logit(member_m^L))
Gates (accepted shared-anchor rules):
    inner   : MiDrug mean dAUC/dAP > 0 with >= 2/3 seeds positive; LncDrug dAUC/dAP >= 0 every seed
    resplit : MiDrug dAUC/dAP > 0 in every split; LncDrug dAUC/dAP >= 0 in every split   (42/2026/31415)
    extra   : same rule on the never-before-used splits 7/77/777
Information gate: the real channel must beat its permutation control on MiDrug on every surface
(mean over seeds/splits of dAUC and dAP, real - control > 0).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG')
from improve import tabular_foundation as base
from provenance_channel_v1 import surfaces, TASKS, SEEDS, SUBSET, logit, variant_table

HERE = Path('/data1/licheng/incrd-hg/iNcRD-HG/improve/research/provenance_next_20260917')
CELLS = HERE / 'cells'
WEIGHTS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4)
GROUP = {'inner': 'inner', 's42': 'resplit', 's2026': 'resplit', 's31415': 'resplit', 'x7': 'extra', 'x77': 'extra', 'x777': 'extra'}


def rows_for(task, bank):
    rows = []
    for s in surfaces(task):
        path = CELLS / ('%s_%s_%s.npz' % (task, s['name'], bank))
        if not path.exists():
            continue
        z = np.load(path, allow_pickle=False)
        for seed, refpath, key, ykey, pkey in s['refs']:
            r = np.load(refpath, allow_pickle=False)
            ref, y, pairs = r[key], r[ykey], r[pkey]
            assert np.array_equal(pairs, z['pairs_val'])
            m_ref = base.metrics(y, ref, pairs)
            for label in variant_table():
                comp = np.mean([z['logit_%s_%s' % (label, m)] for m in SUBSET], axis=0)
                for w in WEIGHTS:
                    v = expit((1 - w) * logit(ref) + w * comp)
                    m = base.metrics(y, v, pairs)
                    rows.append(dict(task=task, bank=bank, surface=s['name'], group=GROUP[s['name']], key='seed%d' % seed if s['name'] == 'inner' else s['name'],
                                     recipe=label, w=w, auc=m['auc'], ap=m['ap'], d_auc=m['auc'] - m_ref['auc'], d_ap=m['ap'] - m_ref['ap'],
                                     macro=m['macro_drug_ap'], d_macro=m['macro_drug_ap'] - m_ref['macro_drug_ap']))
    return rows


def gates(t):
    out = []
    for (bank, recipe, w), sub in t.groupby(['bank', 'recipe', 'w']):
        g = dict(bank=bank, recipe=recipe, w=w)
        ok_all = True
        for task in TASKS:
            for grp in ('inner', 'resplit', 'extra'):
                x = sub[(sub.task == task) & (sub.group == grp)]
                if len(x) == 0:
                    ok = None
                elif task == 'LncDrug':
                    ok = bool((x.d_auc >= 0).all() and (x.d_ap >= 0).all())
                elif grp == 'inner':
                    ok = bool(x.d_auc.mean() > 0 and x.d_ap.mean() > 0 and (x.d_auc > 0).sum() >= 2 and (x.d_ap > 0).sum() >= 2)
                else:
                    ok = bool((x.d_auc > 0).all() and (x.d_ap > 0).all())
                g['%s_%s_ok' % (task[:2], grp)] = ok
                g['%s_%s_dAUC' % (task[:2], grp)] = float(x.d_auc.mean()) if len(x) else np.nan
                g['%s_%s_dAP' % (task[:2], grp)] = float(x.d_ap.mean()) if len(x) else np.nan
                if ok is False:
                    ok_all = False
        g['joint'] = ok_all
        out.append(g)
    return pd.DataFrame(out)


def control_gate(t, real, ctrl):
    """real - control on MiDrug, per surface group; both must be > 0 for every group."""
    res = {}
    for (bank, w), sub in t[t.task == 'MiDrug'].groupby(['bank', 'w']):
        a = sub[sub.recipe == real].groupby('surface')[['d_auc', 'd_ap']].mean()
        b = sub[sub.recipe == ctrl].groupby('surface')[['d_auc', 'd_ap']].mean()
        d = (a - b)
        res[(bank, w)] = dict(n_surfaces=len(d), auc_pos=int((d.d_auc > 0).sum()), ap_pos=int((d.d_ap > 0).sum()),
                              mean_dauc=float(d.d_auc.mean()), mean_dap=float(d.d_ap.mean()))
    return res


if __name__ == '__main__':
    rows = []
    for task in TASKS:
        for bank in ('context76', 'tiered62'):
            rows += rows_for(task, bank)
    t = pd.DataFrame(rows)
    t.to_csv(HERE / 'metrics_v1.csv', index=False)
    g = gates(t)
    g.to_csv(HERE / 'gates_v1.csv', index=False)
    pd.set_option('display.width', 250)
    cols = ['bank', 'recipe', 'w', 'joint', 'Mi_inner_dAUC', 'Mi_inner_dAP', 'Mi_resplit_dAUC', 'Mi_resplit_dAP', 'Mi_extra_dAUC', 'Mi_extra_dAP', 'Ln_inner_dAUC', 'Ln_inner_dAP', 'Ln_resplit_dAUC', 'Ln_resplit_dAP', 'Ln_extra_dAUC', 'Ln_extra_dAP']
    print(g[cols].sort_values(['joint', 'Mi_extra_dAP'], ascending=False).to_string(index=False, float_format=lambda v: '%+.5f' % v))
    for real, ctrl in (('bank+pmid', 'bank+pmid_perm'), ('bank+ctx', 'bank+ctx_perm'), ('bank+prov', 'bank+prov_perm'), ('bank+pmid', 'bank'), ('bank+ctx', 'bank'), ('bank+prov', 'bank')):
        print('\nCONTROL', real, 'vs', ctrl)
        for k, v in control_gate(t, real, ctrl).items():
            print('  ', k, v)
