"""Generic gate evaluator over cached member logits.

usage: evaluate_generic_v1.py <cells_dir> <file_pattern> <members(comma)> <labels(comma)> <weights(comma)> <out_prefix>
file_pattern uses {task} and {surface}; each npz holds logit_<label>_<member> arrays and pairs_val.
Gates: MiDrug inner mean dAUC/dAP > 0 with >=2/3 seeds positive, resplit/extra dAUC/dAP > 0 in every split;
       LncDrug dAUC/dAP >= 0 on every seed/split (strict) and >= -0.001 (tolerance).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG')
sys.path.insert(0, str(Path(__file__).resolve().parent))
from improve import tabular_foundation as base
from provenance_channel_v1 import surfaces, TASKS, logit

GROUP = {'inner': 'inner', 's42': 'resplit', 's2026': 'resplit', 's31415': 'resplit', 'x7': 'extra', 'x77': 'extra', 'x777': 'extra'}


def evaluate(cells, pattern, members, labels, weights):
    rows = []
    for task in TASKS:
        for s in surfaces(task):
            path = Path(cells) / pattern.format(task=task, surface=s['name'])
            if not path.exists():
                continue
            z = np.load(path, allow_pickle=False)
            for seed, refpath, key, ykey, pkey in s['refs']:
                r = np.load(refpath, allow_pickle=False)
                ref, y, pairs = r[key], r[ykey], r[pkey]
                assert np.array_equal(pairs, z['pairs_val'])
                m_ref = base.metrics(y, ref, pairs)
                for label in labels:
                    comp = np.mean([z['logit_%s_%s' % (label, m)] for m in members], axis=0)
                    for w in weights:
                        v = expit((1 - w) * logit(ref) + w * comp)
                        m = base.metrics(y, v, pairs)
                        rows.append(dict(task=task, surface=s['name'], group=GROUP[s['name']], key='seed%d' % seed if s['name'] == 'inner' else s['name'],
                                         recipe=label, w=w, auc=m['auc'], ap=m['ap'], d_auc=m['auc'] - m_ref['auc'], d_ap=m['ap'] - m_ref['ap'],
                                         d_macro=m['macro_drug_ap'] - m_ref['macro_drug_ap']))
    return pd.DataFrame(rows)


def gates(t):
    out = []
    for (recipe, w), sub in t.groupby(['recipe', 'w']):
        mi, ln = sub[sub.task == 'MiDrug'], sub[sub.task == 'LncDrug']
        g = dict(recipe=recipe, w=w)
        inner = mi[mi.group == 'inner']
        g['mi_inner_ok'] = bool(inner.d_auc.mean() > 0 and inner.d_ap.mean() > 0 and (inner.d_auc > 0).sum() >= 2 and (inner.d_ap > 0).sum() >= 2)
        rest = mi[mi.group != 'inner']
        g['mi_split_ok'] = bool((rest.d_auc > 0).all() and (rest.d_ap > 0).all())
        g['ln_strict_ok'] = bool((ln.d_auc >= 0).all() and (ln.d_ap >= 0).all())
        g['ln_tol_ok'] = bool((ln.d_auc >= -0.001).all() and (ln.d_ap >= -0.001).all())
        for grp in ('inner', 'resplit', 'extra'):
            for task, tt in (('Mi', mi), ('Ln', ln)):
                x = tt[tt.group == grp]
                g['%s_%s_dAUC' % (task, grp)] = x.d_auc.mean(); g['%s_%s_dAP' % (task, grp)] = x.d_ap.mean()
        g['Ln_min_dAUC'], g['Ln_min_dAP'] = ln.d_auc.min(), ln.d_ap.min()
        g['Mi_min_dAUC'], g['Mi_min_dAP'] = mi.d_auc.min(), mi.d_ap.min()
        g['Mi_dmacro'] = mi.d_macro.mean()
        g['strict_joint'] = g['mi_inner_ok'] and g['mi_split_ok'] and g['ln_strict_ok']
        g['tol_joint'] = g['mi_inner_ok'] and g['mi_split_ok'] and g['ln_tol_ok']
        out.append(g)
    return pd.DataFrame(out)


def contrast(t, a, b):
    """per-surface mean(real) - mean(control) on MiDrug, by weight."""
    res = []
    for w, sub in t[t.task == 'MiDrug'].groupby('w'):
        x = sub[sub.recipe == a].groupby('surface')[['d_auc', 'd_ap']].mean() - sub[sub.recipe == b].groupby('surface')[['d_auc', 'd_ap']].mean()
        res.append(dict(w=w, real=a, control=b, n=len(x), auc_pos=int((x.d_auc > 0).sum()), ap_pos=int((x.d_ap > 0).sum()), mean_dauc=x.d_auc.mean(), mean_dap=x.d_ap.mean()))
    return pd.DataFrame(res)


if __name__ == '__main__':
    cells, pattern, members, labels, weights, prefix = sys.argv[1:7]
    members, labels = members.split(','), labels.split(',')
    weights = [float(x) for x in weights.split(',')]
    t = evaluate(cells, pattern, members, labels, weights)
    t.to_csv(prefix + '_metrics.csv', index=False)
    g = gates(t)
    g.to_csv(prefix + '_gates.csv', index=False)
    pd.set_option('display.width', 300)
    cols = ['recipe', 'w', 'strict_joint', 'tol_joint', 'Mi_inner_dAUC', 'Mi_inner_dAP', 'Mi_resplit_dAUC', 'Mi_resplit_dAP', 'Mi_extra_dAUC', 'Mi_extra_dAP', 'Mi_min_dAUC', 'Mi_min_dAP', 'Ln_inner_dAUC', 'Ln_inner_dAP', 'Ln_resplit_dAUC', 'Ln_resplit_dAP', 'Ln_extra_dAUC', 'Ln_extra_dAP', 'Ln_min_dAUC', 'Ln_min_dAP', 'Mi_dmacro']
    print(g[cols].to_string(index=False, float_format=lambda v: '%+.5f' % v))
    if len(sys.argv) > 7:
        for pair in sys.argv[7].split(';'):
            a, b = pair.split('>')
            print('\nCONTRAST', a, '-', b)
            print(contrast(t, a, b).to_string(index=False, float_format=lambda v: '%+.5f' % v))
