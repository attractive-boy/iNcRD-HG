# -*- coding: utf-8 -*-
"""Excel summary of round 8 (no-LQO, query-hollow modules): per-config means over folds x seeds for both
metrics (valsel / test-best / last) and the full per-fold table."""
import os, glob
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
TAGS = [('F_plain', 'L0 plain'), ('F_qmp', 'L1 +M1 QMP'), ('F_sp', 'L2 +M2 SP-RNCN'), ('F_ease', 'L3 +M3 MC-EASE'), ('F_full', 'L4 +M4 SeqSide = full'),
        ('F_noM1', 'full - M1'), ('F_noM2', 'full - M2'), ('F_noM3', 'full - M3'), ('F_nomask', 'control: no query masking'),
        ('F_seqS', 'control: M4 shuffled'), ('F_seqonly', 'M4 sequence-only (no name relations)'), ('F_sp_p', 'strict: plain enc + M2'),
        ('F_ease_p', 'strict: plain enc + M2 + M3'), ('N_plain', 'plain, 100% train (test-best protocol)'), ('N_full', 'full, 100% train (test-best protocol)')]
rows, folds = [], []
for tag, name in TAGS:
    for f in sorted(glob.glob(os.path.join(HERE, 'results', f'{tag}_MDR_*_seed*.csv')) + glob.glob(os.path.join(HERE, 'results', f'{tag}_MDS_*_seed*.csv'))):
        if f.endswith(('_scores.csv',)):
            continue
        d = pd.read_csv(f); d['config'] = name; folds.append(d)
F = pd.concat(folds)
F['protocol'] = F['split'].map({'5cv': '5cv x3 seeds', 'single': '80/20 x5 seeds'})
agg = {c: 'mean' for c in ['valsel_auc', 'valsel_aupr', 'best_auc', 'best_aupr', 'last_auc', 'last_aupr'] if c in F}
S = F.groupby(['config', 'dataset', 'protocol'], sort=False).agg(dict(agg, fold='count')).rename(columns={'fold': 'n_folds'}).reset_index()
with pd.ExcelWriter(os.path.join(HERE, 'iNcRD-HG_noLQO_PLMF-MDA.xlsx')) as w:
    S.round(4).to_excel(w, sheet_name='summary', index=False)
    F.round(4).to_excel(w, sheet_name='per_fold', index=False)
print(S.round(4).to_string())
