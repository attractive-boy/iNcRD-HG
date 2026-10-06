import glob, pandas as pd
D = pd.concat([pd.read_csv(f) for f in glob.glob('results/*.csv') if not f.endswith('_scores.csv')], ignore_index=True)
cols = [c for c in ['valsel_auc', 'valsel_aupr', 'best_auc', 'best_aupr', 'last_auc', 'last_aupr'] if c in D]
S = D.groupby(['tag', 'dataset', 'split'])[cols].mean().round(4).reset_index()
S['n_folds'] = D.groupby(['tag', 'dataset', 'split']).size().values
with pd.ExcelWriter('iNcRD-HG_R10_PLMF-MDA.xlsx') as w:
    S.to_excel(w, 'summary', index=False); D.to_excel(w, 'per_fold', index=False)
print(S[S.tag.str.startswith(('G_', 'N_'))].to_string())
