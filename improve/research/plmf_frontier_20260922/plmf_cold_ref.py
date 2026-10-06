# -*- coding: utf-8 -*-
"""Official PLMF-MDA model (cached frozen encoders) on the SAME cold-drug splits as cold_probe.py."""
import argparse, os, sys, time, csv
import numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); BASE = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(BASE, 'plmf_baselines_20260920')); sys.path.insert(0, HERE)
import plmf_ref_5cv as P  # noqa
from cold_probe import load_all, cold_split  # noqa
from sklearn.model_selection import StratifiedKFold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True); ap.add_argument('--protocol', default='plmf', choices=['plmf', 'ldo'])
    ap.add_argument('--seeds', default='0,1,2,3,4'); ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--lr', type=float, default=5e-4); ap.add_argument('--weight_decay', type=float, default=5e-4)
    ap.add_argument('--batch', type=int, default=32); ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    args.no_cuda = False; args.workers = 0; args.mirna_feature_dim = 128; args.drug_feature_dim = 128
    args.drug_input_dim = 69; args.drug_hidden_dim = 128; args.split = f'cold_{args.protocol}'
    P.RNAFMFeatureExtractor.forward = P._cache_forward_rna
    P.ChemBERTaFeatureExtractor.forward = P._cache_forward_smi
    suf = 'R' if args.dataset == 'MDR' else 'S'
    drug_features, _ = P.load_drug_data(os.path.join(P.PLMF_DATA_DIR, args.dataset, f'drug_ID_{suf}.xlsx'))
    mirna_features, _ = P.load_mirna_data(os.path.join(P.PLMF_DATA_DIR, args.dataset, f'miRNA_ID_{suf}.xlsx'))
    P.build_caches(drug_features, mirna_features, args.device)
    D = load_all(args.dataset)
    A = np.zeros((D['nr'], D['nd']), np.int8); A[D['pos'][:, 0], D['pos'][:, 1]] = 1; ddeg = A.sum(0)
    os.makedirs(os.path.join(HERE, 'results'), exist_ok=True); os.makedirs(os.path.join(HERE, 'logs'), exist_ok=True)
    rows = []
    with open(os.path.join(HERE, 'logs', f'plmf_cold_{args.dataset}_{args.protocol}.log'), 'a') as log:
        for seed in [int(s) for s in args.seeds.split(',')]:
            if args.protocol == 'plmf':
                folds = [np.where((ddeg >= 11) & (ddeg <= 20))[0]]
            else:
                bins = np.digitize(ddeg, [2, 6, 21, 101])
                folds = [te for _, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(D['nd']), bins)]
            for fi, test_drugs in enumerate(folds):
                tr, ytr, te, yte = cold_split(D, test_drugs, seed * 10 + fi)
                args.seed = seed; args.fold = fi; t0 = time.time()
                best, last = P.run_fold(args, tr, ytr, te, yte, drug_features, mirna_features, log)
                rows.append(dict(dataset=args.dataset, protocol=args.protocol, seed=seed, fold=fi, n_test=len(yte),
                                 best_auc=best['AUC'], best_aupr=best['AUPR'], best_epoch=best['epoch'], last_auc=last['AUC'], last_aupr=last['AUPR']))
                msg = f"== PLMF-MDA cold {args.dataset} {args.protocol} seed{seed} fold{fi}: best AUC {best['AUC']:.4f} AUPR {best['AUPR']:.4f} (ep{best['epoch']}) | last AUC {last['AUC']:.4f} AUPR {last['AUPR']:.4f} ({time.time()-t0:.0f}s)"
                print(msg, flush=True); log.write(msg + '\n'); log.flush()
    out = os.path.join(HERE, 'results', f'plmf_cold_{args.dataset}_{args.protocol}.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    import pandas as pd; df = pd.DataFrame(rows)
    print(f"## PLMF-MDA cold {args.dataset} {args.protocol}: best AUC {df.best_auc.mean():.4f}±{df.best_auc.std():.4f} AUPR {df.best_aupr.mean():.4f} | last AUC {df.last_auc.mean():.4f}±{df.last_auc.std():.4f} AUPR {df.last_aupr.mean():.4f}")


if __name__ == '__main__':
    main()
