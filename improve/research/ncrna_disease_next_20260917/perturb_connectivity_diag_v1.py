"""Label-free diagnostic: drug-perturbation signature x miRNA-target connectivity."""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

BASE = '/data1/licheng/incrd-hg/iNcRD-HG'
GMT_DIR = '/tmp/opencode'
MTI = '/data1/licheng/incrd-hg/data_external/pathway_20260913/mirtarbase/hsa_MTI.csv'
OUT = os.path.join(BASE, 'improve/research/ncrna_disease_next_20260917')


def load_gmt(path):
    sets = {}
    with open(path) as handle:
        for line in handle:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 3:
                continue
            name = parts[0].strip().lower()
            genes = {g.strip().upper() for g in parts[2:] if g.strip()}
            if genes:
                sets.setdefault(name, set()).update(genes)
    return sets


def main():
    up = {}
    for f in ['OldCMAP_up.gmt', 'DrugPert_up.gmt']:
        up.update(load_gmt(os.path.join(GMT_DIR, f)))
    down = {}
    for f in ['OldCMAP_down.gmt', 'DrugPert_down.gmt']:
        down.update(load_gmt(os.path.join(GMT_DIR, f)))

    a = np.load(os.path.join(BASE, 'improve/results/signed_regulatory_inputs_v1_20260916/arrays.npz'), allow_pickle=True)
    drug_names = [str(x).upper() for x in a['drug_names']]
    rid = pd.read_csv(os.path.join(BASE, 'improve/research/rna_gene_mapping_20260913/rna_identity.tsv'), sep='\t')
    mi_src = [rid[(rid.task == 'MiDrug') & (rid.rna_node_index_0based == i)].source_name.iloc[0] for i in range(605)]

    mti = pd.read_csv(MTI, usecols=['miRNA', 'Target Gene', 'Support Type'])
    mti['mi'] = mti.miRNA.str.lower()
    func = mti[mti['Support Type'] == 'Functional MTI']
    tgt_all, tgt_func = {}, {}
    for mi, gene in zip(mti.mi, mti['Target Gene']):
        tgt_all.setdefault(mi, set()).add(str(gene).upper())
    for mi, gene in zip(func.mi, func['Target Gene']):
        tgt_func.setdefault(mi, set()).add(str(gene).upper())

    import re

    def norm(s):
        return re.sub(r'[^a-z0-9]+', ' ', s.lower()).strip()

    up_norm = {norm(k): v for k, v in up.items()}
    down_norm = {norm(k): v for k, v in down.items()}

    def drug_sets(name):
        pat = r'\b' + re.escape(norm(name)) + r'\b'
        gu, gd = set(), set()
        hit = False
        for k, v in up_norm.items():
            if re.search(pat, k):
                gu |= v
                hit = True
        for k, v in down_norm.items():
            if re.search(pat, k):
                gd |= v
                hit = True
        return (gu, gd) if hit else (None, None)

    covered = [d for d in drug_names if drug_sets(d)[0] is not None]
    print('drugs with perturbation sets:', len(covered), '/', len(drug_names))
    perturb = {d: drug_sets(d) for d in covered}

    arr = np.load(os.path.join(BASE, 'improve/results/numeric_embedding_20260912/features/inner/MiDrug/arrays.npz'))
    pv = arr['pairs_val']; y = arr['y_val']
    ref = np.load(os.path.join(BASE, 'improve/results/tabicl_pathway_context_20260913/predictions/inner/MiDrug/s666/scores.npz'))['tiered_0p5000']

    def feats(targets):
        rows = []
        for r, d in zip(pv[0], pv[1]):
            pair = perturb.get(drug_names[d])
            tg = targets.get(str(mi_src[r]).lower(), set())
            if pair is None or not tg:
                rows.append([np.nan] * 6)
                continue
            gu, gd = pair
            iu, idn = tg & gu, tg & gd
            rows.append([len(iu), len(idn), len(iu) - len(idn),
                         len(iu) / len(tg), len(idn) / len(tg), len(tg)])
        return np.array(rows, dtype=float)

    for label, targets in [('all-MTI', tgt_all), ('functional-MTI', tgt_func)]:
        F = feats(targets)
        mask = np.isfinite(F[:, 0])
        print('\n==', label, 'covered pairs', int(mask.sum()), '/', len(y), 'pos', int(y[mask].sum()))
        names = ['n_up', 'n_down', 'signed', 'frac_up', 'frac_down', 'n_targets']
        for j, nm in enumerate(names):
            s = F[mask, j]
            if np.std(s) < 1e-12:
                continue
            print('  %-10s AUC %.4f AP %.4f' % (nm, roc_auc_score(y[mask], s), average_precision_score(y[mask], s)))
        # blend with reference on covered pairs
        lr = np.log(np.clip(ref, 1e-6, 1 - 1e-6) / (1 - np.clip(ref, 1e-6, 1 - 1e-6)))
        for j, nm in [(2, 'signed'), (0, 'n_up'), (1, 'n_down')]:
            z = F[:, j].copy()
            med = np.nanmedian(z)
            z[~np.isfinite(z)] = med
            z = (z - np.nanmean(z)) / (np.nanstd(z) + 1e-9)
            for w in [0.05, 0.1, 0.2, 0.3]:
                s = lr + w * z
                print('   ref+%.2f*%s  AUC %.4f AP %.4f' % (w, nm, roc_auc_score(y, s), average_precision_score(y, s)))
        print('   ref           AUC %.4f AP %.4f' % (roc_auc_score(y, ref), average_precision_score(y, ref)))


if __name__ == '__main__':
    main()
