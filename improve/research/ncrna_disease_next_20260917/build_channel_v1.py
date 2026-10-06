"""Label-free diagnostic: ncRNA-disease x drug-disease bridge for iNcRD-HG pairs.

Reads external disease-association tables and asks whether a disease-level
bridge carries any signal on the frozen MiDrug / LncDrug validation pairs.
No model is trained; no label beyond the evaluation labels is read.
"""
import gzip
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

DATA = '/data1/licheng/incrd-hg/data_external/ncrna_disease_20260917'
BASE = '/data1/licheng/incrd-hg/iNcRD-HG'
OUT = os.path.join(BASE, 'improve/research/ncrna_disease_next_20260917')


def norm_disease(name, mesh):
    if isinstance(mesh, str) and mesh.strip():
        digits = re.sub(r'[^0-9]', '', mesh)
        if digits:
            return 'MESH:' + digits
    text = re.sub(r'[^a-z0-9]+', ' ', str(name).lower()).strip()
    return 'NAME:' + text


def mir_key(name):
    s = str(name).strip().lower().replace('_', '-')
    s = s.replace('hsa-mir-', 'hsa-mir-').replace('hsa-let-', 'hsa-let-')
    exact = s
    fam = re.sub(r'-(5p|3p|5|3)$', '', s)
    fam = re.sub(r'-(\d+)$', '', fam) if re.match(r'^hsa-(let|mir)-', fam) else fam
    return exact, fam


def load_rnadisease():
    rows = []
    mi = pd.read_excel(os.path.join(DATA, 'RNADisease_miRNA/RNADiseasev4.0_RNA-disease_experiment_miRNA.xlsx'))
    mi = mi[mi.specise == 'Homo sapiens']
    for sym, dis, mesh, score in zip(mi['RNA Symbol'], mi['Disease Name'], mi['MeSH ID'], mi['score']):
        rows.append(('mir', str(sym), norm_disease(dis, mesh), float(score) if pd.notna(score) else 1.0))
    ln = pd.read_excel(os.path.join(DATA, 'RNADisease_lncRNA/RNADiseasev4.0_RNA-disease_experiment_lncRNA.xlsx'))
    ln = ln[ln.specise == 'Homo sapiens']
    for sym, dis, mesh, score in zip(ln['RNA Symbol'], ln['Disease Name'], ln['MeSH ID'], ln['score']):
        rows.append(('lnc', str(sym), norm_disease(dis, mesh), float(score) if pd.notna(score) else 1.0))
    return pd.DataFrame(rows, columns=['kind', 'rna', 'disease', 'weight'])


def load_hmdd():
    h = pd.read_csv(os.path.join(DATA, 'hmdd_v4_alldata.txt'), sep='\t')
    rows = [('mir', str(m), norm_disease(d, None), 1.0) for m, d in zip(h.miRNA, h.disease)]
    return pd.DataFrame(rows, columns=['kind', 'rna', 'disease', 'weight'])


def load_ctd(drugs):
    cols = ['ChemicalName', 'ChemicalID', 'CasRN', 'DiseaseName', 'DiseaseID',
            'DirectEvidence', 'InferenceGeneSymbol', 'InferenceScore', 'OmimIDs', 'PubMedIDs']
    keep = []
    with gzip.open(os.path.join(DATA, 'CTD_chemicals_diseases.tsv.gz'), 'rt') as handle:
        next(handle)
        for line in handle:
            parts = line.rstrip('\n').split('\t')
            if len(parts) != len(cols):
                continue
            name = parts[0].upper()
            if name not in drugs:
                continue
            evidence = parts[5]
            if evidence not in ('therapeutic', 'marker/mechanism'):
                continue
            keep.append((name, norm_disease(parts[3], parts[4])))
    return pd.DataFrame(keep, columns=['drug', 'disease']).drop_duplicates()


def main():
    a = np.load(os.path.join(BASE, 'improve/results/signed_regulatory_inputs_v1_20260916/arrays.npz'), allow_pickle=True)
    mi_names = [str(x) for x in a['rna_names']]
    drug_names = [str(x).upper() for x in a['drug_names']]
    rid = pd.read_csv(os.path.join(BASE, 'improve/research/rna_gene_mapping_20260913/rna_identity.tsv'), sep='\t')
    lnc_names = rid[rid.task == 'LncDrug'].sort_values('rna_node_index_0based').rna_name.tolist()
    lnc_src = lnc_names

    rd = load_rnadisease()
    hd = load_hmdd()
    dis = pd.concat([rd, hd], ignore_index=True)

    # RNA -> set(disease) keyed by exact and family
    rna_exact = {}
    rna_family = {}
    for kind, rna, disease, w in zip(dis.kind, dis.rna, dis.disease, dis.weight):
        if kind == 'mir':
            e, f = mir_key(rna)
        else:
            e = f = str(rna).strip().upper()
        rna_exact.setdefault((kind, e), {})[disease] = max(w, rna_exact.setdefault((kind, e), {}).get(disease, 0.0))
        rna_family.setdefault((kind, f), {})[disease] = max(w, rna_family.setdefault((kind, f), {}).get(disease, 0.0))

    drug_map = load_ctd(set(drug_names))
    print('CTD matched drugs:', drug_map.drug.nunique(), '/', len(drug_names))
    drug_dis = {}
    for d, disname in zip(drug_map.drug, drug_map.disease):
        drug_dis.setdefault(d, set()).add(disname)
    print('drugs with >=1 disease:', len(drug_dis), 'mean diseases',
          np.mean([len(v) for v in drug_dis.values()]))

    def rna_dis_set(kind, src):
        if kind == 'mir':
            e, f = mir_key(src)
            s = dict(rna_exact.get((kind, e), {}))
            if not s:
                s = dict(rna_family.get((kind, f), {}))
            return s
        s = rna_exact.get((kind, str(src).strip().upper()), {})
        return dict(s)

    mir_sets = [rna_dis_set('mir', s) for s in
                [rid[(rid.task == 'MiDrug') & (rid.rna_node_index_0based == i)].source_name.iloc[0]
                 if len(rid[(rid.task == 'MiDrug') & (rid.rna_node_index_0based == i)]) else mi_names[i]
                 for i in range(len(mi_names))]]
    lnc_sets = [rna_dis_set('lnc', s) for s in lnc_src]
    print('miRNA with diseases:', sum(1 for s in mir_sets if s), '/', len(mir_sets))
    print('lncRNA with diseases:', sum(1 for s in lnc_sets if s), '/', len(lnc_sets))

    def pair_features(kind, sets, arrays_path):
        arr = np.load(arrays_path)
        ptr, pv = arr['pairs_train'], arr['pairs_val']
        out = {}
        for name, (pairs, y) in {'train': (ptr, arr['y_train']), 'val': (pv, arr['y_val'])}.items():
            local = sets
            feats = []
            for r, d in zip(pairs[0], pairs[1]):
                rs = local[r]
                ds = drug_dis.get(drug_names[d], set())
                inter = rs.keys() & ds
                wsum = sum(rs[k] for k in inter)
                union = len(rs) + len(ds) - len(inter)
                jac = len(inter) / union if union else 0.0
                cos = len(inter) / (np.sqrt(len(rs)) * np.sqrt(len(ds))) if rs and ds else 0.0
                feats.append([len(inter), wsum, jac, cos, len(rs), len(ds)])
            out[name] = np.array(feats, dtype=np.float64)
        return out

    res = {}
    for task, kind, sets, path in [
        ('MiDrug', 'mir', mir_sets, os.path.join(BASE, 'improve/results/numeric_embedding_20260912/features/inner/MiDrug/arrays.npz')),
        ('LncDrug', 'lnc', lnc_sets, os.path.join(BASE, 'improve/results/numeric_embedding_20260912/features/inner/LncDrug/arrays.npz')),
    ]:
        f = pair_features(kind, sets, path)
        arr = np.load(path)
        y = arr['y_val']
        ref = None
        rp = os.path.join(BASE, 'improve/results/tabicl_pathway_context_20260913/predictions/inner/%s/s666/scores.npz' % task)
        if os.path.exists(rp):
            ref = np.load(rp)['tiered_0p5000']
        names = ['n_shared', 'w_shared', 'jaccard', 'cosine', 'n_rna_dis', 'n_drug_dis']
        print('\n=====', task, 'val n', len(y), 'pos', int(y.sum()))
        for j, nm in enumerate(names):
            s = f['val'][:, j]
            if np.std(s) < 1e-12:
                print('  %-10s constant' % nm)
                continue
            auc = roc_auc_score(y, s)
            ap = average_precision_score(y, s)
            extra = ''
            if ref is not None:
                r = np.corrcoef(s, y - ref)[0, 1] if 'ref' not in dir() else 0
                extra = ' corr_resid=%.3f' % np.corrcoef(s, y - ref)[0, 1]
            print('  %-10s AUC %.4f AP %.4f%s' % (nm, auc, ap, extra))
        res[task] = {nm: float(roc_auc_score(y, f['val'][:, j])) for j, nm in enumerate(names)}
    with open(os.path.join(OUT, 'CHANNEL_DIAGNOSTIC_V1.json'), 'w') as handle:
        json.dump(res, handle, indent=1)
    print('\nwrote', os.path.join(OUT, 'CHANNEL_DIAGNOSTIC_V1.json'))


if __name__ == '__main__':
    main()
