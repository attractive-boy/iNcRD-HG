# -*- coding: utf-8 -*-
"""Build the MPHGNN-style miRNA-gene-drug heterogeneous inputs for PLMF-MDA's
MDRdataset / MDSdataset (label-free; the miRNA-drug relation is added per split
from training positives inside the runner).

MPHGNN (BiB 2026) recipe            ->  what is used here for the PLMF-MDA nodes
  miRNA-gene   ENCORI CLIP targets   ->  ENCORI miRNATarget API (hg38, mRNA), filtered
                                         clipExpNum >= CLIP and >= PROG prediction
                                         programs; top-MAXDEG genes per miRNA by clipExpNum
  gene-drug    DrugBank targets      ->  ChEMBL mechanism-of-action targets (UniChem
                                         DrugBank->ChEMBL; DrugBank blocked)
  gene-gene    HumanNet functional   ->  HumanNet v3 FN; within the gene set keep the
                                         top-GG_K LLS neighbours per gene, symmetrised
  miRNA feat   CCLE expression       ->  'kmer': 2+3-mer frequencies (80-d, all nodes);
                                         'expr': CCLE 938 cell lines (row min-max, from
                                         incrd-run MiExp.csv) where available + flag
  drug feat    ChemBERTa (row minmax)->  ChemBERTa-zinc-base-v1 mean pooled, row min-max
                                         (verified identical construction to MPHGNN's)
  gene feat    gene-similarity 200-d ->  200-d truncated SVD of the sym-normalised LLS
                                         gene-gene adjacency (functional similarity)
Outputs cache/graph_{MDR,MDS}.npz with node features, edge lists and summary."""
import argparse, gzip, json, os
import numpy as np, pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds
from itertools import product

HERE = os.path.dirname(os.path.abspath(__file__))
PLMF = '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/new5_repro_20260919/repos/PLMF-MDA-main/data/unpacked2/data'
HUMANNET = '/data1/licheng/incrd-hg/data_external/humannet_20260921/HumanNet-FN.tsv.gz'
MIEXP = '/data1/licheng/incrd-hg/incrd-run/2.construct_network/MiExp.csv'
SUF = {'MDR': 'R', 'MDS': 'S'}

ap = argparse.ArgumentParser()
ap.add_argument('--clip', type=int, default=5, help='min ENCORI clipExpNum')
ap.add_argument('--prog', type=int, default=3, help='min number of target-prediction programs')
ap.add_argument('--maxdeg', type=int, default=50, help='max genes per miRNA (top by clipExpNum)')
ap.add_argument('--gg_k', type=int, default=10, help='HumanNet top-k neighbours per gene')
ap.add_argument('--gene_dim', type=int, default=200)
a = ap.parse_args()


def kmer_feat(seqs, ks=(2, 3)):
    alph = 'ACGU'
    cols = []
    for k in ks:
        cols += [''.join(p) for p in product(alph, repeat=k)]
    idx = {c: i for i, c in enumerate(cols)}
    X = np.zeros((len(seqs), len(cols)), np.float32)
    for r, s in enumerate(seqs):
        s = s.upper().replace('T', 'U')
        for k in ks:
            n = max(len(s) - k + 1, 1)
            for i in range(len(s) - k + 1):
                j = idx.get(s[i:i + k])
                if j is not None:
                    X[r, j] += 1.0 / n
    return X


def main():
    enc = pd.read_csv(os.path.join(HERE, 'cache', 'encori_targets.tsv.gz'), sep='\t', dtype=str)
    for c in ['clipExpNum', 'degraExpNum', 'PITA', 'RNA22', 'miRmap', 'microT', 'miRanda', 'PicTar', 'TargetScan']:
        enc[c] = pd.to_numeric(enc[c], errors='coerce').fillna(0)
    enc['prog'] = enc[['PITA', 'RNA22', 'miRmap', 'microT', 'miRanda', 'PicTar', 'TargetScan']].gt(0).sum(1)
    enc = enc[(enc.clipExpNum >= a.clip) & (enc.prog >= a.prog)]
    enc = enc.sort_values('clipExpNum', ascending=False).drop_duplicates(['miRNAname', 'geneID'])
    # Ensembl gene id -> Entrez via geneName is not needed: HumanNet uses Entrez; map through
    # the ENCORI geneName -> UniProt-derived symbols of drug targets and HumanNet symbol file.
    sym = pd.read_csv(HUMANNET.replace('.tsv.gz', '.symbol.tsv.gz'), sep='\t', header=None, comment='#') \
        if os.path.exists(HUMANNET.replace('.tsv.gz', '.symbol.tsv.gz')) else None
    hn = pd.read_csv(HUMANNET, sep='\t', header=None, comment='#', names=['g1', 'g2', 'lls'])
    ent2sym = json.load(open(os.path.join(HERE, 'cache', 'entrez_symbol.json'))) if os.path.exists(os.path.join(HERE, 'cache', 'entrez_symbol.json')) else {}
    dt = json.load(open(os.path.join(HERE, 'cache', 'drug_targets.json')))['drugs']
    exp = pd.read_csv(MIEXP, index_col=0)
    exp = exp.loc[sorted(exp.index, key=lambda c: int(c.split('-')[-1]))]  # same cell-line order as incrd-hg
    expT = exp.T
    expT.index = [i.upper() for i in expT.index]

    for ds in ('MDR', 'MDS'):
        suf = SUF[ds]
        mir = pd.read_excel(f'{PLMF}/{ds}/miRNA_ID_{suf}.xlsx')
        dru = pd.read_excel(f'{PLMF}/{ds}/drug_ID_{suf}.xlsx')
        mnames = mir['miRNA_Name'].astype(str).tolist()
        dids = dru['DrugBank_ID'].astype(str).tolist()
        # --- gene universe: symbols (ENCORI geneName / UniProt symbol), Entrez via drug targets + symbol map
        # ENCORI gives Ensembl id + symbol; HumanNet is Entrez. Use the symbol<->Entrez table (cache/entrez_symbol.json,
        # built by build_gene_ids.py from HumanNet symbol file) to place ENCORI genes on HumanNet.
        sym2ent = {v: int(k) for k, v in ent2sym.items()}
        e = enc[enc.miRNAname.isin(mnames)].copy()
        e['entrez'] = e.geneName.map(sym2ent)
        n_unmapped = e.entrez.isna().sum()
        e = e.dropna(subset=['entrez'])
        e['entrez'] = e.entrez.astype(int)
        # top-maxdeg per miRNA
        e = e.sort_values(['miRNAname', 'clipExpNum'], ascending=[True, False]).groupby('miRNAname').head(a.maxdeg)
        mg = [(mnames.index(m), g) for m, g in zip(e.miRNAname, e.entrez)]
        gd = [(g['entrez'], dids.index(d)) for d in dids for g in dt.get(d, {}).get('genes', [])]
        genes = sorted({g for _, g in mg} | {g for g, _ in gd})
        gidx = {g: i for i, g in enumerate(genes)}
        mg_e = np.array([(m, gidx[g]) for m, g in mg], np.int64).reshape(-1, 2)
        gd_e = np.array([(gidx[g], d) for g, d in gd], np.int64).reshape(-1, 2)
        # --- HumanNet within gene set
        h = hn[hn.g1.isin(gidx) & hn.g2.isin(gidx)]
        h = pd.concat([h, h.rename(columns={'g1': 'g2', 'g2': 'g1'})], ignore_index=True)
        h['i'] = h.g1.map(gidx); h['j'] = h.g2.map(gidx)
        h = h[h.i != h.j].drop_duplicates(['i', 'j'])
        # gene feature: SVD of sym-normalised weighted adjacency (+ self loops)
        n = len(genes)
        A = sparse.coo_matrix((h.lls.to_numpy(np.float64), (h.i.to_numpy(), h.j.to_numpy())), shape=(n, n)).tocsr()
        A = A + sparse.identity(n, format='csr')
        dinv = 1.0 / np.sqrt(np.asarray(A.sum(1)).ravel())
        An = sparse.diags(dinv) @ A @ sparse.diags(dinv)
        k = min(a.gene_dim, n - 2)
        U, S, _ = svds(An.asfptype(), k=k, random_state=0)
        order = np.argsort(-S); U, S = U[:, order], S[order]
        gene_x = (U * np.sqrt(S)).astype(np.float32)
        if k < a.gene_dim:
            gene_x = np.concatenate([gene_x, np.zeros((n, a.gene_dim - k), np.float32)], 1)
        # top-k neighbours per gene for metapath gene-gene links
        top = h.sort_values(['i', 'lls'], ascending=[True, False]).groupby('i').head(a.gg_k)
        gg = set(zip(top.i, top.j)) | set(zip(top.j, top.i))
        gg_e = np.array(sorted(gg), np.int64).reshape(-1, 2)
        # --- node features
        kmer = kmer_feat(mir['miRNA_Sequence'].astype(str).tolist())
        has = np.array([m.upper() in expT.index for m in mnames])
        ex = np.zeros((len(mnames), expT.shape[1]), np.float32)
        if has.any():
            v = expT.loc[[m.upper() for m, hh in zip(mnames, has) if hh]].to_numpy(np.float32)
            v = (v - v.min(1, keepdims=True)) / (v.max(1, keepdims=True) - v.min(1, keepdims=True) + 1e-8)
            ex[has] = v
        cb = np.load(os.path.join(HERE, 'cache', 'drug_chemberta_zinc.npz'))
        cbmap = {d: i for i, d in enumerate(cb['drugbank_id'])}
        dx = cb['x'][[cbmap[d] for d in dids]].astype(np.float32)
        dx = (dx - dx.min(1, keepdims=True)) / (dx.max(1, keepdims=True) - dx.min(1, keepdims=True) + 1e-8)
        summary = dict(dataset=ds, n_mirna=len(mnames), n_drug=len(dids), n_gene=n,
                       mirna_gene_edges=int(len(mg_e)), mirna_with_gene=int(len(set(mg_e[:, 0]))),
                       gene_drug_edges=int(len(gd_e)), drug_with_gene=int(len(set(gd_e[:, 1]))),
                       gene_gene_edges_topk=int(len(gg_e)), gene_gene_edges_full=int(len(h)),
                       encori_rows_unmapped_symbol=int(n_unmapped), mirna_with_ccle_expr=int(has.sum()),
                       filters=dict(clip=a.clip, prog=a.prog, maxdeg=a.maxdeg, gg_k=a.gg_k, gene_dim=a.gene_dim))
        print(json.dumps(summary))
        np.savez_compressed(os.path.join(HERE, 'cache', f'graph_{ds}.npz'),
                            mirna_names=np.array(mnames), drug_ids=np.array(dids), genes=np.array(genes),
                            mirna_gene=mg_e, gene_drug=gd_e, gene_gene=gg_e,
                            mirna_kmer=kmer, mirna_expr=ex, mirna_has_expr=has, drug_x=dx, gene_x=gene_x,
                            summary=json.dumps(summary))


if __name__ == '__main__':
    main()
