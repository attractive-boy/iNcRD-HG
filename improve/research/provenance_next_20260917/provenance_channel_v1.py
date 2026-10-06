"""Curation-provenance channels for the accepted shared anchor (v1).

Every curated ncRNA-drug association in ncRNADrug is a hyper-relational fact
(RNA, drug, PMID, experimental context).  The accepted feature bank keeps only the
(RNA, drug) projection.  This module rebuilds, from TRAINING-POSITIVE RECORDS ONLY,
two label-side provenance channels:

  pmid  : study co-reporting.  A PMID is a hyperedge over the RNAs and drugs it
          reports.  For a query (r, d): number / weight of training PMIDs whose
          hyperedge contains both r and d (through other pairs), plus study-level
          exposure counts (distinct PMIDs per entity, high-throughput fraction).
  ctx   : experimental-context co-occurrence.  Cell line (Condition), phenotype and
          Cellosaurus-normalised disease tokens of r's training records vs d's
          training records, IDF-weighted.

Training rows are cross-fitted with the frozen fold ids (records of a fold's
positives are removed when that fold is featurised); query rows use all training
positives.  Permutation controls shuffle the PMID column (resp. context tokens)
across records of the task, preserving all marginals but destroying pair-level
co-membership.

The features are appended to the accepted context76/tiered62 bank and scored by the
accepted shared-anchor members (hgb, et, mlp, lr) fused at the accepted weight into
the frozen TabICLv2 reference.  No neural weight is touched.
"""
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG')
from improve import tabular_foundation as base  # noqa: E402
from improve.context_anchor_verify import learner  # noqa: E402

ROOT = Path('/data1/licheng/incrd-hg/iNcRD-HG/improve')
HERE = ROOT / 'research/provenance_next_20260917'
RECORDS = ROOT / 'research/exposure_next_20260917/records.csv'
TASKS = ('MiDrug', 'LncDrug')
SEEDS = (666, 0, 1)
SUBSET = ('hgb', 'et', 'mlp', 'lr')
PMID_NAMES = ('prov_shared_pmid', 'prov_shared_pmid_w', 'prov_rna_npmid', 'prov_drug_npmid',
              'prov_rna_nrec', 'prov_drug_nrec', 'prov_rna_ht_frac', 'prov_drug_ht_frac')
CTX_NAMES = ('ctx_cond_idf', 'ctx_pheno_idf', 'ctx_disease_idf', 'ctx_cond_n', 'ctx_pheno_n', 'ctx_disease_n')
CONTROL_SEED = 20260917


def surfaces(task):
    """(name, bank_dir_or_file pattern, fold-id source, reference loader)."""
    out = []
    out.append(dict(name='inner', bank=lambda b: ROOT / ('results/foundation_transfer_20260914/inputs/%s_%s.npz' % (b, task)),
                    folds=ROOT / ('results/tabicl_graph_context_20260912/features/inner/%s/arrays.npz' % task),
                    refs=[(s, ROOT / ('results/tabicl_pathway_context_20260913/predictions/inner/%s/s%d/scores.npz' % (task, s)), 'tiered_0p5000', 'y_val', 'pairs_val') for s in SEEDS]))
    for s in (42, 2026, 31415):
        out.append(dict(name='s%d' % s, bank=lambda b, s=s: ROOT / ('results/resplit_foundation_verify_20260914/inputs_s%d/%s_%s.npz' % (s, b, task)),
                        folds=ROOT / ('results/transfer_splits_20260914/inputs/%s/%d/arrays.npz' % (task, s)),
                        refs=[(0, ROOT / ('results/transfer_splits_20260914/runs/%s/%d/tiered/scores.npz' % (task, s)), 'pred', 'y', 'pairs')]))
    for s in (7, 77, 777):
        out.append(dict(name='x%d' % s, bank=lambda b, s=s: ROOT / ('results/transfer_splits_extra_20260914/inputs_s%d/%s_%s.npz' % (s, b, task)),
                        folds=ROOT / ('results/transfer_splits_extra_20260914/inputs/%s/%d/arrays.npz' % (task, s)),
                        refs=[(0, ROOT / ('results/transfer_splits_extra_20260914/runs/%s/%d/tiered/scores.npz' % (task, s)), 'pred', 'y', 'pairs')]))
    return out


def load_records(task, variant, seed=CONTROL_SEED):
    rec = pd.read_csv(RECORDS)
    rec = rec[rec.task == task].reset_index(drop=True)
    for col in ('cond', 'pheno', 'diseases'):
        rec[col] = rec[col].fillna('').astype(str)
    rng = np.random.default_rng(seed)
    if variant in ('perm_pmid', 'perm_all'):
        rec['PMID'] = rec.PMID.values[rng.permutation(len(rec))]
        rec['ht'] = rec.ht.values[rng.permutation(len(rec))]
    if variant in ('perm_ctx', 'perm_all'):
        for col in ('cond', 'pheno', 'diseases'):
            rec[col] = rec[col].values[rng.permutation(len(rec))]
    return rec


def tokens(s):
    return [t for t in str(s).split('|') if t]


class Provenance:
    """Feature tables from a set of training-positive pairs (support)."""

    def __init__(self, rec, support_pairs):
        keep = np.array([(int(a), int(b)) in support_pairs for a, b in zip(rec.ri, rec.di)])
        R = rec[keep]
        self.p_rna = defaultdict(set)
        self.p_drug = defaultdict(set)
        self.rna_of_p = defaultdict(set)
        self.drug_of_p = defaultdict(set)
        self.rna_rec = Counter()
        self.drug_rec = Counter()
        self.rna_ht = Counter()
        self.drug_ht = Counter()
        self.ctx_r = {c: defaultdict(Counter) for c in ('cond', 'pheno', 'diseases')}
        self.ctx_d = {c: defaultdict(Counter) for c in ('cond', 'pheno', 'diseases')}
        self.df = {c: Counter() for c in ('cond', 'pheno', 'diseases')}
        for r, d, p, ht, cond, pheno, dis in zip(R.ri, R.di, R.PMID, R.ht, R.cond, R.pheno, R.diseases):
            r, d, p = int(r), int(d), int(p)
            self.p_rna[r].add(p); self.p_drug[d].add(p)
            self.rna_of_p[p].add(r); self.drug_of_p[p].add(d)
            self.rna_rec[r] += 1; self.drug_rec[d] += 1
            self.rna_ht[r] += int(ht); self.drug_ht[d] += int(ht)
            for c, v in (('cond', cond), ('pheno', pheno), ('diseases', dis)):
                for t in set(tokens(v)):
                    self.ctx_r[c][r][t] += 1; self.ctx_d[c][d][t] += 1; self.df[c][t] += 1
        self.n_ctx = {c: max(sum(self.df[c].values()), 1) for c in self.df}

    def features(self, pairs):
        rows = []
        for r, d in zip(pairs[0], pairs[1]):
            r, d = int(r), int(d)
            pr, pdg = self.p_rna.get(r, set()), self.p_drug.get(d, set())
            shared = pr & pdg
            w = sum(1.0 / (len(self.rna_of_p[p]) * len(self.drug_of_p[p])) for p in shared)
            row = [len(shared), w, len(pr), len(pdg), self.rna_rec[r], self.drug_rec[d],
                   self.rna_ht[r] / max(self.rna_rec[r], 1), self.drug_ht[d] / max(self.drug_rec[d], 1)]
            idfs, ns = [], []
            for c in ('cond', 'pheno', 'diseases'):
                rc, dc = self.ctx_r[c].get(r, {}), self.ctx_d[c].get(d, {})
                sh = set(rc) & set(dc)
                idfs.append(sum(math.log(self.n_ctx[c] / self.df[c][t]) for t in sh))
                ns.append(len(sh))
            rows.append(row + idfs + ns)
        return np.asarray(rows, dtype=np.float32)


def channel_arrays(rec, pairs_train, y_train, fold_ids, pairs_val):
    """Cross-fitted training features + full-support query features."""
    pos = pairs_train[:, y_train == 1]
    all_support = set(zip(pos[0].tolist(), pos[1].tolist()))
    x_train = np.zeros((pairs_train.shape[1], len(PMID_NAMES) + len(CTX_NAMES)), dtype=np.float32)
    for k in np.unique(fold_ids):
        q = np.flatnonzero(fold_ids == k)
        fold_pos = set(zip(pairs_train[0, q][y_train[q] == 1].tolist(), pairs_train[1, q][y_train[q] == 1].tolist()))
        enc = Provenance(rec, all_support - fold_pos)
        x_train[q] = enc.features(pairs_train[:, q])
    enc = Provenance(rec, all_support)
    x_val = enc.features(pairs_val)
    return x_train, x_val


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return np.log(p) - np.log1p(-p)


def member_logits(x_tr, y_tr, x_q):
    out = {}
    for name in SUBSET:
        m = learner(name)
        m.fit(x_tr, y_tr)
        out[name] = logit(m.predict_proba(x_q)[:, 1])
    return out


def run_cell(task, surface, bank, variants, out_dir):
    """Fit members for every variant on one (task, surface, bank) cell; cache logits."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / ('%s_%s_%s.npz' % (task, surface['name'], bank))
    if dest.exists():
        return dest
    with np.load(surface['bank'](bank), allow_pickle=False) as z:
        x_tr, y_tr, x_q = z['x_train'], z['y_train'], z['x_val']
        ptr, pq = z['pairs_train'], z['pairs_val']
    with np.load(surface['folds'], allow_pickle=False) as z:
        fold_ids = z['fold_ids']
        assert np.array_equal(z['pairs_val'], pq)
        if 'pairs_train' in z.files:
            assert np.array_equal(z['pairs_train'], ptr)
    assert len(fold_ids) == len(y_tr)
    store = dict(y_val=y_tr[:0], pairs_val=pq)
    feats = {}
    for variant in ('real', 'perm_pmid', 'perm_ctx', 'perm_all'):
        rec = load_records(task, variant)
        feats[variant] = channel_arrays(rec, ptr, y_tr, fold_ids, pq)
        store['feat_train_' + variant] = feats[variant][0]
        store['feat_val_' + variant] = feats[variant][1]
    npm, nct = len(PMID_NAMES), len(CTX_NAMES)
    for label, (variant, cols) in variants.items():
        if variant is None:
            xt, xq = x_tr, x_q
        else:
            ft, fq = feats[variant]
            xt, xq = np.c_[x_tr, ft[:, cols]], np.c_[x_q, fq[:, cols]]
        ml = member_logits(xt, y_tr, xq)
        for name, v in ml.items():
            store['logit_%s_%s' % (label, name)] = v
        print('fitted', task, surface['name'], bank, label, flush=True)
    np.savez_compressed(dest, **store)
    return dest


def variant_table():
    npm, nct = len(PMID_NAMES), len(CTX_NAMES)
    pm, ct, both = list(range(npm)), list(range(npm, npm + nct)), list(range(npm + nct))
    return {
        'bank': (None, None),
        'bank+pmid': ('real', pm), 'bank+ctx': ('real', ct), 'bank+prov': ('real', both),
        'bank+pmid_perm': ('perm_pmid', pm), 'bank+ctx_perm': ('perm_ctx', ct), 'bank+prov_perm': ('perm_all', both),
    }


if __name__ == '__main__':
    task, sname, bank, out_dir = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    surface = next(s for s in surfaces(task) if s['name'] == sname)
    run_cell(task, surface, bank, variant_table(), out_dir)
