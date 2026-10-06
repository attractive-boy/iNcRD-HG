"""Disease-bridge information channel on top of the accepted shared anchor.

score = sigmoid((1-w)*logit(reference) + w*mean_m logit(member_m on [bank | disease]))

bank            = the accepted frozen feature bank (context76 / tiered62)
disease         = label-free ncRNA-disease x drug-disease bridge columns
control         = same columns from a fixed permutation of the drug->disease map

Evaluation follows the accepted shared-anchor gates on the historical inner
partition and the three development resplits.  Run:
    /data1/licheng/incrd-hg/.venv-dgl/bin/python -m \
        improve.research.ncrna_disease_next_20260917.disease_anchor_verify_v1
"""
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG')
from improve import tabular_foundation as base  # noqa: E402
from improve.context_anchor_verify import learner, SEED, SPLITS  # noqa: E402

DATA = Path('/data1/licheng/incrd-hg/data_external/ncrna_disease_20260917')
ROOT = Path('/data1/licheng/incrd-hg/iNcRD-HG/improve')
INNER = ROOT / 'results/foundation_transfer_20260914/inputs'
INREF = ROOT / 'results/tabicl_pathway_context_20260913'
RB = ROOT / 'results/resplit_foundation_verify_20260914'
RREF = ROOT / 'results/transfer_splits_20260914'
OUT = ROOT / 'research/ncrna_disease_next_20260917'
TASKS = ('MiDrug', 'LncDrug')
SEEDS = (666, 0, 1)
SUBSET = ('hgb', 'et', 'mlp', 'lr')
WEIGHTS = (0.15, 0.2, 0.25, 0.3, 0.4)


def norm_disease(name, mesh):
    if isinstance(mesh, str) and mesh.strip():
        digits = re.sub(r'[^0-9]', '', mesh)
        if digits:
            return 'MESH:' + digits
    return 'NAME:' + re.sub(r'[^a-z0-9]+', ' ', str(name).lower()).strip()


def mir_key(name):
    s = str(name).strip().lower().replace('_', '-')
    exact = s
    fam = re.sub(r'-(5p|3p|5|3)$', '', s)
    fam = re.sub(r'-(\d+)$', '', fam) if re.match(r'^hsa-(let|mir)-', fam) else fam
    return exact, fam


def load_disease_tables():
    rows = []
    for sub, kind in [('RNADisease_miRNA/RNADiseasev4.0_RNA-disease_experiment_miRNA.xlsx', 'mir'),
                      ('RNADisease_lncRNA/RNADiseasev4.0_RNA-disease_experiment_lncRNA.xlsx', 'lnc')]:
        x = pd.read_excel(DATA / sub)
        x = x[x.specise == 'Homo sapiens']
        for sym, dis, mesh, score in zip(x['RNA Symbol'], x['Disease Name'], x['MeSH ID'], x['score']):
            rows.append((kind, str(sym), norm_disease(dis, mesh), float(score) if pd.notna(score) else 1.0))
    h = pd.read_csv(DATA / 'hmdd_v4_alldata.txt', sep='\t')
    for m, d in zip(h.miRNA, h.disease):
        rows.append(('mir', str(m), norm_disease(d, None), 1.0))
    return pd.DataFrame(rows, columns=['kind', 'rna', 'disease', 'weight'])


def load_ctd(drugs):
    import gzip
    keep = []
    with gzip.open(DATA / 'CTD_chemicals_diseases.tsv.gz', 'rt') as handle:
        next(handle)
        for line in handle:
            parts = line.rstrip('\n').split('\t')
            if len(parts) != 10:
                continue
            name = parts[0].upper()
            if name not in drugs or parts[5] not in ('therapeutic', 'marker/mechanism'):
                continue
            keep.append((name, norm_disease(parts[3], parts[4])))
    return pd.DataFrame(keep, columns=['drug', 'disease']).drop_duplicates()


class Channel:
    def __init__(self, permute=False, seed=20260917):
        a = np.load(ROOT / 'results/signed_regulatory_inputs_v1_20260916/arrays.npz', allow_pickle=True)
        self.mi_names = [str(x) for x in a['rna_names']]
        self.drug_names = [str(x).upper() for x in a['drug_names']]
        rid = pd.read_csv(ROOT / 'research/rna_gene_mapping_20260913/rna_identity.tsv', sep='\t')
        self.mi_src = [rid[(rid.task == 'MiDrug') & (rid.rna_node_index_0based == i)].source_name.iloc[0]
                       for i in range(len(self.mi_names))]
        self.lnc_src = rid[rid.task == 'LncDrug'].sort_values('rna_node_index_0based').rna_name.tolist()

        dis = load_disease_tables()
        exact, family = {}, {}
        for kind, rna, disease, w in zip(dis.kind, dis.rna, dis.disease, dis.weight):
            e, f = (mir_key(rna) if kind == 'mir' else (str(rna).strip().upper(),) * 2)
            exact.setdefault((kind, e), {})[disease] = max(w, exact.setdefault((kind, e), {}).get(disease, 0.0))
            family.setdefault((kind, f), {})[disease] = max(w, family.setdefault((kind, f), {}).get(disease, 0.0))
        self.exact, self.family = exact, family

        dm = load_ctd(set(self.drug_names))
        drug_dis = {}
        for d, s in zip(dm.drug, dm.disease):
            drug_dis.setdefault(d, set()).add(s)
        drugs_with = [d for d in self.drug_names if d in drug_dis]
        if permute:
            rng = np.random.default_rng(seed)
            shuffled = list(rng.permutation(drugs_with))
            drug_dis = {d: drug_dis[s] for d, s in zip(drugs_with, shuffled)}
        self.drug_dis = drug_dis
        freq = Counter()
        for v in drug_dis.values():
            freq.update(v)
        n = max(len(drug_dis), 1)
        self.idf = lambda x: float(np.log(n / freq.get(x, 1)))

    def rna_sets(self, task):
        if task == 'MiDrug':
            out = []
            for s in self.mi_src:
                e, f = mir_key(s)
                out.append(dict(self.exact.get(('mir', e), {}) or self.family.get(('mir', f), {})))
            return out
        return [dict(self.exact.get(('lnc', str(s).upper()), {})) for s in self.lnc_src]

    def features(self, task, pairs):
        sets = self.rna_sets(task)
        rows = []
        for r, d in zip(pairs[0], pairs[1]):
            rs = sets[r]
            ds = self.drug_dis.get(self.drug_names[d], set())
            inter = [k for k in ds if k in rs]
            rows.append([len(inter), sum(self.idf(k) for k in inter),
                         sum(self.idf(k) * rs[k] for k in inter), len(rs), len(ds)])
        return np.asarray(rows, dtype=np.float32)


def inner_reference(task, seed):
    with np.load(INREF / ('predictions/inner/%s/s%d/scores.npz' % (task, seed)), allow_pickle=False) as z:
        return z['tiered_0p5000'].copy(), z['y_val'].copy(), z['pairs_val'].copy()


def resplit_reference(task, split):
    with np.load(RREF / ('runs/%s/%d/tiered/scores.npz' % (task, split)), allow_pickle=False) as z:
        return z['pred'].copy(), z['y'].copy(), z['pairs'].copy()


def bank_arrays(bank, task, scope, split):
    if scope == 'inner':
        return np.load(INNER / ('%s_%s.npz' % (bank, task)), allow_pickle=False)
    return np.load(RB / ('inputs_s%d/%s_%s.npz' % (split, bank, task)), allow_pickle=False)


def component(x_tr, y_tr, x_q):
    logits = []
    for name in SUBSET:
        m = learner(name)
        m.fit(x_tr, y_tr)
        p = np.clip(m.predict_proba(x_q)[:, 1], 1e-12, 1 - 1e-12)
        logits.append(np.log(p) - np.log1p(-p))
    return np.mean(logits, axis=0)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    real = Channel(permute=False)
    ctrl = Channel(permute=True)
    rows = []
    for bank in ('context76', 'tiered62'):
        for scope in ('inner', 'resplit'):
            splits = [None] if scope == 'inner' else list(SPLITS)
            for split in splits:
                for task in TASKS:
                    z = bank_arrays(bank, task, scope, split)
                    x_tr, y_tr, x_q = z['x_train'], z['y_train'], z['x_val']
                    ptr, pq = z['pairs_train'], z['pairs_val']
                    dr_tr, dr_q = real.features(task, ptr), real.features(task, pq)
                    dc_tr, dc_q = ctrl.features(task, ptr), ctrl.features(task, pq)
                    comps = {
                        'bank': component(x_tr, y_tr, x_q),
                        'bank+disease': component(np.c_[x_tr, dr_tr], y_tr, np.c_[x_q, dr_q]),
                        'bank+control': component(np.c_[x_tr, dc_tr], y_tr, np.c_[x_q, dc_q]),
                    }
                    for name, comp in comps.items():
                        for w in WEIGHTS:
                            if scope == 'inner':
                                for seed in SEEDS:
                                    ref, y, pairs = inner_reference(task, seed)
                                    val = expit((1 - w) * (np.log(np.clip(ref, 1e-12, 1 - 1e-12)) -
                                                           np.log1p(-np.clip(ref, 1e-12, 1 - 1e-12))) + w * comp)
                                    m, r = base.metrics(y, val, pairs), base.metrics(y, ref, pairs)
                                    rows.append(dict(bank=bank, recipe=name, w=w, scope='inner', task=task,
                                                     key='seed%d' % seed, d_auc=m['auc'] - r['auc'],
                                                     d_ap=m['ap'] - r['ap'], auc=m['auc'], ap=m['ap']))
                            else:
                                ref, y, pairs = resplit_reference(task, split)
                                val = expit((1 - w) * (np.log(np.clip(ref, 1e-12, 1 - 1e-12)) -
                                                       np.log1p(-np.clip(ref, 1e-12, 1 - 1e-12))) + w * comp)
                                m, r = base.metrics(y, val, pairs), base.metrics(y, ref, pairs)
                                rows.append(dict(bank=bank, recipe=name, w=w, scope='resplit', task=task,
                                                 key='split%d' % split, d_auc=m['auc'] - r['auc'],
                                                 d_ap=m['ap'] - r['ap'], auc=m['auc'], ap=m['ap']))
                    print('done', bank, scope, split, task, flush=True)

    t = pd.DataFrame(rows)
    t.to_csv(OUT / 'disease_anchor_metrics_v1.csv', index=False)
    summary = []
    for (bank, recipe), sub in t.groupby(['bank', 'recipe']):
        for w, sw in sub.groupby('w'):
            g = {}
            for task in TASKS:
                it = sw[(sw.scope == 'inner') & (sw.task == task)]
                rt = sw[(sw.scope == 'resplit') & (sw.task == task)]
                inner_ok = (bool((it.d_auc.values >= 0).all() and (it.d_ap.values >= 0).all()) if task == 'LncDrug'
                            else bool(it.d_auc.mean() > 0 and it.d_ap.mean() > 0 and
                                      (it.d_auc > 0).sum() >= 2 and (it.d_ap > 0).sum() >= 2))
                res_ok = (bool((rt.d_auc.values > 0).all() and (rt.d_ap.values > 0).all()) if task == 'MiDrug'
                          else bool((rt.d_auc.values >= 0).all() and (rt.d_ap.values >= 0).all()))
                g[task] = dict(inner_ok=inner_ok, resplit_ok=res_ok,
                               inner_d_auc=float(it.d_auc.mean()), inner_d_ap=float(it.d_ap.mean()),
                               resplit_d_auc=float(rt.d_auc.mean()), resplit_d_ap=float(rt.d_ap.mean()))
            summary.append(dict(bank=bank, recipe=recipe, w=w,
                                inner_Mi_dAUC=g['MiDrug']['inner_d_auc'], inner_Mi_dAP=g['MiDrug']['inner_d_ap'],
                                inner_Ln_dAUC=g['LncDrug']['inner_d_auc'], inner_Ln_dAP=g['LncDrug']['inner_d_ap'],
                                resplit_Mi_dAUC=g['MiDrug']['resplit_d_auc'], resplit_Mi_dAP=g['MiDrug']['resplit_d_ap'],
                                resplit_Ln_dAUC=g['LncDrug']['resplit_d_auc'], resplit_Ln_dAP=g['LncDrug']['resplit_d_ap'],
                                joint=bool(all(g[k][p] for k in TASKS for p in ('inner_ok', 'resplit_ok')))))
    s = pd.DataFrame(summary).sort_values(['joint', 'resplit_Mi_dAP'], ascending=False)
    s.to_csv(OUT / 'disease_anchor_summary_v1.csv', index=False)
    result = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                  channel='ncRNA-disease x drug-disease bridge (RNADisease v4 + HMDD v4 + CTD)',
                  composition='sigmoid((1-w)*logit(reference) + w*mean_m logit(member_m on [bank|disease]))',
                  gates='accepted shared-anchor inner + resplit gates; permutation control must fail',
                  joint_recipes=int(s.joint.sum()), total_recipes=int(len(s)),
                  best=s[s.joint].head(5).to_dict('records'))
    base.write_json(OUT / 'DISEASE_ANCHOR_ASSESSMENT_V1.json', result)
    print(s.head(15).to_string(index=False))
    print('joint:', int(s.joint.sum()), '/', len(s))


if __name__ == '__main__':
    run()
