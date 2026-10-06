"""External information channels for the ncRNA-drug pair (label-free)."""
import gzip
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path('/data1/licheng/incrd-hg/data_external/ncrna_disease_20260917')
GMT = DATA / 'perturbation_gmt'
MTI = Path('/data1/licheng/incrd-hg/data_external/pathway_20260913/mirtarbase/hsa_MTI.csv')
LNCTARD = Path('/data1/licheng/incrd-hg/data_external/pathway_20260913/lnctard/lnctard2.0.zip')
ROOT = Path('/data1/licheng/incrd-hg/iNcRD-HG/improve')


def _load_gmt(path):
    sets = {}
    with open(path) as handle:
        for line in handle:
            p = line.rstrip('\n').split('\t')
            if len(p) < 3:
                continue
            genes = {g.strip().upper() for g in p[2:] if g.strip()}
            if genes:
                sets.setdefault(p[0].strip().lower(), set()).update(genes)
    return sets


def _norm(s):
    return re.sub(r'[^a-z0-9]+', ' ', str(s).lower()).strip()


class MechanismChannel:
    """Drug mechanism-disease (union over target genes' CTD diseases) x RNA disease."""

    def __init__(self, rna_channel, permute=False, seed=20260917):
        import json
        drug_dis = json.load(open(ROOT / 'research/ncrna_disease_next_20260917/gene_disease_v1.json'))
        drug_dis = {d.upper(): set(v) for d, v in drug_dis.items()}
        keys = list(drug_dis)
        if permute:
            rng = np.random.default_rng(seed)
            perm = list(rng.permutation(keys))
            drug_dis = {k: drug_dis[p] for k, p in zip(keys, perm)}
        self.drug_dis = drug_dis
        freq = Counter()
        for v in drug_dis.values():
            freq.update(v)
        self.rna = rna_channel
        n = max(len(drug_dis), 1)
        self.idf = lambda x: float(np.log(n / freq.get(x, 1)))

    def features(self, task, pairs):
        sets = self.rna.rna_sets(task)
        rows = []
        for r, d in zip(pairs[0], pairs[1]):
            rs = sets[r]
            ds = self.drug_dis.get(self.rna.drug_names[d], set())
            inter = [k for k in ds if k in rs]
            rows.append([len(inter), sum(self.idf(k) for k in inter),
                         sum(self.idf(k) * rs[k] for k in inter), len(ds)])
        return np.asarray(rows, dtype=np.float32)


class PerturbChannel:
    """Drug perturbation up/down gene sets crossed with RNA target genes."""

    def __init__(self, permute=False, seed=20260917):
        self.up, self.down = {}, {}
        for f in ['OldCMAP_up.gmt', 'DrugPert_up.gmt']:
            self.up.update(_load_gmt(GMT / f))
        for f in ['OldCMAP_down.gmt', 'DrugPert_down.gmt']:
            self.down.update(_load_gmt(GMT / f))
        self.up_n = {_norm(k): v for k, v in self.up.items()}
        self.down_n = {_norm(k): v for k, v in self.down.items()}

        a = np.load(ROOT / 'results/signed_regulatory_inputs_v1_20260916/arrays.npz', allow_pickle=True)
        self.mi_names = [str(x) for x in a['rna_names']]
        self.drug_names = [str(x).upper() for x in a['drug_names']]
        rid = pd.read_csv(ROOT / 'research/rna_gene_mapping_20260913/rna_identity.tsv', sep='\t')
        self.mi_src = [rid[(rid.task == 'MiDrug') & (rid.rna_node_index_0based == i)].source_name.iloc[0]
                       for i in range(len(self.mi_names))]

        mti = pd.read_csv(MTI, usecols=['miRNA', 'Target Gene'])
        self.mirna_targets = {}
        for mi, g in zip(mti.miRNA.str.lower(), mti['Target Gene']):
            self.mirna_targets.setdefault(mi, set()).add(str(g).upper())

        drug_genes = {}
        for d in self.drug_names:
            pat = r'\b' + re.escape(_norm(d)) + r'\b'
            gu, gd = set(), set()
            for k, v in self.up_n.items():
                if re.search(pat, k):
                    gu |= v
            for k, v in self.down_n.items():
                if re.search(pat, k):
                    gd |= v
            if gu or gd:
                drug_genes[d] = (gu, gd)
        keys = list(drug_genes)
        if permute:
            rng = np.random.default_rng(seed)
            perm = list(rng.permutation(keys))
            drug_genes = {k: drug_genes[p] for k, p in zip(keys, perm)}
        self.drug_genes = drug_genes

    def features(self, pairs):
        rows = []
        for r, d in zip(pairs[0], pairs[1]):
            tg = self.mirna_targets.get(str(self.mi_src[r]).lower(), set())
            pair = self.drug_genes.get(self.drug_names[d])
            if not tg or pair is None:
                rows.append([np.nan] * 6)
                continue
            gu, gd = pair
            iu, idn = tg & gu, tg & gd
            rows.append([len(iu), len(idn), len(iu) - len(idn),
                         len(iu) / len(tg), len(idn) / len(tg), len(tg)])
        return np.asarray(rows, dtype=np.float32)
