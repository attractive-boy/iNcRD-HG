# -*- coding: utf-8 -*-
"""External-relation bank (XRel) for PLMF-MDA warm protocols.
Every column is label-free w.r.t. the miRNA-drug benchmark: it uses (i) external knowledge that is independent of the
benchmark labels and (ii) the SUPPORT positives only (for label-transfer columns). Groups:
  dz      CTD drug-disease (direct evidence: therapeutic / marker-mechanism) x HMDD v4 miRNA-disease -> shared-disease bridge,
          plus drug-side disease-profile kNN label transfer.
  gene    miRTarBase miRNA-target genes x drug gene sets (ChEMBL mechanism targets, LINCS L1000 / GEO / DrugMatrix / CMap
          perturbation signatures) -> target-signature overlap / enrichment, plus gene-set drug kNN and target-profile miRNA kNN transfer.
  cerna   iNcRD-HG lncRNA-miRNA interactions x lncRNA-drug associations -> ceRNA-mediated bridge (lncRNA-drug labels are a different RNA type).
  cluster miRBase v23.1 genomic clusters (precursors within 10 kb, same strand) -> cluster-mate support statistics.
  pgx     GDSC2 drug-response (LN_IC50) profile similarity between drugs -> pharmacological kNN label transfer.
Shuffle control: permute the external profiles across entities (drug side and/or miRNA side), keeping the marginals."""
import os, sys, re, json, gzip, numpy as np, pandas as pd, scipy.sparse as sp
sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/mphgnn_plmf_20260921')
from plmf_splits import load_dataset
HERE = os.path.dirname(os.path.abspath(__file__))
EXT = '/data1/licheng/incrd-hg/data_external'
DT = json.load(open('/data1/licheng/incrd-hg/iNcRD-HG/improve/research/mphgnn_plmf_20260921/cache/drug_targets.json'))['drugs']
SYN_FILE = os.path.join(HERE, 'data', 'drug_synonyms.json')
DROP_COMPOUND = {'DB00515'}   # InChIKey resolves to transplatin (stereo lost in SMILES); keep DrugBank-registered synonyms only
SALT = set('hydrochloride hcl mesylate mesilate sodium anhydrous dihydrate monohydrate trihydrate hydrate citrate tartrate maleate sulfate sulphate acetate phosphate malate tosylate besylate fumarate succinate bromide chloride potassium calcium disodium hemihydrate lactate ditosylate dimesylate dihydrochloride hydrobromide'.split())


def norm(s): return re.sub(r'[^a-z0-9]', '', str(s).lower())
def base_name(s):
    s = re.sub(r'\(.*?\)', ' ', str(s).lower())
    return ' '.join(t for t in re.split(r'[\s,]+', s) if t and t not in SALT)
def stem(n):
    return re.sub(r'-(3p|5p)$', '', str(n).lower().strip())


class DrugNames:
    """DrugBank id -> {normalised synonyms}, {CAS}, CID (PubChem, cached in data/drug_synonyms.json)."""
    def __init__(self, dbs):
        syn = json.load(open(SYN_FILE)) if os.path.exists(SYN_FILE) else {}
        self.syn, self.cas, self.cid = {}, {}, {}
        for db in dbs:
            s = set(); rec = syn.get(db, {}); pn = (DT.get(db) or {}).get('pref_name')
            dbsyn = list(rec.get('db_synonyms', [])) + list(rec.get('manual', []))
            comp = [] if db in DROP_COMPOUND else [rec.get('title')] + list(rec.get('synonyms', []))
            for x in [pn, rec.get('pref_name')] + comp + dbsyn:
                if x and not re.fullmatch(r'(DB\d{5}|\d+|.{0,2})', str(x)):
                    s.add(norm(x)); s.add(norm(base_name(x)))
            s.discard('')
            # CAS: prefer DrugBank-registered synonyms (the InChIKey-derived compound may be a stereo/salt variant, e.g. cisplatin -> transplatin)
            cas_db = {x for x in dbsyn if re.fullmatch(r'\d{2,7}-\d{2}-\d', x)}
            self.syn[db] = s; self.cas[db] = cas_db if dbsyn else set(rec.get('cas', [])); self.cid[db] = rec.get('cid')
    def match_name(self, name):
        n = norm(name); b = norm(base_name(name))
        return [db for db, s in self.syn.items() if n in s or (b and b in s)]


def topk_rows(S, k):
    S = np.array(S, dtype=np.float32); np.fill_diagonal(S, 0)
    if k >= S.shape[1]: return S
    thr = np.sort(S, axis=1)[:, -k][:, None]
    return np.where((S >= thr) & (S > 0), S, 0).astype(np.float32)


class XRel:
    def __init__(self, ds, groups=('dz', 'gene', 'cerna', 'cluster', 'pgx'), shuffle=None, rng=None, k=10):
        self.ds = ds; self.groups = set(groups); self.k = k
        mir, dru, pos, neg = load_dataset(ds)
        self.names = mir['miRNA_Name'].astype(str).str.lower().values; self.stems = np.array([stem(n) for n in self.names])
        self.dbs = dru['DrugBank_ID'].astype(str).values; self.nr, self.nd = len(mir), len(dru)
        self.dn = DrugNames(self.dbs)
        rng = rng or np.random.RandomState(0); shuffle = set(shuffle or [])
        pr = rng.permutation(self.nr); pd_ = rng.permutation(self.nd)
        self.M = {}; self.log = {}
        builders = dict(dz=self._build_dz, gene=self._build_gene, cerna=self._build_cerna, cluster=self._build_cluster, pgx=self._build_pgx, geo=lambda: self._build_geo(True), geoall=lambda: self._build_geo(False), dt=lambda: self._build_dt(True), dtall=lambda: self._build_dt(False))
        for g in [g for g in builders if g in self.groups]:
            cache = os.path.join(HERE, 'cache', f'xrel_{ds}_{g}.npz')
            if os.path.exists(cache):
                z = np.load(cache, allow_pickle=True); self.M.update({k: z[k] for k in z.files})
            else:
                out = builders[g](); self.M.update(out); np.savez_compressed(cache, **out)
        M = self.M
        # ---- shuffle controls (marginal-preserving permutations of external profiles across entities)
        if 'dz' in shuffle: M['Dt'], M['Dm'] = M['Dt'][pd_], M['Dm'][pd_]
        if 'gene' in shuffle:
            M['T'], M['Tf'] = M['T'][pr], M['Tf'][pr]
            for s in self.gene_sources: M['G_' + s] = M['G_' + s][pd_]
        if 'cerna' in shuffle: M['L'] = M['L'][:, pr]; M['LD'] = M['LD'][:, pd_]
        if 'cluster' in shuffle: M['clu'] = M['clu'][pr]
        if 'pgx' in shuffle: M['P'] = M['P'][pd_]
        for g in ('geo', 'geoall', 'dt', 'dtall'):
            if g in shuffle:
                for k in [k for k in M if k.startswith(g + '_') and M[k].ndim == 2]: M[k] = M[k][pr]     # pair matrices: permute the miRNA axis (drug marginals kept)
        # ---- derived similarity graphs (label-free)
        if 'dz' in self.groups:
            Dall = ((M['Dt'] + M['Dm']) > 0).astype(np.float32)
            df = M['H'].sum(0) + Dall.sum(0); self.idf_dz = np.log((self.nr + self.nd + 1) / (df + 1)).astype(np.float32) + 1
            self.Dall = Dall; self.Dw = Dall * self.idf_dz[None, :]; self.Hw = M['H'] * self.idf_dz[None, :]
            self.S_dz = topk_rows(self._cos(self.Dw), k)
        if 'gene' in self.groups:
            Gu = np.zeros_like(M['G_' + self.gene_sources[0]])
            for s in self.gene_sources: Gu = np.maximum(Gu, (M['G_' + s] >= 0.2).astype(np.float32))
            self.Gu = Gu; self.S_gd = topk_rows(self._jacc(Gu), k); self.S_gm = topk_rows(self._cos(M['T']), 2 * k)
        if 'cerna' in self.groups:
            self.S_lm = topk_rows(self._cos(M['L'].T), 2 * k)
        if 'pgx' in self.groups:
            self.S_px = topk_rows(M['P'], k)

    # ---------------------------------------------------------------- builders
    @staticmethod
    def _cos(X):
        X = np.asarray(X, np.float32); n = np.linalg.norm(X, axis=1, keepdims=True); Xn = X / np.maximum(n, 1e-8); return Xn @ Xn.T
    @staticmethod
    def _jacc(X):
        X = (np.asarray(X) > 0).astype(np.float32); inter = X @ X.T; s = X.sum(1); return inter / np.maximum(s[:, None] + s[None, :] - inter, 1)

    def _build_dz(self):
        ctd = pd.read_csv(f'{EXT}/ncrna_disease_20260917/CTD_chemicals_diseases.tsv.gz', sep='\t', comment='#', header=None, dtype=str,
                          names=['chem', 'cid', 'cas', 'dis', 'did', 'ev', 'gene', 'score', 'omim', 'pmids'], usecols=[0, 2, 3, 5])
        ctd = ctd[ctd['ev'].notna()]
        h = pd.read_csv(f'{EXT}/ncrna_disease_20260917/hmdd_v4_alldata.txt', sep='\t', encoding='latin1', on_bad_lines='skip')
        h['m'] = h['miRNA'].astype(str).str.lower().str.strip(); h['dz'] = h['disease'].astype(str).str.strip()
        dzs = sorted(set(h['dz']) | set(ctd['dis'])); di = {d: i for i, d in enumerate(dzs)}
        H = np.zeros((self.nr, len(dzs)), np.float32); st = {}
        for i, s in enumerate(self.stems): st.setdefault(s, []).append(i)
        for m, dz in zip(h['m'], h['dz']):
            for i in st.get(m, []): H[i, di[dz]] = 1
        # chemical -> drug ids (name / CAS)
        chem2db = {}
        cas2db = {}
        for db, cs in self.dn.cas.items():
            for c in cs: cas2db.setdefault(c, set()).add(db)
        for chem, cas in ctd[['chem', 'cas']].drop_duplicates().itertuples(index=False):
            hits = set(self.dn.match_name(chem)) | cas2db.get(str(cas), set())
            if hits: chem2db[chem] = hits
        Dt = np.zeros((self.nd, len(dzs)), np.float32); Dm = np.zeros_like(Dt); dbi = {db: j for j, db in enumerate(self.dbs)}
        for chem, dis, ev in ctd[['chem', 'dis', 'ev']].itertuples(index=False):
            for db in chem2db.get(chem, ()):
                (Dt if 'therapeutic' in ev else Dm)[dbi[db], di[dis]] = 1
                if 'therapeutic' in ev and 'marker' in ev: Dm[dbi[db], di[dis]] = 1
        self.log = getattr(self, 'log', {}); self.log['dz'] = dict(n_dz=len(dzs), drugs_matched=int(((Dt + Dm).sum(1) > 0).sum()), mirna_cov=float((H.sum(1) > 0).mean()))
        print('[xrel dz]', self.log['dz'], flush=True)
        return dict(H=H, Dt=Dt, Dm=Dm)

    gene_sources = ['chembl', 'lincs', 'geo', 'dmx', 'cmap']

    def _build_gene(self):
        mt = pd.read_csv(f'{EXT}/pathway_20260913/mirtarbase/hsa_MTI.csv', usecols=['miRNA', 'Target Gene', 'Support Type'])
        mt['m'] = mt['miRNA'].astype(str).str.lower(); mt['g'] = mt['Target Gene'].astype(str).str.upper()
        genes = {}
        def gid(g): return genes.setdefault(g, len(genes))
        sets = {s: {} for s in self.gene_sources}          # src -> db -> {gene: count}, nsig[src][db]
        nsig = {s: {} for s in self.gene_sources}
        def add(src, db, gs):
            d = sets[src].setdefault(db, {}); nsig[src][db] = nsig[src].get(db, 0) + 1
            for g in gs: d[g] = d.get(g, 0) + 1
        for db in self.dbs:
            gs = [x['symbol'].upper() for x in (DT.get(db) or {}).get('genes', []) if x.get('symbol')]
            if gs: add('chembl', db, gs)
        for f in ('LINCS_L1000_Chem_Pert_down.gmt', 'LINCS_L1000_Chem_Pert_up.gmt'):
            if not os.path.exists(f'{EXT}/perturbation_20260917/{f}'): continue
            for line in open(f'{EXT}/perturbation_20260917/{f}'):
                p = line.rstrip('\n').split('\t'); t = p[0]
                m = re.match(r'^\S+ \S+ \d+H-(.+)-[0-9.]+$', t)
                if not m: continue
                for db in self.dn.match_name(m.group(1)): add('lincs', db, [g.upper() for g in p[2:] if g])
        cid2db = {str(c): db for db, c in self.dn.cid.items() if c}
        for f in ('Drug_Perturbations_from_GEO_up.gmt', 'Drug_Perturbations_from_GEO_down.gmt'):
            if not os.path.exists(f'{EXT}/perturbation_20260917/{f}'): continue
            for line in open(f'{EXT}/perturbation_20260917/{f}'):
                p = line.rstrip('\n').split('\t'); t = p[0]; hits = set()
                m = re.search(r'\b(DB\d{5})\b', t)
                if m and m.group(1) in set(self.dbs): hits.add(m.group(1))
                m2 = re.search(r'\b(\d{3,9})\b human', t)
                if m2 and m2.group(1) in cid2db: hits.add(cid2db[m2.group(1)])
                if not hits: hits = set(self.dn.match_name(re.split(r'[,\s]\s*(?:DB\d{5}|\d{3,9}|EC50|human)', t)[0]))
                for db in hits: add('geo', db, [g.upper() for g in p[2:] if g])
        for line in open(f'{EXT}/perturbation_20260917/DrugMatrix.gmt'):
            p = line.rstrip('\n').split('\t'); name = re.split(r'-\d', p[0])[0]
            for db in self.dn.match_name(name): add('dmx', db, [g.upper() for g in p[2:] if g])
        for f in ('OldCMAP_up.gmt', 'OldCMAP_down.gmt'):
            if not os.path.exists(f'{EXT}/ncrna_disease_20260917/perturbation_gmt/{f}'): continue
            for line in open(f'{EXT}/ncrna_disease_20260917/perturbation_gmt/{f}'):
                p = line.rstrip('\n').split('\t'); name = p[0].rsplit('-', 1)[0]
                for db in self.dn.match_name(name): add('cmap', db, [g.upper() for g in p[2:] if g])
        for g in mt['g'].unique(): gid(g)
        for s in self.gene_sources:
            for d in sets[s].values():
                for g in d: gid(g)
        ng = len(genes); ni = {n: i for i, n in enumerate(self.names)}; sti = {}
        for i, s in enumerate(self.stems): sti.setdefault(s, []).append(i)
        T = np.zeros((self.nr, ng), np.float32); Tf = np.zeros_like(T)
        for m, g, sup in mt[['m', 'g', 'Support Type']].itertuples(index=False):
            rows = [ni[m]] if m in ni else sti.get(stem(m), []) if '-' in m and m.endswith(('-3p', '-5p')) and False else ([ni[m]] if m in ni else [])
            for i in rows:
                T[i, genes[g]] = 1
                if sup == 'Functional MTI': Tf[i, genes[g]] = 1
        out = dict(T=T, Tf=Tf, genes=np.array(list(genes)))
        dbi = {db: j for j, db in enumerate(self.dbs)}
        for s in self.gene_sources:
            G = np.zeros((self.nd, ng), np.float32)
            for db, d in sets[s].items():
                for g, c in d.items(): G[dbi[db], genes[g]] = c / nsig[s][db]
            out['G_' + s] = G
        self.log = getattr(self, 'log', {}); self.log['gene'] = dict(ng=ng, mirna_cov=float((T.sum(1) > 0).mean()), **{s: int(len(sets[s])) for s in self.gene_sources})
        print('[xrel gene]', self.log['gene'], flush=True)
        return out

    def _build_cerna(self):
        lm = pd.read_csv('/data1/licheng/incrd-hg/incrd-run/2.construct_network/LncMiRNA_interaction.csv'); ld = pd.read_csv('/data1/licheng/incrd-hg/incrd-run/2.construct_network/LncDrug_edge.csv')
        lncs = sorted(set(lm['lncRNA'].astype(str)) | set(ld['ncRNA_Name'].astype(str))); li = {l: i for i, l in enumerate(lncs)}
        ni = {n: i for i, n in enumerate(self.names)}; dbi = {db: j for j, db in enumerate(self.dbs)}
        L = np.zeros((len(lncs), self.nr), np.float32); LD = np.zeros((len(lncs), self.nd), np.float32)
        for l, m in zip(lm['lncRNA'].astype(str), lm['miRNA'].astype(str).str.lower()):
            if m in ni: L[li[l], ni[m]] = 1
        for l, db in zip(ld['ncRNA_Name'].astype(str), ld['DrugBank_ID'].astype(str)):
            if db in dbi: LD[li[l], dbi[db]] = 1
        self.log = getattr(self, 'log', {}); self.log['cerna'] = dict(n_lnc=len(lncs), mirna_cov=float((L.sum(0) > 0).mean()), drug_cov=float((LD.sum(0) > 0).mean()))
        print('[xrel cerna]', self.log['cerna'], flush=True)
        return dict(L=L, LD=LD)

    def _build_cluster(self, window=10000):
        pre = []; mat2pre = {}
        for l in open(f'{EXT}/mirbase_20260922/hsa.gff3'):
            if l.startswith('#'): continue
            r = l.rstrip('\n').split('\t'); a = dict(kv.split('=') for kv in r[8].split(';'))
            if r[2] == 'miRNA_primary_transcript': pre.append((a['ID'], r[0], r[6], int(r[3]), int(r[4]), a['Name'].lower()))
            else: mat2pre.setdefault(a['Name'].lower(), []).append(a['Derives_from'])
        pre_df = pd.DataFrame(pre, columns=['id', 'chr', 'strand', 'start', 'end', 'name']).sort_values(['chr', 'strand', 'start'])
        cl = {}; cur = -1; last = None
        for row in pre_df.itertuples(index=False):
            if last is None or row.chr != last.chr or row.strand != last.strand or row.start - last.end > window: cur += 1
            cl[row.id] = cur; last = row
        by_name = {}
        for row in pre_df.itertuples(index=False):
            by_name.setdefault(row.name, []).append(row.id); by_name.setdefault(re.sub(r'-\d+$', '', row.name), []).append(row.id)
        clu = np.full(self.nr, -1, np.int64); nxt = cur + 1
        for i, (n, s) in enumerate(zip(self.names, self.stems)):
            pids = mat2pre.get(n) or mat2pre.get(s + '-3p') or mat2pre.get(s + '-5p') or by_name.get(s.replace('mir', 'mir').replace('hsa-mir', 'hsa-mir')) or by_name.get(s.replace('-mir-', '-mir-'))
            if not pids:
                pids = by_name.get(re.sub(r'hsa-mir', 'hsa-mir', s)) or by_name.get(s.replace('hsa-mir-', 'hsa-mir-'))
            if pids: clu[i] = cl[pids[0]]
            else: clu[i] = nxt; nxt += 1
        mapped = (clu <= cur)
        sizes = pd.Series(clu[mapped]).value_counts()
        self.log = getattr(self, 'log', {}); self.log['cluster'] = dict(mirna_mapped=float(mapped.mean()), mirna_in_multi=float(np.isin(clu, sizes[sizes > 1].index).mean()))
        print('[xrel cluster]', self.log['cluster'], flush=True)
        return dict(clu=clu)

    def _build_pgx(self):
        f = f'{EXT}/drug_response_20260916/gdsc/GDSC2_fitted_dose_response_27Oct23.xlsx'
        P = np.zeros((self.nd, self.nd), np.float32); ok = False
        try:
            g = pd.read_excel(f, usecols=['DRUG_NAME', 'SANGER_MODEL_ID', 'LN_IC50'])
            g['db'] = None; name2db = {}
            for nm in g['DRUG_NAME'].unique():
                hits = self.dn.match_name(nm)
                if hits: name2db[nm] = hits[0]
            g = g[g['DRUG_NAME'].isin(name2db)]; g['db'] = g['DRUG_NAME'].map(name2db)
            M = g.groupby(['db', 'SANGER_MODEL_ID'])['LN_IC50'].mean().unstack()
            Z = M.sub(M.mean(1), axis=0).div(M.std(1) + 1e-6, axis=0)
            C = Z.T.corr(method='spearman', min_periods=100).fillna(0).values
            dbi = {db: j for j, db in enumerate(self.dbs)}; idx = [dbi[d] for d in Z.index]
            P[np.ix_(idx, idx)] = np.maximum(C, 0); np.fill_diagonal(P, 0); ok = True
            self.log = getattr(self, 'log', {}); self.log['pgx'] = dict(drugs_matched=len(idx), cell_lines=int(M.shape[1]))
        except Exception as e:
            self.log = getattr(self, 'log', {}); self.log['pgx'] = dict(error=str(e)[:80])
        print('[xrel pgx]', self.log['pgx'], flush=True)
        return dict(P=P)

    def _build_geo(self, disjoint=True):
        """GEO omics evidence from ncRNADrug (DR_GEO: differential expression resistant-vs-sensitive; DT_GEO: drug-treated vs untreated).
        disjoint=True keeps only GEO series whose PubMed ids are NOT cited by any ncRNADrug curated record (label-free, study-disjoint)."""
        import json
        ND = f'{EXT}/ncrnadrug_20260922'; tag = 'geo' if disjoint else 'geoall'
        gp = json.load(open(f'{ND}/gse_pmid.json'))
        cur = set()
        for f in ('DR_Curated.txt', 'DT_Curated.txt'):
            c = pd.read_csv(f'{ND}/{f}', sep='\t', dtype=str, encoding='latin1', on_bad_lines='skip', quoting=3, usecols=lambda x: x == 'PMID')
            cur |= set(pd.to_numeric(c['PMID'], errors='coerce').dropna().astype(int).astype(str))
        tied = {g for g, v in gp.items() if any(p in cur for p in v.get('pmids', []))}
        eff = 'resistant' if self.ds == 'MDR' else 'sensitive'
        ni = {n: i for i, n in enumerate(self.names)}; sti = {}
        for i, st in enumerate(self.stems): sti.setdefault(st, []).append(i)
        dbi = {db: j for j, db in enumerate(self.dbs)}
        def rows_of(name):
            n = name.lower().strip()
            return [ni[n]] if n in ni else sti.get(stem(n), [])
        out = {k: np.zeros((self.nr, self.nd), np.float32) for k in ('dr_same', 'dr_opp', 'dr_lp', 'dt_n', 'dt_sig')}
        d = pd.read_csv(f'{ND}/DR_GEO.txt', sep='\t', dtype=str, encoding='latin1', on_bad_lines='skip', quoting=3)
        d = d[(d['ncRNA_Type'] == 'miRNA') & d['DrugBank_ID'].isin(dbi)]
        if disjoint: d = d[~d['GSE_Number'].isin(tied)]
        d['p'] = pd.to_numeric(d['PValue'], errors='coerce')
        seen = set()
        for gse, name, db, pat, p in d[['GSE_Number', 'ncRNA_Name', 'DrugBank_ID', 'Pattern', 'p']].itertuples(index=False):
            for i in rows_of(name):
                j = dbi[db]; key = (gse, i, j, pat)
                if key in seen: continue
                seen.add(key)
                if str(pat).lower() == eff: out['dr_same'][i, j] += 1; out['dr_lp'][i, j] = max(out['dr_lp'][i, j], -np.log10(max(p, 1e-10)) if np.isfinite(p) else 0)
                else: out['dr_opp'][i, j] += 1
        t = pd.read_csv(f'{ND}/DT_GEO.txt', sep='\t', dtype=str, encoding='latin1', on_bad_lines='skip', quoting=3)
        t = t[(t['ncRNA_Type'] == 'miRNA') & t['DrugBank_ID'].isin(dbi)]
        if disjoint: t = t[~t['GSE_Number'].isin(tied)]
        t['p'] = pd.to_numeric(t['PValue'], errors='coerce'); seen = set()
        for gse, name, db, p in t[['GSE_Number', 'ncRNA_Name', 'DrugBank_ID', 'p']].itertuples(index=False):
            for i in rows_of(name):
                j = dbi[db]; key = (gse, i, j)
                if key in seen: continue
                seen.add(key); out['dt_n'][i, j] += 1
                if np.isfinite(p) and p < 0.05: out['dt_sig'][i, j] += 1
        cov_d = ((out['dr_same'] + out['dr_opp'] + out['dt_n']).sum(0) > 0).astype(np.float32)
        res = {f'{tag}_{k}': v for k, v in out.items()}; res[f'{tag}_cov_d'] = cov_d
        self.log = getattr(self, 'log', {}); self.log[tag] = dict(n_gse_dr=int(d['GSE_Number'].nunique()), n_gse_dt=int(t['GSE_Number'].nunique()), tied_excluded=len(tied) if disjoint else 0, drug_cov=float(cov_d.mean()), pairs_dr=int(((out['dr_same'] + out['dr_opp']) > 0).sum()), pairs_dt=int((out['dt_n'] > 0).sum()))
        print(f'[xrel {tag}]', self.log[tag], flush=True)
        return res

    def _build_dt(self, disjoint=True):
        """Curated drug-perturbs-miRNA records (ncRNADrug DT_Curated: drug treatment changes miRNA expression; a different relation
        curated from a largely different literature). disjoint=True drops records whose PMID also carries a curated drug-response record."""
        ND = f'{EXT}/ncrnadrug_20260922'; tag = 'dt' if disjoint else 'dtall'
        dr = pd.read_csv(f'{ND}/DR_Curated.txt', sep='\t', dtype=str, encoding='latin1', on_bad_lines='skip', quoting=3, usecols=lambda x: x == 'PMID')
        dr_pm = set(pd.to_numeric(dr['PMID'], errors='coerce').dropna().astype(int))
        t = pd.read_csv(f'{ND}/DT_Curated.txt', sep='\t', dtype=str, encoding='latin1', on_bad_lines='skip', quoting=3)
        t = t[t['ncRNA_Type'].str.lower().str.contains('mirna', na=False)]; t = t[t['DrugBank_ID'].isin(set(self.dbs))]
        t['pm'] = pd.to_numeric(t['PMID'], errors='coerce')
        if disjoint: t = t[~t['pm'].isin(dr_pm)]
        ni = {n: i for i, n in enumerate(self.names)}; sti = {}
        for i, st in enumerate(self.stems): sti.setdefault(st, []).append(i)
        dbi = {db: j for j, db in enumerate(self.dbs)}
        out = {k: np.zeros((self.nr, self.nd), np.float32) for k in ('n', 'up', 'down')}; pm = {}
        for name, db, pat, p in t[['ncRNA_Name', 'DrugBank_ID', 'ncRNA_Expression_Pattern', 'pm']].itertuples(index=False):
            n = str(name).lower().strip(); rows = [ni[n]] if n in ni else sti.get(stem(n), [])
            for i in rows:
                j = dbi[db]; out['n'][i, j] += 1
                if 'up' in str(pat).lower(): out['up'][i, j] += 1
                elif 'down' in str(pat).lower(): out['down'][i, j] += 1
                if np.isfinite(p): pm.setdefault((i, j), set()).add(int(p))
        npm = np.zeros((self.nr, self.nd), np.float32)
        for (i, j), v in pm.items(): npm[i, j] = len(v)
        cov_d = (out['n'].sum(0) > 0).astype(np.float32)
        res = {f'{tag}_{k}': v for k, v in out.items()}; res[f'{tag}_npm'] = npm; res[f'{tag}_cov_d'] = cov_d
        self.log = getattr(self, 'log', {}); self.log[tag] = dict(records=int(len(t)), pairs=int((out['n'] > 0).sum()), drug_cov=float(cov_d.mean()), mirna_cov=float((out['n'].sum(1) > 0).mean()))
        print(f'[xrel {tag}]', self.log[tag], flush=True)
        return res

    # ---------------------------------------------------------------- features
    def group_cols(self, g, A, r, d, tag):
        G = np.zeros((g.max() + 1, self.nr), np.float32); G[g, np.arange(self.nr)] = 1
        size = G.sum(1)[g] - 1; GA = G @ A; fsum = GA[g[r], d] - A[r, d]; fdeg = (G @ A.sum(1))[g] - A.sum(1)
        return {f'{tag}_size': size[r], f'{tag}_sum': fsum, f'{tag}_mean': fsum / np.maximum(size[r], 1), f'{tag}_deg': fdeg[r], f'{tag}_frac': fsum / np.maximum(fdeg[r], 1)}

    def transform(self, sup, q):
        sup = np.asarray(sup); q = np.asarray(q); r, d = q[:, 0], q[:, 1]; M = self.M
        A = np.zeros((self.nr, self.nd), np.float32); A[sup[:, 0], sup[:, 1]] = 1
        cols = {}
        if 'dz' in self.groups:
            H = M['H']; Dt, Dm = M['Dt'], M['Dm']
            cols['dz_ther'] = (H[r] * Dt[d]).sum(1); cols['dz_mark'] = (H[r] * Dm[d]).sum(1)
            cols['dz_idf'] = (self.Hw[r] * self.Dw[d]).sum(1)
            nh = np.linalg.norm(self.Hw, axis=1); ndw = np.linalg.norm(self.Dw, axis=1)
            cols['dz_cos'] = cols['dz_idf'] / np.maximum(nh[r] * ndw[d], 1e-8)
            cols['dz_nd_d'] = self.Dall.sum(1)[d]; cols['dz_has_d'] = (self.Dall.sum(1)[d] > 0).astype(np.float32)
            SA = A @ self.S_dz.T                    # (m, d) <- sum_d' A[m,d'] S[d,d']
            cols['dz_knn'] = SA[r, d]; cols['dz_knn_norm'] = SA[r, d] / np.maximum(self.S_dz.sum(1)[d], 1e-8)
        if 'gene' in self.groups:
            T, Tf = M['T'], M['Tf']; ng = T.shape[1]; nT = T.sum(1); nTf = Tf.sum(1)
            cols['gene_nT'] = nT[r]; cols['gene_hasT'] = (nT[r] > 0).astype(np.float32)
            for s in self.gene_sources:
                G = (M['G_' + s] >= 0.2).astype(np.float32); nG = G.sum(1)
                ov = (T[r] * G[d]).sum(1); exp = nT[r] * nG[d] / ng
                cols[f'gene_ov_{s}'] = ov; cols[f'gene_lr_{s}'] = np.log((ov + 0.5) / (exp + 0.5))
                cols[f'gene_wov_{s}'] = (T[r] * M['G_' + s][d]).sum(1)
            Gu = self.Gu; nGu = Gu.sum(1)
            ov = (T[r] * Gu[d]).sum(1); ovf = (Tf[r] * Gu[d]).sum(1)
            cols['gene_ov_all'] = ov; cols['gene_lr_all'] = np.log((ov + 0.5) / (nT[r] * nGu[d] / ng + 0.5))
            cols['gene_ovf_all'] = ovf; cols['gene_lrf_all'] = np.log((ovf + 0.5) / (nTf[r] * nGu[d] / ng + 0.5))
            cols['gene_nG_d'] = nGu[d]; cols['gene_hasG_d'] = (nGu[d] > 0).astype(np.float32)
            SA = A @ self.S_gd.T; cols['gene_dknn'] = SA[r, d]; cols['gene_dknn_norm'] = SA[r, d] / np.maximum(self.S_gd.sum(1)[d], 1e-8)
            SmA = self.S_gm @ A; cols['gene_mknn'] = SmA[r, d]; cols['gene_mknn_norm'] = SmA[r, d] / np.maximum(self.S_gm.sum(1)[r], 1e-8)
        if 'cerna' in self.groups:
            L, LD = M['L'], M['LD']; B = L.T @ LD; nLm = L.sum(0); nLd = LD.sum(0)
            cols['cerna_bridge'] = B[r, d]; cols['cerna_norm'] = B[r, d] / np.maximum(nLm[r], 1); cols['cerna_nL_m'] = nLm[r]; cols['cerna_nL_d'] = nLd[d]
            SmA = self.S_lm @ A; cols['cerna_mknn'] = SmA[r, d]; cols['cerna_mknn_norm'] = SmA[r, d] / np.maximum(self.S_lm.sum(1)[r], 1e-8)
        if 'cluster' in self.groups:
            cols.update(self.group_cols(M['clu'], A, r, d, 'clu'))
        if 'pgx' in self.groups:
            SA = A @ self.S_px.T; cols['pgx_knn'] = SA[r, d]; cols['pgx_knn_norm'] = SA[r, d] / np.maximum(self.S_px.sum(1)[d], 1e-8); cols['pgx_has_d'] = (self.S_px.sum(1)[d] > 0).astype(np.float32)
        for g in ('dt', 'dtall'):
            if g in self.groups:
                for k in ('n', 'up', 'down', 'npm'): cols[f'{g}_{k}'] = M[f'{g}_{k}'][r, d]
                cols[f'{g}_has'] = (M[f'{g}_n'][r, d] > 0).astype(np.float32); cols[f'{g}_cov_d'] = M[f'{g}_cov_d'][d]
        for g in ('geo', 'geoall'):
            if g in self.groups:
                for k in ('dr_same', 'dr_opp', 'dr_lp', 'dt_n', 'dt_sig'):
                    cols[f'{g}_{k}'] = M[f'{g}_{k}'][r, d]
                cols[f'{g}_dr_has'] = (M[f'{g}_dr_same'][r, d] + M[f'{g}_dr_opp'][r, d] > 0).astype(np.float32)
                cols[f'{g}_cov_d'] = M[f'{g}_cov_d'][d]
        return pd.DataFrame(cols, dtype=np.float32)


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--dataset', default='MDR'); ap.add_argument('--rebuild', action='store_true')
    ap.add_argument('--groups', default='dz,gene,cerna,cluster,pgx,geo,geoall')
    a = ap.parse_args()
    for g in a.groups.split(','):
        c = os.path.join(HERE, 'cache', f'xrel_{a.dataset}_{g}.npz')
        if a.rebuild and os.path.exists(c): os.remove(c)
    x = XRel(a.dataset, groups=a.groups.split(','))
    mir, dru, pos, neg = load_dataset(a.dataset)
    f = x.transform(pos[:4000], pos[4000:4200])
    print(f.describe().T[['mean', 'std', 'max']].to_string())
    json.dump(x.log, open(os.path.join(HERE, 'logs', f'xrel_build_{a.dataset}_{a.groups.replace(",", "-")}.json'), 'w'), indent=1)
