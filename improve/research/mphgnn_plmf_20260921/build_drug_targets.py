# -*- coding: utf-8 -*-
"""Drug -> gene (Entrez) targets for the PLMF-MDA MDR/MDS drugs, for MPHGNN's
gene-drug relation.  MPHGNN used DrugBank drug-gene links; DrugBank is not
reachable from this host (HTTP 403), so ChEMBL mechanism-of-action targets are
used instead (same resolution path as improve/drug_target.py):
  SMILES -> RDKit InChIKey -> ChEMBL molecule (exact standard InChIKey)
         -> parent molecule -> mechanisms -> target components (UniProt)
         -> UniProt REST -> Entrez GeneID.
Primary resolution is UniChem (DrugBank ID -> ChEMBL ID, exact cross-reference);
InChIKey is the fallback.  Mechanisms are collected for the molecule AND for every
child salt of its parent (ChEMBL files mechanisms under the marketed salt, e.g.
imatinib mesylate CHEMBL1642 for CHEMBL941).
Fallback for InChIKey misses: PubChem standardises the SMILES and returns its
own InChIKey, which is retried against ChEMBL.  Everything is recorded in
cache/drug_targets.json with the matching tier per drug.
"""
import json, os, sys, time, urllib.request, urllib.parse, ssl
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

HERE = os.path.dirname(os.path.abspath(__file__))
PLMF = '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/new5_repro_20260919/repos/PLMF-MDA-main/data/unpacked2/data'
OUT = os.path.join(HERE, 'cache', 'drug_targets.json')
CHEMBL = 'https://www.ebi.ac.uk/chembl/api/data'
PUBCHEM = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug'
UNIPROT = 'https://rest.uniprot.org/uniprotkb'
CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE
UA = {'User-Agent': 'incrd-hg/1.0', 'Accept': 'application/json'}


def get(url, tries=4, timeout=60, raw=False):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            b = urllib.request.urlopen(req, timeout=timeout, context=CTX).read()
            return b.decode() if raw else json.loads(b)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            last = e
        except Exception as e:
            last = e
        time.sleep(1.5 * (i + 1))
    print('  ! give up', url[:100], type(last).__name__, file=sys.stderr, flush=True)
    return None


def chunks(xs, n):
    for i in range(0, len(xs), n):
        yield xs[i:i + n]


def main():
    dr = pd.concat([pd.read_excel(f'{PLMF}/MDR/drug_ID_R.xlsx'), pd.read_excel(f'{PLMF}/MDS/drug_ID_S.xlsx')])
    dr = dr.drop_duplicates('DrugBank_ID').reset_index(drop=True)
    drugs = {}
    for dbid, smi in zip(dr.DrugBank_ID, dr.SMILES):
        m = Chem.MolFromSmiles(str(smi))
        drugs[dbid] = dict(smiles=str(smi), inchikey=Chem.MolToInchiKey(m) if m is not None else None,
                           chembl=None, parent=None, tier=None, targets=[], genes=[])
    print('drugs', len(drugs), 'with inchikey', sum(d['inchikey'] is not None for d in drugs.values()), flush=True)

    # 0) UniChem: DrugBank (sourceID 2) -> ChEMBL (sourceID 1)
    for dbid, d in drugs.items():
        body = json.dumps({'type': 'sourceID', 'compound': dbid, 'sourceID': 2}).encode()
        try:
            req = urllib.request.Request('https://www.ebi.ac.uk/unichem/api/v1/compounds', data=body,
                                         headers={**UA, 'Content-Type': 'application/json'})
            j = json.loads(urllib.request.urlopen(req, timeout=60, context=CTX).read())
        except Exception as e:
            j = None
        for c in (j or {}).get('compounds', []):
            ids = [s_['compoundId'] for s_ in c.get('sources', []) if s_.get('sourceID') == 1]
            if ids:
                d['chembl'] = ids[0]; d['tier'] = 'unichem'
                break
    print('chembl by unichem', sum(d['chembl'] is not None for d in drugs.values()), flush=True)
    # parent ids for unichem hits
    uids = sorted({d['chembl'] for d in drugs.values() if d['chembl']})
    for part in chunks(uids, 25):
        j = get(f"{CHEMBL}/molecule.json?molecule_chembl_id__in={','.join(part)}&limit=100")
        for mol in (j or {}).get('molecules', []):
            for d in drugs.values():
                if d['chembl'] == mol['molecule_chembl_id']:
                    d['parent'] = (mol.get('molecule_hierarchy') or {}).get('parent_chembl_id') or mol['molecule_chembl_id']
                    d['pref_name'] = mol.get('pref_name')

    # 1) ChEMBL by exact standard InChIKey (for UniChem misses)
    key2db = {}
    for dbid, d in drugs.items():
        if d['inchikey'] and not d['chembl']:
            key2db.setdefault(d['inchikey'], []).append(dbid)
    keys = list(key2db)
    for part in chunks(keys, 25):
        j = get(f"{CHEMBL}/molecule.json?molecule_structures__standard_inchi_key__in={','.join(part)}&limit=100")
        for mol in (j or {}).get('molecules', []):
            k = (mol.get('molecule_structures') or {}).get('standard_inchi_key')
            for dbid in key2db.get(k, []):
                drugs[dbid]['chembl'] = mol['molecule_chembl_id']
                drugs[dbid]['parent'] = (mol.get('molecule_hierarchy') or {}).get('parent_chembl_id') or mol['molecule_chembl_id']
                drugs[dbid]['tier'] = 'inchikey'
                drugs[dbid]['pref_name'] = mol.get('pref_name')
    print('chembl by inchikey', sum(d['chembl'] is not None for d in drugs.values()), flush=True)

    # 2) misses: PubChem standardised InChIKey (+ connectivity-layer match as last resort)
    for dbid, d in drugs.items():
        if d['chembl']:
            continue
        j = get(f"{PUBCHEM}/compound/smiles/{urllib.parse.quote(d['smiles'], safe='')}/property/InChIKey,Title/JSON")
        props = ((j or {}).get('PropertyTable') or {}).get('Properties') or []
        if not props:
            continue
        pk = props[0].get('InChIKey'); d['pubchem_cid'] = props[0].get('CID'); d['pubchem_title'] = props[0].get('Title')
        if pk and pk != d['inchikey']:
            j = get(f"{CHEMBL}/molecule.json?molecule_structures__standard_inchi_key__in={pk}&limit=5")
            mols = (j or {}).get('molecules', [])
            if mols:
                mol = mols[0]
                d['chembl'] = mol['molecule_chembl_id']
                d['parent'] = (mol.get('molecule_hierarchy') or {}).get('parent_chembl_id') or mol['molecule_chembl_id']
                d['tier'] = 'pubchem_inchikey'; d['pref_name'] = mol.get('pref_name')
                continue
        if pk:  # connectivity layer only (first 14 chars): stereo/salt-insensitive
            j = get(f"{CHEMBL}/molecule.json?molecule_structures__standard_inchi_key__startswith={pk[:14]}&limit=5")
            mols = (j or {}).get('molecules', [])
            if mols:
                mol = mols[0]
                d['chembl'] = mol['molecule_chembl_id']
                d['parent'] = (mol.get('molecule_hierarchy') or {}).get('parent_chembl_id') or mol['molecule_chembl_id']
                d['tier'] = 'inchikey_connectivity'; d['pref_name'] = mol.get('pref_name')
                continue
        if d.get('pubchem_title'):
            j = get(f"{CHEMBL}/molecule.json?pref_name__iexact={urllib.parse.quote(d['pubchem_title'])}&limit=1")
            mols = (j or {}).get('molecules', [])
            if mols:
                mol = mols[0]
                d['chembl'] = mol['molecule_chembl_id']
                d['parent'] = (mol.get('molecule_hierarchy') or {}).get('parent_chembl_id') or mol['molecule_chembl_id']
                d['tier'] = 'pref_name'; d['pref_name'] = mol.get('pref_name')
    print('chembl total', sum(d['chembl'] is not None for d in drugs.values()),
          {t: sum(d['tier'] == t for d in drugs.values()) for t in ('unichem', 'inchikey', 'pubchem_inchikey', 'inchikey_connectivity', 'pref_name')}, flush=True)

    # 3) mechanisms for molecule and parent ids
    ids = sorted({x for d in drugs.values() for x in (d['chembl'], d['parent']) if x})
    mech = {}   # keyed by molecule id AND parent id
    for part in chunks(ids, 40):
        for field in ('molecule_chembl_id__in', 'parent_molecule_chembl_id__in'):
            j = get(f"{CHEMBL}/mechanism.json?{field}={','.join(part)}&limit=1000")
            for m in (j or {}).get('mechanisms', []):
                if m.get('target_chembl_id'):
                    mech.setdefault(m['molecule_chembl_id'], set()).add(m['target_chembl_id'])
                    mech.setdefault(m.get('parent_molecule_chembl_id') or m['molecule_chembl_id'], set()).add(m['target_chembl_id'])
    for dbid, d in drugs.items():
        t = set()
        for x in (d['chembl'], d['parent']):
            t |= mech.get(x, set())
        d['targets'] = sorted(t)
    print('drugs with >=1 mechanism target', sum(bool(d['targets']) for d in drugs.values()), flush=True)

    # 4) targets -> UniProt accessions (human, protein targets incl. complexes/families)
    tids = sorted({t for d in drugs.values() for t in d['targets']})
    tacc = {}
    for part in chunks(tids, 40):
        j = get(f"{CHEMBL}/target.json?target_chembl_id__in={','.join(part)}&limit=1000")
        for t in (j or {}).get('targets', []):
            if t.get('organism') not in (None, 'Homo sapiens'):
                continue
            accs = [c['accession'] for c in t.get('target_components', []) if c.get('component_type') == 'PROTEIN' and c.get('accession')]
            tacc[t['target_chembl_id']] = dict(name=t.get('pref_name'), type=t.get('target_type'), accessions=accs)
    accs = sorted({a for t in tacc.values() for a in t['accessions']})
    print('targets', len(tids), 'human protein targets', len(tacc), 'accessions', len(accs), flush=True)

    # 5) UniProt -> Entrez GeneID
    acc2gene = {}
    for part in chunks(accs, 100):
        txt = get(f"{UNIPROT}/accessions?accessions={','.join(part)}&fields=accession,gene_primary,xref_geneid&format=tsv", raw=True)
        for line in (txt or '').splitlines()[1:]:
            f = line.split('\t')
            if len(f) >= 3:
                gids = [g for g in f[2].strip(';').split(';') if g.strip()]
                acc2gene[f[0]] = dict(symbol=f[1], geneid=[int(g) for g in gids])
    for dbid, d in drugs.items():
        genes = {}
        for t in d['targets']:
            for a in tacc.get(t, {}).get('accessions', []):
                for g in acc2gene.get(a, {}).get('geneid', []):
                    genes[g] = acc2gene[a]['symbol']
        d['genes'] = [dict(entrez=g, symbol=s) for g, s in sorted(genes.items())]
    n_cov = sum(bool(d['genes']) for d in drugs.values())
    print('drugs with >=1 Entrez gene', n_cov, '/', len(drugs), 'edges', sum(len(d['genes']) for d in drugs.values()), flush=True)
    json.dump(dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   source='UniChem DrugBank->ChEMBL; ChEMBL mechanism-of-action targets incl. salt children (REST); UniProt GeneID cross-refs; DrugBank itself unreachable (403)',
                   drugs=drugs, targets=tacc, acc2gene=acc2gene), open(OUT, 'w'), indent=1)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
