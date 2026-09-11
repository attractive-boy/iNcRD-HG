# Drug-side information channel: target overlap and ATC class.
#
# Every non-degree feature in `gbdt.py` that touches the drug axis reads exactly
# one matrix -- `drugSimMat.csv`, Tanimoto over Morgan fingerprints, built by
# `drug_similarity.py`.  `simviews.py` added ChemBERTa cosine as a second view,
# but that is still a function of the molecular structure: two drugs with
# different scaffolds and the same protein target score as dissimilar under both.
# Drug response is mediated by targets, not by fingerprints, so target overlap
# is a genuinely orthogonal channel rather than a re-weighting of the existing
# one.
#
# It is also the only channel that is symmetric across the two tasks by
# construction: LncDrug and MiDrug share the same 216 drug nodes, so anything
# that improves the drug-side representation lands on both.  Every mechanism in
# `RESULTS.md` that helped one task only was an ncRNA-side change, and the
# density asymmetry (0.96 vs 8.9 training drug edges per ncRNA) is an ncRNA-side
# property.  This route does not touch it.
#
# Identifier resolution, most exact first, because a wrong drug mapping is a
# silent corruption of the similarity matrix:
#
#   1. PubChem CID -> standard InChIKey -> ChEMBL molecule by InChIKey.
#      Structure identity on both sides; no name matching involved.
#   2. ChEMBL exact preferred-name match (`pref_name__iexact`).
#   3. ChEMBL free-text search, first hit, RECORDED AS SUCH so the downstream
#      matrices can be rebuilt without the fuzzy tier.
#
# Nothing here reads any association edge, so none of it sits inside the
# K-fold guard: the target and ATC annotations are node attributes, not
# functions of `A`.
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

PUBCHEM = 'https://pubchem.ncbi.nlm.nih.gov/rest/pug'
CHEMBL = 'https://www.ebi.ac.uk/chembl/api/data'
UA = {'User-Agent': 'incrd-hg/1.0', 'Accept': 'application/json'}


def _get(url, tries=4, timeout=30):
    """GET with backoff.  Both services rate-limit rather than fail hard."""
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            last = e
        except Exception as e:                       # timeouts, resets
            last = e
        time.sleep(1.5 * (i + 1))
    print(f"  ! give up: {url.split('?')[0]} ({type(last).__name__})", file=sys.stderr)
    return None


def inchikeys(cids, chunk=100):
    """PubChem CID -> standard InChIKey.  Batched; the endpoint takes a list."""
    out = {}
    cids = [int(c) for c in cids]
    for i in range(0, len(cids), chunk):
        part = cids[i:i + chunk]
        j = _get(f"{PUBCHEM}/compound/cid/{','.join(map(str, part))}/property/InChIKey/JSON")
        for rec in (j or {}).get('PropertyTable', {}).get('Properties', []):
            if rec.get('InChIKey'):
                out[int(rec['CID'])] = rec['InChIKey']
        print(f"  inchikey {min(i + chunk, len(cids))}/{len(cids)}")
    return out


def chembl_by_inchikey(keys, chunk=25):
    """Exact structure match.  `__in` takes a comma list."""
    out = {}
    keys = list(keys)
    for i in range(0, len(keys), chunk):
        part = keys[i:i + chunk]
        q = f"{CHEMBL}/molecule.json?molecule_structures__standard_inchi_key__in={','.join(part)}&limit=100"
        j = _get(q)
        for m in (j or {}).get('molecules', []):
            k = (m.get('molecule_structures') or {}).get('standard_inchi_key')
            if k:
                out[k] = m
        print(f"  chembl/inchikey {min(i + chunk, len(keys))}/{len(keys)}")
    return out


def chembl_by_name(name):
    q = urllib.parse.quote(str(name))
    j = _get(f"{CHEMBL}/molecule.json?pref_name__iexact={q}&limit=1")
    for m in (j or {}).get('molecules', []):
        return m, 'pref_name'
    j = _get(f"{CHEMBL}/molecule/search.json?q={q}&limit=1")
    for m in (j or {}).get('molecules', []):
        return m, 'search'
    return None, None


def mechanisms(chembl_ids, chunk=20):
    """molecule -> [(target_chembl_id, action_type, mechanism_of_action)]."""
    out = {}
    ids = list(chembl_ids)
    for i in range(0, len(ids), chunk):
        part = ids[i:i + chunk]
        j = _get(f"{CHEMBL}/mechanism.json?molecule_chembl_id__in={','.join(part)}&limit=1000")
        for m in (j or {}).get('mechanisms', []):
            out.setdefault(m['molecule_chembl_id'], []).append(
                dict(target=m.get('target_chembl_id'),
                     action=m.get('action_type'),
                     moa=m.get('mechanism_of_action')))
        print(f"  mechanism {min(i + chunk, len(ids))}/{len(ids)}")
    return out


def target_components(target_ids, chunk=20):
    """target -> [uniprot accessions].  A ChEMBL target can be a complex, and
    two drugs hitting different subunits of one complex should count as
    overlapping; the accession level is what makes that visible."""
    out = {}
    ids = [t for t in target_ids if t]
    for i in range(0, len(ids), chunk):
        part = ids[i:i + chunk]
        j = _get(f"{CHEMBL}/target.json?target_chembl_id__in={','.join(part)}&limit=1000")
        for t in (j or {}).get('targets', []):
            acc = [c.get('accession') for c in (t.get('target_components') or [])
                   if c.get('accession')]
            out[t['target_chembl_id']] = dict(
                accessions=acc, type=t.get('target_type'),
                name=t.get('pref_name'))
        print(f"  target {min(i + chunk, len(ids))}/{len(ids)}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--net-dir', required=True)
    p.add_argument('--out', required=True, help='JSON cache to write')
    p.add_argument('--refresh', action='store_true')
    a = p.parse_args()

    drugs = pd.read_csv(os.path.join(a.net_dir, 'Drug.csv'))
    print(f"{len(drugs)} drug nodes")

    if os.path.exists(a.out) and not a.refresh:
        cache = json.load(open(a.out))
        print(f"loaded cache {a.out}")
    else:
        cache = {}

    keys = cache.get('inchikey') or {str(k): v for k, v in
                                     inchikeys(drugs['CID']).items()}
    cache['inchikey'] = keys
    print(f"InChIKey: {len(keys)}/{len(drugs)}")

    mols = cache.get('molecule')
    if not mols:
        by_key = chembl_by_inchikey(sorted(set(keys.values())))
        mols, how = {}, {}
        for _, row in drugs.iterrows():
            name, cid = row['Drug_Name'], str(int(row['CID']))
            k = keys.get(cid)
            m = by_key.get(k) if k else None
            tier = 'inchikey' if m else None
            if m is None:
                m, tier = chembl_by_name(name)
            if m is not None:
                mols[name] = dict(chembl_id=m['molecule_chembl_id'],
                                  pref_name=m.get('pref_name'),
                                  atc=m.get('atc_classifications') or [],
                                  max_phase=m.get('max_phase'),
                                  tier=tier)
                how[tier] = how.get(tier, 0) + 1
        print('resolution tiers:', how)
    cache['molecule'] = mols
    print(f"ChEMBL molecule: {len(mols)}/{len(drugs)}")

    ids = sorted({v['chembl_id'] for v in mols.values()})
    mech = cache.get('mechanism') or mechanisms(ids)
    cache['mechanism'] = mech
    tids = sorted({m['target'] for v in mech.values() for m in v if m['target']})
    tgt = cache.get('target') or target_components(tids)
    cache['target'] = tgt

    with open(a.out, 'w') as f:
        json.dump(cache, f, indent=1)

    n_mech = sum(1 for v in mols.values() if mech.get(v['chembl_id']))
    n_atc = sum(1 for v in mols.values() if v['atc'])
    print(f"\nwrote {a.out}")
    print(f"  drugs with >=1 mechanism target : {n_mech}/{len(drugs)} "
          f"({100 * n_mech / len(drugs):.1f}%)")
    print(f"  drugs with >=1 ATC code         : {n_atc}/{len(drugs)} "
          f"({100 * n_atc / len(drugs):.1f}%)")
    print(f"  distinct targets {len(tids)}, distinct uniprot "
          f"{len({x for t in tgt.values() for x in t['accessions']})}")


if __name__ == '__main__':
    main()
