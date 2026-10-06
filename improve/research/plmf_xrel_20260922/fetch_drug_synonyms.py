"""Resolve PLMF-MDA drugs (DrugBank id -> InChIKey from drug_targets.json) to PubChem CID, IUPAC/title, synonyms and CAS numbers
(PubChem PUG REST is reachable). Output data/drug_synonyms.json: {db: {cid, title, synonyms[], cas[]}}."""
import json, ssl, time, urllib.request, urllib.parse, os, re, sys
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
sys.path.insert(0, '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/mphgnn_plmf_20260921')
from plmf_splits import load_dataset
DT = json.load(open('/data1/licheng/incrd-hg/iNcRD-HG/improve/research/mphgnn_plmf_20260921/cache/drug_targets.json'))['drugs']
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'drug_synonyms.json')
done = json.load(open(OUT)) if os.path.exists(OUT) else {}
def get(url, tries=3):
    for t in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'incrd-hg/1.0'}), timeout=40, context=ctx).read()
        except Exception as e:
            err = e; time.sleep(1.5 * (t + 1))
    raise err
ids = {}
for ds in ('MDR', 'MDS'):
    mir, dru, pos, neg = load_dataset(ds)
    for db, smi in zip(dru['DrugBank_ID'].astype(str), dru['SMILES'].astype(str)): ids[db] = smi
for i, (db, smi) in enumerate(sorted(ids.items())):
    if db in done: continue
    rec = {'pref_name': (DT.get(db) or {}).get('pref_name'), 'inchikey': (DT.get(db) or {}).get('inchikey'), 'cid': None, 'title': None, 'synonyms': [], 'cas': []}
    try:
        if rec['inchikey']:
            d = json.loads(get(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/inchikey/{rec["inchikey"]}/cids/JSON'))
            cids = d.get('IdentifierList', {}).get('CID', [])
        else:
            d = json.loads(get('https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/cids/JSON?' + urllib.parse.urlencode({'smiles': smi})))
            cids = d.get('IdentifierList', {}).get('CID', [])
        if cids:
            rec['cid'] = cids[0]
            d = json.loads(get(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cids[0]}/synonyms/JSON'))
            syn = d.get('InformationList', {}).get('Information', [{}])[0].get('Synonym', [])
            rec['synonyms'] = syn[:80]; rec['title'] = syn[0] if syn else None
            rec['cas'] = [s for s in syn if re.fullmatch(r'\d{2,7}-\d{2}-\d', s)]
    except Exception as e:
        rec['error'] = str(e)[:120]
    done[db] = rec; json.dump(done, open(OUT, 'w'), indent=1)
    print(i, db, rec['pref_name'], rec['cid'], rec['title'], rec['cas'][:2], flush=True); time.sleep(0.6)
print('DONE', len(done))
