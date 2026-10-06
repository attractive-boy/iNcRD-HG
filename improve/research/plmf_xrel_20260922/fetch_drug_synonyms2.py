"""Second pass: DrugBank-registered synonyms from PubChem Substance (sourceid/DrugBank/<id>) + manual overrides; retry errors."""
import json, ssl, time, urllib.request, os, re
ctx = ssl.create_default_context(); ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE
OUT = 'data/drug_synonyms.json'; d = json.load(open(OUT))
MANUAL = {'DB00515': ['cisplatin', 'cis-diamminedichloroplatinum', 'CDDP'], 'DB00526': ['oxaliplatin'], 'DB00958': ['carboplatin'], 'DB01169': ['arsenic trioxide', 'arsenic(III) oxide'],
          'DB00290': ['bleomycin'], 'DB00541': ['vincristine'], 'DB17021': [], 'DB08818': ['hyaluronic acid', 'hyaluronan']}
def get(url, tries=4):
    for t in range(tries):
        try: return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'incrd-hg/1.0'}), timeout=40, context=ctx).read()
        except Exception as e:
            err = e; time.sleep(2 * (t + 1))
    raise err
for i, (db, rec) in enumerate(sorted(d.items())):
    try:
        j = json.loads(get(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/sourceid/DrugBank/{db}/synonyms/JSON'))
        syn = j['InformationList']['Information'][0].get('Synonym', [])
        rec['db_synonyms'] = syn[:60]
        if not rec.get('cid'):
            try:
                jj = json.loads(get(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/sourceid/DrugBank/{db}/cids/JSON'))
                cids = jj['InformationList']['Information'][0].get('CID', [])
                if cids: rec['cid'] = cids[0]
            except Exception: pass
    except Exception as e:
        rec['db_error'] = str(e)[:80]
    if rec.get('error') and rec.get('cid'):
        try:
            j = json.loads(get(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{rec["cid"]}/synonyms/JSON'))
            syn = j['InformationList']['Information'][0].get('Synonym', []); rec['synonyms'] = syn[:80]; rec['title'] = syn[0] if syn else rec.get('title')
            rec['cas'] = [s for s in syn if re.fullmatch(r'\d{2,7}-\d{2}-\d', s)]; rec.pop('error', None)
        except Exception as e: rec['error'] = str(e)[:80]
    rec['manual'] = MANUAL.get(db, [])
    for s in rec.get('db_synonyms', []) + rec['manual']:
        if re.fullmatch(r'\d{2,7}-\d{2}-\d', s) and s not in rec['cas']: rec['cas'].append(s)
    d[db] = rec; print(i, db, rec.get('pref_name'), (rec.get('db_synonyms') or [None])[0], flush=True); time.sleep(0.5)
json.dump(d, open(OUT, 'w'), indent=1); print('DONE')
