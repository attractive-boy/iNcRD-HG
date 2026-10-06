import csv, json, ssl, time, urllib.request, urllib.parse, re
ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
def get(u):
    return urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'incrd-hg/1.0'}), timeout=30, context=ctx).read()
def count(term):
    for t in range(4):
        try: return int(json.loads(get('https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&retmode=json&rettype=count&term=' + urllib.parse.quote(term)))['esearchresult']['count'])
        except Exception: time.sleep(2 * (t + 1))
    return -1
def pubchem_name(db):
    try:
        d = json.loads(get(f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/sourceid/DrugBank/{db}/synonyms/JSON'))
        syn = d['InformationList']['Information'][0]['Synonym']; return syn[0]
    except Exception as e: return ''
rows = list(csv.DictReader(open('data/pubmed_drug_plmf.csv')))
for r in rows:
    if int(r['n_all']) > 5: continue
    name = r['name']; cand = []
    if name and re.search(r'ANHYDROUS|HYDROCHLORIDE|SODIUM|MESYLATE|CITRATE|TARTRATE|MALEATE|SULFATE|ACETATE|PHOSPHATE|DIHYDRATE|TRIHYDRATE|MALATE|TOSYLATE', name):
        cand.append(name.split()[0])
    pc = pubchem_name(r['drugbank_id'])
    if pc: cand.append(pc)
    best = None
    for c in cand:
        q = f'"{c}"[tiab]'; n = count(q)
        if n > int(r['n_all']): best = (c, n, count(q + ' AND (miRNA[tiab] OR microRNA[tiab] OR miR[tiab])'), count(q + ' AND (resistance[tiab] OR resistant[tiab] OR sensitivity[tiab])'), count(q + ' AND (cancer[tiab] OR tumor[tiab] OR tumour[tiab] OR carcinoma[tiab])')); r.update(dict(name=c, n_all=best[1], n_mirna=best[2], n_resist=best[3], n_cancer=best[4])); break
    print(r['drugbank_id'], name, '->', cand, best, flush=True); time.sleep(0.4)
w = csv.DictWriter(open('data/pubmed_drug_plmf.csv', 'w', newline=''), fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
