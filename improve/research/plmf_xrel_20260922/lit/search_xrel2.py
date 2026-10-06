"""Targeted prior-art queries (title/abstract-focused) for the external-relation channels."""
import json, ssl, time, urllib.request, urllib.parse
from pathlib import Path
OUT = Path(__file__).resolve().parent
ctx = ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
def get(url, timeout=40):
    req = urllib.request.Request(url, headers={'User-Agent':'Mozilla/5.0 LitScan/1.0 (mailto:lit@example.org)'})
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r: return r.read()
EPMC = {
 'mda_title': 'TITLE:("miRNA-drug" OR "microRNA-drug" OR "miRNA–drug") AND (TITLE:association OR TITLE:resistance OR TITLE:sensitivity OR TITLE:interaction) AND (TITLE:predict* OR ABSTRACT:"predict")',
 'mda_disease_side': '(TITLE:"miRNA-drug" OR TITLE:"microRNA-drug") AND ABSTRACT:disease AND ABSTRACT:(association) AND ABSTRACT:(predict*)',
 'mda_target_gene_side': '(TITLE:"miRNA-drug" OR TITLE:"microRNA-drug") AND ABSTRACT:("target gene" OR miRTarBase OR "gene set" OR "protein")',
 'mda_ic50_side': '(ABSTRACT:"miRNA" OR ABSTRACT:"microRNA") AND ABSTRACT:("IC50" OR "GDSC" OR "drug response profile") AND ABSTRACT:("association prediction" OR "resistance prediction") AND ABSTRACT:("drug")',
 'cerna_resist_pred': '(ABSTRACT:"lncRNA-miRNA" OR ABSTRACT:"ceRNA") AND ABSTRACT:("drug resistance" OR "drug sensitivity") AND (TITLE:predict* OR TITLE:"computational") AND ABSTRACT:("association")',
 'mirna_cluster_feature': '(ABSTRACT:"miRNA cluster" OR ABSTRACT:"genomic cluster" OR ABSTRACT:"polycistronic") AND ABSTRACT:("association prediction" OR "link prediction") AND ABSTRACT:(miRNA)',
 'lincs_ncrna_pred': '(ABSTRACT:"LINCS" OR ABSTRACT:"L1000" OR ABSTRACT:"Connectivity Map") AND (ABSTRACT:miRNA OR ABSTRACT:"non-coding") AND ABSTRACT:(drug) AND ABSTRACT:("association prediction" OR "link prediction" OR "predicting")',
 'ctd_mirna_drug': 'ABSTRACT:("Comparative Toxicogenomics Database" OR CTD) AND ABSTRACT:(miRNA) AND ABSTRACT:(drug) AND ABSTRACT:("association" ) AND ABSTRACT:(predict*)',
}
results = {}
for k,q in EPMC.items():
    url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search?' + urllib.parse.urlencode({'query': q, 'format':'json','pageSize':40,'sort':'P_PDATE_D desc','resultType':'lite'})
    try:
        d=json.loads(get(url)); rows=[]
        for r in d.get('resultList',{}).get('result',[]):
            rows.append({'pmid': r.get('pmid'), 'title': r.get('title'), 'journal': r.get('journalTitle'), 'year': r.get('pubYear')})
        results['epmc:'+k]=rows; print('epmc',k,len(rows),d.get('hitCount'),flush=True)
    except Exception as ex: results['epmc:'+k]={'error':str(ex)}; print('epmc',k,'ERR',ex,flush=True)
    time.sleep(1.0)
json.dump(results, open(OUT/'scan_xrel2.json','w'), indent=1, ensure_ascii=False)
print('DONE')
