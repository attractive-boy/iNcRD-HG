# -*- coding: utf-8 -*-
"""Entrez <-> symbol table from HumanNet's symbol-annotated network file (same
release as the Entrez file, so the two are aligned row by row)."""
import gzip, json, os, ssl, urllib.request, hashlib
import pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
D = '/data1/licheng/incrd-hg/data_external/humannet_20260921'
CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE
symf = os.path.join(D, 'HumanNet-FN.symbol.tsv.gz')
if not os.path.exists(symf):
    u = 'https://www.inetbio.org/humannet/networks/HumanNet-FN.symbol.tsv.gz'
    data = urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'Mozilla/5.0'}), timeout=120, context=CTX).read()
    open(symf, 'wb').write(data)
    open(os.path.join(D, 'SOURCE.txt'), 'a').write(f'\nHumanNet-FN.symbol.tsv.gz  URL: {u}  sha256: {hashlib.sha256(data).hexdigest()}\n')
ent = pd.read_csv(os.path.join(D, 'HumanNet-FN.tsv.gz'), sep='\t', header=None, comment='#')
sym = pd.read_csv(symf, sep='\t', header=None, comment='#')
assert len(ent) == len(sym), (len(ent), len(sym))
m = {}
for e, s in zip(pd.concat([ent[0], ent[1]]), pd.concat([sym[0], sym[1]])):
    m[int(e)] = str(s)
json.dump({str(k): v for k, v in m.items()}, open(os.path.join(HERE, 'cache', 'entrez_symbol.json'), 'w'))
print('entrez->symbol', len(m), 'example', list(m.items())[:3])
