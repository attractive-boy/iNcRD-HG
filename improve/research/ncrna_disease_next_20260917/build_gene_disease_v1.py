"""Build drug->mechanism-disease sets from CTD gene-disease + ChEMBL drug targets.

The drug's mechanism-disease profile is the union of the CTD diseases of its
ChEMBL target genes. Output is a JSON of drug -> sorted disease keys.
"""
import csv
import gzip
import json

import numpy as np

HGNC = '/data1/licheng/incrd-hg/iNcRD-HG/improve/results/tabicl_pathway_inputs_20260913/sources/hgnc.txt.gz'
DICT = '/data1/licheng/incrd-hg/iNcRD-HG/improve/results/tabicl_pathway_context_inputs_20260913/dictionary.json'
CHEMBL = '/data1/licheng/incrd-hg/data_external/functional_priority_20260913/chembl/drug_gene_expanded.npz'
CTD = '/data1/licheng/incrd-hg/data_external/ncrna_disease_20260917/CTD_genes_diseases.tsv.gz'
OUT = '/data1/licheng/incrd-hg/iNcRD-HG/improve/research/ncrna_disease_next_20260917/gene_disease_v1.json'


def norm_disease(name, mesh):
    import re
    if isinstance(mesh, str) and mesh.strip():
        digits = re.sub(r'[^0-9]', '', mesh)
        if digits:
            return 'MESH:' + digits
    return 'NAME:' + re.sub(r'[^a-z0-9]+', ' ', str(name).lower()).strip()


def main():
    ens2entrez = {}
    with gzip.open(HGNC, 'rt') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        for row in reader:
            ens = (row.get('ensembl_gene_id') or '').strip()
            ez = (row.get('entrez_id') or '').strip()
            if ens and ez:
                ens2entrez[ens] = ez

    d = json.load(open(DICT))
    genes = d['genes']
    drug_names = [str(x).upper() for x in d['drug_names']]
    z = np.load(CHEMBL)
    indices, indptr = z['indices'], z['indptr']
    drug_genes = {}
    target_entrez = set()
    for i, name in enumerate(drug_names):
        gs = set()
        for j in indices[indptr[i]:indptr[i + 1]]:
            ez = ens2entrez.get(genes[j])
            if ez:
                gs.add(ez)
        if gs:
            drug_genes[name] = gs
        target_entrez |= gs
    print('drugs with mapped targets', len(drug_genes), 'target entrez', len(target_entrez), flush=True)

    gene_dis = {}
    with gzip.open(CTD, 'rt') as handle:
        for line in handle:
            if line.startswith('#'):
                continue
            p = line.rstrip('\n').split('\t')
            if len(p) != 9:
                continue
            if p[1] not in target_entrez:
                continue
            if p[4] not in ('marker/mechanism', 'therapeutic'):
                continue
            gene_dis.setdefault(p[1], set()).add(norm_disease(p[2], p[3]))
    print('target genes with CTD disease', len(gene_dis), flush=True)

    drug_dis = {}
    for name, ezs in drug_genes.items():
        s = set()
        for ez in ezs:
            s |= gene_dis.get(ez, set())
        if s:
            drug_dis[name] = sorted(s)
    print('drugs with mechanism diseases', len(drug_dis), flush=True)
    json.dump(drug_dis, open(OUT, 'w'))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
