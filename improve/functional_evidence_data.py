"""Download and freeze human single-protein drug binding evidence from ChEMBL.

Only exact structure/preferred-name molecule mappings are reused. Binding
activity is a distinct evidence channel and is never labelled as inhibition.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

import numpy as np
from scipy import sparse

from improve import tabular_foundation as base


ROOT = Path('/data1/licheng/incrd-hg')
DEFAULT_OUT = ROOT/'data_external/functional_priority_20260913/chembl'
OLD = ROOT/'incrd-run/14.drug/chembl.json'
REFERENCE = Path('improve/results/tabicl_pathway_context_inputs_20260913')
ANNOTATIONS = Path('improve/results/tabicl_pathway_inputs_20260913')
API = 'https://www.ebi.ac.uk/chembl/api/data/'
ACTIVITY_FIELDS = ('activity_id','assay_chembl_id','assay_type','assay_variant_mutation',
    'data_validity_comment','document_chembl_id','document_year','molecule_chembl_id',
    'potential_duplicate','standard_flag','standard_relation','standard_type','standard_units',
    'standard_value','target_chembl_id','target_tax_id')


def read_or_fetch(out, label, url):
    dest = out/'sources'/(label+'.json')
    meta = dest.with_suffix('.meta.json')
    if dest.exists() and meta.exists():
        m = base.read_json(meta)
        if m['url'] != url or base.digest(dest) != m['sha256']:
            raise ValueError('Cached public source changed: '+label)
        return base.read_json(dest)
    error = None
    for attempt in range(3):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'NcRNAFunctionalEvidence/1.0','Accept':'application/json'})
            with urllib.request.urlopen(req,timeout=40) as response:
                raw=response.read();status=response.status
            obj=json.loads(raw)
            dest.parent.mkdir(parents=True,exist_ok=True)
            temp=dest.with_suffix('.partial');temp.write_bytes(raw);temp.replace(dest)
            base.write_json(meta,dict(url=url,http_status=status,sha256=hashlib.sha256(raw).hexdigest(),
                bytes=len(raw),retrieved_at_utc=datetime.now(timezone.utc).isoformat()))
            return obj
        except Exception as exc:
            error=exc
            if attempt < 2:
                time.sleep(2*(attempt+1))
    raise RuntimeError(f'Public source failed: {label}: {error}')


def pages(out, label, endpoint, params, key):
    offset, records, total = 0, [], None
    for page in range(200):
        url=API+endpoint+'.json?'+urllib.parse.urlencode(dict(params,limit=1000,offset=offset))
        obj=read_or_fetch(out,f'{label}_{page:03d}',url)
        if 'page_meta' not in obj or key not in obj:
            raise ValueError('Unexpected ChEMBL response')
        meta=obj['page_meta']
        if total is None:total=int(meta['total_count'])
        if total != int(meta['total_count']):
            raise ValueError('Public query changed size during pagination; refreeze sources')
        current=obj[key];records.extend(current)
        if not meta.get('next'):
            if len(records) != total:raise ValueError('Incomplete pagination')
            print('FETCHED',label,len(records),flush=True)
            return records
        if not current:raise ValueError('Empty nonterminal page')
        offset+=len(current)
    raise ValueError('Pagination safety limit exceeded; do not use partial evidence')


def valid_activity(row):
    try:value=float(row['standard_value'])
    except (ValueError,TypeError,KeyError):return False
    return (row.get('assay_type')=='B' and str(row.get('target_tax_id'))=='9606'
        and row.get('standard_type') in ('IC50','Ki','Kd') and row.get('standard_units')=='nM'
        and row.get('standard_relation')=='=' and 0 < value <= 1000
        and row.get('standard_flag')==1 and row.get('potential_duplicate')==0
        and not row.get('data_validity_comment') and not row.get('assay_variant_mutation'))


def chunks(items,size):
    return [items[i:i+size] for i in range(0,len(items),size)]


def acquire(out,workers):
    old=base.read_json(OLD)
    ids=sorted({v['chembl_id'] for v in old['molecule'].values() if v['tier'] in ('inchikey','pref_name')})
    spec=dict(molecule_mapping_sha256=base.digest(OLD),molecule_ids=ids,
        activity_policy='Human binding assay; exact nM IC50/Ki/Kd >0 and <=1000; standard flag, no validity warning, duplicate or mutant; assay confidence 9; single protein target',
        parameters=dict(assay_type='B',target_tax_id=9606,standard_type__in='IC50,Ki,Kd',
            standard_units='nM',standard_value__gt=0,standard_value__lte=1000,standard_relation='=',
            standard_flag=1,potential_duplicate=0))
    out.mkdir(parents=True,exist_ok=True)
    if (out/'QUERY_SPEC.json').exists() and base.read_json(out/'QUERY_SPEC.json') != spec:
        raise ValueError('Use a new source directory for changed query rules')
    base.write_json(out/'QUERY_SPEC.json',spec)
    status=read_or_fetch(out,'status',API+'status.json')
    def activity_job(item):
        i,group=item
        return pages(out,'activity_'+str(i),'activity',
            dict(spec['parameters'],molecule_chembl_id__in=','.join(group),only=','.join(ACTIVITY_FIELDS)),'activities')
    with ThreadPoolExecutor(max_workers=workers) as pool:
        batches=list(pool.map(activity_job,enumerate(chunks(ids,10))))
    activity=[row for batch in batches for row in batch]
    if len({row['activity_id'] for row in activity}) != len(activity):
        raise ValueError('Duplicate activity IDs across pages; refreeze the query')
    if any(row['molecule_chembl_id'] not in ids for row in activity):
        raise ValueError('Server returned an unrequested molecule')
    kept=[row for row in activity if valid_activity(row)]
    assay_ids=sorted({x['assay_chembl_id'] for x in kept})
    target_ids=sorted({x['target_chembl_id'] for x in kept})
    jobs=[]
    for label,endpoint,field,groups,key,only in [
        ('assay','assay','assay_chembl_id__in',chunks(assay_ids,40),'assays','assay_chembl_id,confidence_score,target_chembl_id,relationship_type'),
        ('target','target','target_chembl_id__in',chunks(target_ids,40),'targets','target_chembl_id,target_type,organism,tax_id,target_components,pref_name')]:
        for i,group in enumerate(groups):jobs.append((label+str(i),endpoint,{field:','.join(group),'only':only},key))
    def lookup_job(job):
        label,endpoint,params,key=job
        return endpoint,pages(out,label,endpoint,params,key)
    assays,targets={},{}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for kind,records in pool.map(lookup_job,jobs):
            dest=assays if kind=='assay' else targets
            field=kind+'_chembl_id'
            dest.update({x[field]:x for x in records})
    if set(assays) != set(assay_ids) or set(targets) != set(target_ids):
        raise ValueError('Missing assay or target metadata')
    base.write_json(out/'retrieved.json',dict(status=status,activities=activity,assays=assays,targets=targets))
    base.write_json(out/'DOWNLOAD_COMPLETE.json',dict(query_spec_sha256=base.digest(out/'QUERY_SPEC.json'),
        retrieved_sha256=base.digest(out/'retrieved.json'),complete=True,
        activities=len(activity),locally_valid=len(kept),assays=len(assays),targets=len(targets),
        sources={str(path.relative_to(out)):base.digest(path) for path in sorted((out/'sources').glob('*.json'))}))


def verify_snapshot(out,reference,annotations):
    complete=base.read_json(out/'DOWNLOAD_COMPLETE.json')
    if (not complete['complete'] or base.digest(out/'retrieved.json') != complete['retrieved_sha256']
        or base.digest(out/'QUERY_SPEC.json') != complete['query_spec_sha256']):
        raise ValueError('Only complete verified public snapshots can be used')
    spec=base.read_json(out/'QUERY_SPEC.json')
    if base.digest(OLD) != spec['molecule_mapping_sha256']:
        raise ValueError('Molecule identity mapping changed')
    for name,expected in complete['sources'].items():
        if base.digest(out/name) != expected:
            raise ValueError('Cached public source changed: '+name)
    manifest=base.read_json(reference/'INPUTS.json')
    if manifest['labels_read']:
        raise ValueError('Reference evidence used RNA-drug labels')
    for name in ('dictionary.json','drug_gene.npz'):
        if base.digest(reference/name) != manifest['files'][name]:
            raise ValueError('Reference input changed: '+name)
    hgnc=annotations/'sources/hgnc.txt.gz'
    if base.digest(hgnc) != manifest['external_sources'][str(hgnc.resolve())]:
        raise ValueError('HGNC gene mapping changed')


def resolve_target(target,acc_to_gene):
    if target.get('target_type')!='SINGLE PROTEIN' or str(target.get('tax_id'))!='9606':
        return '', 'not_human_single_protein'
    accessions={c.get('accession') for c in target.get('target_components',[]) if c.get('accession')}
    if len(accessions)!=1:
        return '', 'ambiguous_or_missing_gene'
    mapped=acc_to_gene.get(next(iter(accessions)),set())
    if len(mapped)!=1:
        return '', 'ambiguous_or_missing_gene'
    return next(iter(mapped)), ''


def build(out,reference,annotations):
    verify_snapshot(out,reference,annotations)
    data=base.read_json(out/'retrieved.json')
    dictionary=base.read_json(reference/'dictionary.json')
    genes={g:i for i,g in enumerate(dictionary['genes'])}
    old=base.read_json(OLD)
    by_molecule=defaultdict(list)
    for i,name in enumerate(dictionary['drug_names']):
        m=old['molecule'].get(name,{})
        if m.get('tier') in ('inchikey','pref_name'):by_molecule[m['chembl_id']].append(i)
    acc_to_gene=defaultdict(set)
    with gzip.open(annotations/'sources/hgnc.txt.gz','rt') as handle:
        for row in csv.DictReader(handle,delimiter='\t'):
            if row.get('status')!='Approved' or row.get('locus_group')!='protein-coding gene':continue
            gene=row.get('ensembl_gene_id','').split('.')[0]
            if gene not in genes:continue
            for acc in row.get('uniprot_ids','').split('|'):
                if acc:acc_to_gene[acc].add(gene)
    accepted=defaultdict(list);counts=Counter()
    for row in data['activities']:
        counts['retrieved_activities']+=1
        if not valid_activity(row):counts['invalid_activity']+=1;continue
        assay=data['assays'][row['assay_chembl_id']]
        if assay.get('confidence_score')!=9 or assay.get('target_chembl_id')!=row['target_chembl_id']:
            counts['not_confidence_9_same_target']+=1;continue
        gene,reason=resolve_target(data['targets'][row['target_chembl_id']],acc_to_gene)
        if reason:counts[reason]+=1;continue
        counts['accepted_activities']+=1
        for idx in by_molecule[row['molecule_chembl_id']]:accepted[idx,gene].append(row)
    rows=[]
    for (idx,gene),records in sorted(accepted.items()):
        rows.append(dict(drug_index=idx,drug_name=dictionary['drug_names'][idx],gene_id=gene,
            evidence_type='measured_binding_activity',direction='unknown',
            median_pactivity=float(np.median([9-np.log10(float(x['standard_value'])) for x in records])),
            activity_ids=';'.join(map(str,sorted({x['activity_id'] for x in records}))),
            chembl_assays=';'.join(sorted({x['assay_chembl_id'] for x in records})),
            chembl_targets=';'.join(sorted({x['target_chembl_id'] for x in records})),
            documents=';'.join(sorted({x['document_chembl_id'] for x in records if x.get('document_chembl_id')})),
            unique_assays=len({x['assay_chembl_id'] for x in records})))
    if not rows:raise ValueError('No valid activity evidence; inspect source audit')
    with (out/'drug_gene_activity.tsv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(rows)
    mat=sparse.csr_matrix((np.ones(len(rows)),([r['drug_index'] for r in rows],
        [genes[r['gene_id']] for r in rows])),shape=(len(dictionary['drug_names']),len(genes)))
    mat.data[:]=1
    sparse.save_npz(out/'drug_gene_activity.npz',mat)
    previous=sparse.load_npz(reference/'drug_gene.npz')
    if previous.shape != mat.shape:
        raise ValueError('Drug/gene dictionary shape mismatch')
    expanded=previous.maximum(mat)
    sparse.save_npz(out/'drug_gene_expanded.npz',expanded)
    old_coverage=np.asarray(previous.sum(1)).ravel()>0;new_coverage=np.asarray(expanded.sum(1)).ravel()>0
    summary=dict(filters=dict(counts),activity_drugs=int((np.asarray(mat.sum(1)).ravel()>0).sum()),
        old_drugs=int(old_coverage.sum()),expanded_drugs=int(new_coverage.sum()),total_drugs=mat.shape[0],
        newly_covered_drugs=[dictionary['drug_names'][i] for i in np.flatnonzero(new_coverage & ~old_coverage)],
        old_edges=int(previous.nnz),activity_edges=int(mat.nnz),expanded_edges=int(expanded.nnz),
        max_activity_targets_per_drug=int(np.asarray(mat.sum(1)).max()),
        no_rna_drug_labels_read=True,drug_action_not_inferred_from_activity=True)
    base.write_json(out/'COVERAGE.json',summary)
    files=['drug_gene_activity.tsv','drug_gene_activity.npz','drug_gene_expanded.npz','COVERAGE.json','QUERY_SPEC.json','DOWNLOAD_COMPLETE.json']
    base.write_json(out/'INPUTS.json',dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        files={name:base.digest(out/name) for name in files},reference_dir=str(reference.resolve()),
        reference_manifest_sha256=base.digest(reference/'INPUTS.json'),
        hgnc_sha256=base.digest(annotations/'sources/hgnc.txt.gz'),dictionary_sha256=base.digest(reference/'dictionary.json'),
        code_sha256=base.digest(Path(__file__)),labels_read=False))
    print('COVERAGE',summary,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('download','build','all'))
    parser.add_argument('--out-dir',default=str(DEFAULT_OUT))
    parser.add_argument('--reference-dir',default=str(REFERENCE))
    parser.add_argument('--annotations-dir',default=str(ANNOTATIONS))
    parser.add_argument('--workers',type=int,default=3)
    args=parser.parse_args();out=Path(args.out_dir)
    if args.command in ('download','all'):acquire(out,args.workers)
    if args.command in ('build','all'):build(out,Path(args.reference_dir),Path(args.annotations_dir))


if __name__=='__main__':
    main()
