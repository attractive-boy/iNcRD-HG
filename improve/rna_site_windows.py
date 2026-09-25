"""Prepare a bounded miRNA-site pilot with real RefSeq transcript flanks."""
import argparse
from collections import Counter,defaultdict
import csv
from datetime import datetime,timezone
import gzip
from pathlib import Path
import urllib.parse
import urllib.request

from scipy import sparse

from improve import tabular_foundation as base


ROOT=Path('/data1/licheng/incrd-hg')
SITES=ROOT/'data_external/pathway_20260913/mirtarbase/MicroRNA_Target_Sites.csv'
RNA=Path('improve/results/tabicl_reli_inputs_20260912')
REFERENCE=Path('improve/results/tabicl_pathway_context_inputs_20260913')
HGNC=Path('improve/results/tabicl_pathway_inputs_20260913/sources/hgnc.txt.gz')
DRUG_INPUTS=ROOT/'data_external/functional_priority_20260913/chembl'
DEFAULT_OUT=ROOT/'data_external/functional_priority_20260913/mirbind_windows'


def fasta_records(text):
    records={};name=None;parts=[]
    for line in text.splitlines()+['>__END__']:
        line=line.strip()
        if not line:
            continue
        if line.startswith('>'):
            if name is not None:
                if name in records or not parts:
                    raise ValueError('Duplicate or empty FASTA record')
                records[name]=''.join(parts).upper()
            name=line[1:].split()[0];parts=[]
        else:
            if name is None:
                raise ValueError('Expected FASTA header')
            parts.append(line)
    return records


def exact_window(transcript,site,length=50):
    """Require a unique, forward-sense exact match and real centered flanks."""
    transcript,site=transcript.upper().replace('U','T'),site.upper().replace('U','T')
    if not site or len(site)>length or not set(site)<=set('ACGT'):
        return None,'invalid_site'
    pos=transcript.find(site)
    if pos<0:
        return None,'not_found_in_forward_transcript'
    if transcript.find(site,pos+1)>=0:
        return None,'nonunique_site'
    start=pos-(length-len(site))//2;end=start+length
    if start<0 or end>len(transcript):
        return None,'insufficient_real_flanks'
    window=transcript[start:end]
    if not set(window)<=set('ACGT'):
        return None,'ambiguous_flank'
    return dict(target_sequence=window,window_start_0=start,site_start_0=pos,
                site_length=len(site),window_length=length),''


def build(args):
    out=Path(args.out_dir)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'INPUTS.json').exists():
        raise ValueError('Pilot inputs already frozen')
    rmanifest=base.read_json(RNA/'inputs.json');reference=base.read_json(REFERENCE/'INPUTS.json')
    for name in ('MiDrug_sequences.json','LncDrug_sequences.json'):
        if base.digest(RNA/name)!=rmanifest['files'][name]:
            raise ValueError('RNA sequence source changed')
    for path in (SITES,HGNC.resolve()):
        if base.digest(path)!=reference['external_sources'][str(path)]:
            raise ValueError('Auxiliary source changed')
    dictionary=base.read_json(REFERENCE/'dictionary.json')
    if base.digest(REFERENCE/'dictionary.json')!=reference['files']['dictionary.json']:
        raise ValueError('Gene dictionary changed')
    dg_manifest=base.read_json(DRUG_INPUTS/'INPUTS.json')
    if base.digest(DRUG_INPUTS/'drug_gene_expanded.npz')!=dg_manifest['files']['drug_gene_expanded.npz']:
        raise ValueError('Drug target dictionary changed')
    dg=sparse.load_npz(DRUG_INPUTS/'drug_gene_expanded.npz')
    targeted={dictionary['genes'][i] for i in dg.indices}
    rna={r['name']:dict(r,index=i) for i,r in enumerate(base.read_json(RNA/'MiDrug_sequences.json'))}
    genes={}
    with gzip.open(HGNC,'rt') as f:
        for row in csv.DictReader(f,delimiter='\t'):
            if row['status']!='Approved' or row['locus_group']!='protein-coding gene':
                continue
            gene=row['ensembl_gene_id'].split('.')[0]
            accessions=sorted(x for x in row['refseq_accession'].split('|') if x.startswith('NM_'))[:2]
            if gene in targeted and accessions:
                genes[row['entrez_id']]=dict(gene_id=gene,symbol=row['symbol'],refseq=accessions)
    groups=defaultdict(dict);counts=Counter();lengths=Counter();exact50=set();matched_rnas=set()
    with SITES.open(encoding='utf-8-sig') as f:
        for number,row in enumerate(csv.DictReader(f),2):
            if row['Species (miRNA)']!='hsa' or row['Species (Target Gene)']!='hsa':
                continue
            counts['human_human_rows']+=1
            name=row['miRNA'].upper()
            if name not in rna:
                continue
            site=row['Target Site'].strip().upper().replace('U','T')
            counts['project_mirna_rows']+=1;lengths[len(site)]+=1;matched_rnas.add(name)
            if len(site)==50 and set(site)<=set('ACGT'):
                counts['exact50_acgt_rows']+=1;exact50.add(name)
            entrez=row['Target Gene (Entrez ID)'].removesuffix('.0')
            if entrez not in genes or not 12<=len(site)<=50 or not set(site)<=set('ACGT'):
                continue
            sequence=rna[name]['sequence'].upper().replace('U','T')
            if not 18<=len(sequence)<=28 or not set(sequence)<=set('ACGT'):
                continue
            key=(name,site)
            groups[entrez].setdefault(key,dict(rna_name=name,rna_index=rna[name]['index'],
                mirna_accession=rna[name]['accession'],mirna_sequence=sequence,site_sequence=site,
                site_source_line=number,mirtarbase_id=row['miRTarBase ID'],pmid=row['References (PMID)'],
                support_type=row['Support Type']))
    selected=sorted(groups,key=lambda key:genes[key]['gene_id'])[:args.genes]
    requested=sorted({acc for key in selected for acc in genes[key]['refseq']})
    spec=dict(genes={key:genes[key] for key in selected},requested_accessions=requested,
        selection='First genes by Ensembl ID with project miRNA sites and expanded drug-target evidence; up to two NM accessions per gene; no RNA-drug labels',
        max_sites_per_gene=args.sites_per_gene,window_length=50,
        sequence_policy='Exact unique forward-transcript match, centered 50 nt window with real ACGT flanks; no target padding or truncation')
    if (out/'QUERY.json').exists() and base.read_json(out/'QUERY.json')!=spec:
        raise ValueError('Pilot query changed; use a new output directory')
    base.write_json(out/'QUERY.json',spec)
    url='https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?'+urllib.parse.urlencode(
        dict(db='nuccore',id=','.join(requested),rettype='fasta',retmode='text'))
    path=out/'refseq_transcripts.fasta';meta=out/'REFSEQ_SOURCE.json'
    if path.exists() and meta.exists():
        provenance=base.read_json(meta)
        if provenance['url']!=url or base.digest(path)!=provenance['sha256']:
            raise ValueError('Cached RefSeq response changed')
    else:
        req=urllib.request.Request(url,headers={'User-Agent':'NcRNAFunctionalPilot/1.0'})
        with urllib.request.urlopen(req,timeout=40) as response:
            raw=response.read()
        path.write_bytes(raw)
        base.write_json(meta,dict(url=url,sha256=base.digest(path),
            retrieved_at_utc=datetime.now(timezone.utc).isoformat()))
    transcripts=fasta_records(path.read_text())
    if not transcripts:
        raise ValueError('No RefSeq FASTA records returned')
    if any(name.split('.')[0] not in requested for name in transcripts):
        raise ValueError('Unexpected RefSeq record')
    by_accession={name.split('.')[0]:name for name in transcripts}
    rows=[];mapping=Counter()
    for key in selected:
        hits=0
        for pair,row in sorted(groups[key].items()):
            for accession in genes[key]['refseq']:
                if accession not in by_accession:
                    mapping['missing_reference_accession']+=1;continue
                record=by_accession[accession]
                window,reason=exact_window(transcripts[record],row['site_sequence'])
                if reason:
                    mapping[reason]+=1;continue
                rows.append(dict(row,gene_id=genes[key]['gene_id'],gene_symbol=genes[key]['symbol'],
                    entrez_id=key,refseq_transcript=record,**window))
                hits+=1
                break
            if hits>=args.sites_per_gene:
                break
        counts['genes_with_verified_windows']+=int(hits>0)
    if rows:
        with (out/'sites.tsv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');writer.writeheader();writer.writerows(rows)
    report=dict(counts=dict(counts),project_mirnas_with_sites=len(matched_rnas),
        project_mirnas_with_exact50_acgt=len(exact50),top_site_lengths=lengths.most_common(15),
        requested_genes=len(selected),requested_transcripts=len(requested),returned_transcripts=len(transcripts),
        verified_windows=len(rows),verified_mirnas=len({r['rna_name'] for r in rows}),
        mapping_attempt_outcomes=dict(mapping),target_padding_used=False,site_orientation_verified=True,
        reference_policy='Current NM transcript versions returned by NCBI; exact identity and position retained, not claimed to be original experimental isoforms',
        downstream_evaluation=False)
    base.write_json(out/'COVERAGE.json',report)
    names=['QUERY.json','REFSEQ_SOURCE.json','refseq_transcripts.fasta','COVERAGE.json']+(['sites.tsv'] if rows else [])
    base.write_json(out/'INPUTS.json',dict(files={name:base.digest(out/name) for name in names},labels_read=False,
        code_sha256=base.digest(Path(__file__)),sources={str(path.resolve()):base.digest(path) for path in
            (SITES,HGNC,RNA/'MiDrug_sequences.json',REFERENCE/'dictionary.json',DRUG_INPUTS/'drug_gene_expanded.npz')}))
    print('RNA_SITE_PILOT',report,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',default=str(DEFAULT_OUT))
    parser.add_argument('--genes',type=int,default=16)
    parser.add_argument('--sites-per-gene',type=int,default=4)
    build(parser.parse_args())


if __name__=='__main__':
    main()
