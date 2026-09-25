"""Align RNA expression with public gene-effect and drug-viability assays.

No RNA-drug association labels are opened here. Molecular identity matching
keeps stereochemistry, charges and fragments; it does not silently merge salts.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from rdkit import Chem,rdBase

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'results/dependency_data_20260914'
CONTEXT=ROOT/'results/tabicl_pathway_context_inputs_20260913'
NET=ROOT.parents[1]/'incrd-run/2.construct_network'
OUT=ROOT/'results/functional_response_inputs_20260914'


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def canonical(value):
    if not isinstance(value,str) or not value:return ''
    molecule=Chem.MolFromSmiles(value)
    if molecule is None:return ''
    return Chem.MolToSmiles(molecule,canonical=True,isomericSmiles=True)


def build(out):
    out=Path(out)
    if out.exists():raise ValueError('Use a new aligned-input directory')
    ga=json.loads((DATA/'DATA_AUDIT.json').read_text())
    da=json.loads((DATA/'PRISM_AUDIT.json').read_text())
    if digest(DATA/'crispr_22Q2_matrix.npz')!=ga['matrix_sha256'] or digest(DATA/'prism_19q3_matrix.npz')!=da['matrix_sha256']:
        raise ValueError('Converted phenotype matrix changed')
    if digest(DATA/'prism_compounds.csv')!=da['metadata_sha256']:raise ValueError('Compound metadata changed')
    dictionary=json.loads((CONTEXT/'dictionary.json').read_text())
    expression=[];rna_names=[];offsets=[0]
    for task,name in [('MiDrug','MiExp.csv'),('LncDrug','LncExp.csv')]:
        frame=pd.read_csv(NET/name,index_col=0)
        frame.columns=frame.columns.str.upper()
        if frame.index.duplicated().any() or frame.columns.duplicated().any():raise ValueError('Duplicate RNA expression identity')
        frame=frame[dictionary['rna_names'][task]]
        expression.append(frame);rna_names+=list(frame.columns);offsets.append(len(rna_names))
    if set(expression[0].index)!=set(expression[1].index):raise ValueError('RNA expression cell identities differ')
    # These files contain the same ModelIDs in different row orders. Every
    # later selection is by ModelID, never by the original row position.
    expression_reordered=not expression[0].index.equals(expression[1].index)
    with np.load(DATA/'crispr_22Q2_matrix.npz',allow_pickle=False) as z:
        gene_values=z['values'];gene_cells=z['cells'];entrez=z['entrez_ids'];symbols=z['symbols']
    with np.load(DATA/'prism_19q3_matrix.npz',allow_pickle=False) as z:
        drug_values=z['values'];drug_cells=z['cells'];compounds=z['compounds']
    meta=pd.read_csv(DATA/'prism_compounds.csv',keep_default_na=False).set_index('compound').loc[compounds]
    # Preserve one prespecified assay condition. Dose differences elsewhere in
    # the release are not collapsed into an invented common IC50 measurement.
    eligible=(meta.screen_id.to_numpy()=='HTS')&np.isclose(meta.dose.to_numpy(dtype=float),2.5,rtol=0,atol=.001)
    drug_values=drug_values[:,eligible];compounds=compounds[eligible];meta=meta.iloc[np.flatnonzero(eligible)]
    cells=np.array(sorted(set(expression[0].index)&(set(gene_cells)|set(drug_cells))),dtype=str)
    x=np.concatenate([f.loc[cells].to_numpy(dtype=np.float32) for f in expression],axis=1)
    if not np.isfinite(x).all():raise ValueError('Nonfinite RNA expression')
    gi={c:i for i,c in enumerate(gene_cells)};di={c:i for i,c in enumerate(drug_cells)}
    yg=np.full((len(cells),len(entrez)),np.nan,dtype=np.float32)
    yd=np.full((len(cells),len(compounds)),np.nan,dtype=np.float32)
    for i,c in enumerate(cells):
        if c in gi:yg[i]=gene_values[gi[c]]
        if c in di:yd[i]=drug_values[di[c]]
    cell_meta=pd.read_csv(DATA/'metadata_22Q2.csv',keep_default_na=False).set_index('depmap_id')
    if cell_meta.index.duplicated().any():raise ValueError('Duplicate ModelID in metadata')
    lineages=np.array([str(cell_meta.loc[c,'lineage']) or 'Unknown' if c in cell_meta.index else 'Unknown' for c in cells],dtype=str)
    drug_names=dictionary['drug_names']
    source_drugs=pd.read_csv(NET/'Drug.csv',dtype=str).set_index('Drug_Name')
    source_smiles=pd.read_csv(NET/'drug_smiles.csv',dtype=str).set_index('drug_cid')
    cids=source_drugs.loc[drug_names,'CID'].to_numpy(str)
    exact={}
    for i,value in enumerate(meta.smiles):
        key=canonical(value)
        if key:exact.setdefault(key,[]).append(i)
    drug_map=np.zeros((len(drug_names),len(compounds)),dtype=np.float32);mappings=[]
    for i,(name,cid) in enumerate(zip(drug_names,cids)):
        key=canonical(source_smiles.loc[cid,'smiles']);indices=exact.get(key,[]) if key else []
        if indices:drug_map[i,indices]=1/len(indices)
        mappings.append(dict(drug_index=i,drug_name=name,cid=cid,canonical_smiles=key,
                             assay_matches=len(indices),compounds='|'.join(compounds[indices]),
                             assay_names='|'.join(meta.iloc[indices]['name'].tolist())))
    hgnc=ROOT/'results/tabicl_pathway_inputs_20260913/sources/hgnc.txt.gz'
    genes=pd.read_csv(hgnc,sep='\t',dtype=str,keep_default_na=False)
    candidates={}
    for row in genes.loc[genes.status.eq('Approved')].to_dict('records'):
        if row['entrez_id'] and row['ensembl_gene_id']:
            candidates.setdefault(row['entrez_id'],set()).add(row['ensembl_gene_id'])
    lookup={g:i for i,g in enumerate(dictionary['genes'])}
    mapped=np.array([lookup.get(next(iter(candidates[e])),-1) if e in candidates and len(candidates[e])==1 else -1 for e in entrez],dtype=np.int64)
    unique_coordinates,coordinate_counts=np.unique(mapped[mapped>=0],return_counts=True)
    ambiguous_coordinates=unique_coordinates[coordinate_counts>1]
    ambiguous=np.isin(mapped,ambiguous_coordinates)
    mapped[ambiguous]=-1
    crosswalk=pd.DataFrame(dict(entrez_id=entrez,symbol=symbols,dictionary_gene_index=mapped,
                               excluded_ambiguous_coordinate=ambiguous))
    dg=sparse.load_npz(CONTEXT/'drug_gene.npz').tocsr()
    kept=np.flatnonzero(mapped>=0)
    old=dg[:,mapped[kept]].tocoo()
    targets=sparse.csr_matrix((old.data,(old.row,kept[old.col])),shape=(len(drug_names),len(entrez)),dtype=np.float32)
    counts=np.asarray(targets.sum(1)).ravel();targets=sparse.diags(1/np.maximum(counts,1))@targets
    if len(cells)<100 or np.count_nonzero(drug_map.sum(1))<10:raise ValueError('Insufficient biological alignment')
    out.mkdir(parents=True)
    np.savez_compressed(out/'arrays.npz',x=x,gene=yg,drug=yd,cells=cells,lineages=lineages,
                        entrez_ids=entrez,compounds=compounds,rna_names=np.array(rna_names),
                        rna_offsets=np.array(offsets),drug_names=np.array(drug_names))
    sparse.save_npz(out/'drug_assay_map.npz',sparse.csr_matrix(drug_map))
    sparse.save_npz(out/'drug_gene_map.npz',targets.tocsr())
    pd.DataFrame(mappings).to_csv(out/'drug_identity_matches.csv',index=False)
    crosswalk.to_csv(out/'gene_crosswalk.csv',index=False)
    meta.to_csv(out/'assay_conditions.csv')
    paths=[DATA/n for n in ('DATA_AUDIT.json','PRISM_AUDIT.json','crispr_22Q2_matrix.npz','prism_19q3_matrix.npz','prism_compounds.csv','metadata_22Q2.csv')]
    paths += [CONTEXT/'dictionary.json',CONTEXT/'drug_gene.npz',hgnc,*[NET/n for n in ('MiExp.csv','LncExp.csv','Drug.csv','drug_smiles.csv')]]
    record=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),association_labels_read=False,
        cells=len(cells),rna_features=x.shape[1],variable_rna_features=int((x.std(0)>1e-6).sum()),
        gene_effect_cells=int(np.isfinite(yg).any(1).sum()),drug_viability_cells=int(np.isfinite(yd).any(1).sum()),
        cells_with_both=int((np.isfinite(yg).any(1)&np.isfinite(yd).any(1)).sum()),
        genes=len(entrez),assay_compounds=len(compounds),excluded_other_conditions=int((~eligible).sum()),
        matched_drugs=int((drug_map.sum(1)>0).sum()),gene_annotated_drugs=int((counts>0).sum()),
        mapped_genes=int((mapped>=0).sum()),ambiguous_gene_coordinates_excluded=int(ambiguous.sum()),
        expression_rows_aligned_by_modelid=expression_reordered,unmatched_original_cells=len(expression[0])-len(cells),
        lineages={str(g):int((lineages==g).sum()) for g in np.unique(lineages)},
        missing_gene_entries=int(np.isnan(yg).sum()),missing_drug_entries=int(np.isnan(yd).sum()),
        dose_policy='HTS screen with dose within 0.001 of nominal 2.5 uM. All other conditions excluded before fitting.',
        drug_identity='Exact RDKit canonical isomeric SMILES; charges/fragments/stereochemistry preserved. Equal weights only over matched chemical identity in the prespecified dose window.',
        phenotype_semantics='22Q2 continuous CRISPR gene effects and PRISM 19Q3 logfold viability relative to DMSO. These are separate supervised endpoints, neither IC50 nor measured ncRNA intervention effects.',
        gene_target_semantics='Existing unsigned target annotation retained as a separate proxy; gene knockout is not assumed equivalent to drug treatment.',
        rdkit_version=rdBase.rdkitVersion,source_sha256=digest(__file__),input_sha256={str(p):digest(p) for p in paths},
        files={p.name:digest(p) for p in out.iterdir() if p.is_file()})
    (out/'INPUTS.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('input_sha256','files')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out-dir',default=str(OUT))
    build(ap.parse_args().out_dir)
