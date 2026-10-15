"""Original 216 drug structures as internal molecular descriptors, no new entities."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rdkit import Chem,rdBase

HERE=Path(__file__).resolve().parent
CLEAN=HERE.parent/'midrug_mirna_only_20261006'
RAW=HERE.parents[3]/'incrd-orig/2.construct_network'
def sha256(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def categorical(value,choices):return choices.index(value) if value in choices else len(choices)


def main():
    provenance=json.loads((CLEAN/'groups_provenance.json').read_text())['original_source_hashes']
    files=['Drug.csv','drug_smiles.csv','drug_features-chemBertA.csv']
    for f in files:assert sha256(RAW/f)==provenance[f]
    manifest=json.loads((CLEAN/'manifest.json').read_text())
    assert sha256(CLEAN/'graph.pt')==manifest['graph_sha256']
    graph=torch.load(CLEAN/'graph.pt',map_location='cpu')
    assert set(graph.node_types)=={'miRNA','drug'}
    cids=pd.read_csv(RAW/'Drug.csv')['CID'].astype(int).tolist()
    strings=pd.read_csv(RAW/'drug_smiles.csv').set_index('drug_cid').loc[cids,'smiles'].tolist()
    features=pd.read_csv(RAW/'drug_features-chemBertA.csv',index_col=0)
    assert features.index.astype(int).tolist()==cids
    x=features.values.astype(np.float32)
    x=(x-x.min(1,keepdims=True))/(x.max(1,keepdims=True)-x.min(1,keepdims=True)+1e-8)
    np.testing.assert_array_equal(x,graph['drug'].x.numpy())
    assert len(cids)==216 and len(set(cids))==216
    atoms=[];bonds=[];edges=[];batch=[];offsets=[0];canonical=[];fragments=[]
    hybrid=['UNSPECIFIED','S','SP','SP2','SP3','SP3D','SP3D2']
    bondtypes=['SINGLE','DOUBLE','TRIPLE','AROMATIC']
    stereo=['STEREONONE','STEREOANY','STEREOZ','STEREOE','STEREOCIS','STEREOTRANS']
    for d,string in enumerate(strings):
        mol=Chem.MolFromSmiles(string);assert mol is not None and mol.GetNumAtoms()>0
        canonical.append(Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True))
        fragments.append(len(Chem.GetMolFrags(mol)))
        start=len(atoms)
        for a in mol.GetAtoms():
            atoms.append([min(a.GetAtomicNum(),118),min(a.GetTotalDegree(),8),
                min(max(a.GetFormalCharge(),-4),4)+4,min(a.GetTotalNumHs(),4),
                categorical(str(a.GetHybridization()),hybrid),int(a.GetIsAromatic()),
                int(a.IsInRing()),min(int(a.GetChiralTag()),4)])
            batch.append(d)
        for b in mol.GetBonds():
            f=[categorical(str(b.GetBondType()),bondtypes),int(b.GetIsConjugated()),
               int(b.IsInRing()),categorical(str(b.GetStereo()),stereo)]
            u,v=start+b.GetBeginAtomIdx(),start+b.GetEndAtomIdx()
            edges.extend([[u,v],[v,u]]);bonds.extend([f,f])
        offsets.append(len(atoms))
    atoms=np.asarray(atoms,dtype=np.int64);bonds=np.asarray(bonds,dtype=np.int64)
    edges=np.asarray(edges,dtype=np.int64).T;batch=np.asarray(batch,dtype=np.int64)
    assert np.all(atoms>=0) and np.all(atoms<np.asarray([119,9,9,5,8,2,2,5])[None,:])
    assert np.all(bonds>=0) and np.all(bonds<np.asarray([5,2,2,7])[None,:])
    assert np.array_equal(batch[edges[0]],batch[edges[1]])
    assert np.all(np.diff(offsets)>0)
    path=HERE/'molecular_inputs.npz'
    arrays=dict(atom_features=atoms,bond_features=bonds,edge_index=edges,atom_drug=batch,
        atom_offsets=np.asarray(offsets),drug_cids=np.asarray(cids),canonical_smiles=np.asarray(canonical))
    if path.exists():
        with np.load(path) as prior:
            for k,v in arrays.items():np.testing.assert_array_equal(prior[k],v)
    else:np.savez_compressed(path,**arrays)
    report=dict(status='prepared_and_verified',n_drugs=216,n_atoms=len(atoms),n_directed_bonds=edges.shape[1],
        atom_counts=dict(min=int(np.diff(offsets).min()),median=float(np.median(np.diff(offsets))),max=int(np.diff(offsets).max())),
        multicomponent_drugs=sum(n>1 for n in fragments),rdkit_version=rdBase.rdkitVersion,
        atom_fields=['atomic_number','total_degree','formal_charge_plus4','total_hydrogens','hybridization','aromatic','ring','chirality'],
        atom_cardinalities=[119,9,9,5,8,2,2,5],
        bond_fields=['bond_type','conjugated','ring','stereo'],bond_cardinalities=[5,2,2,7],
        hybridization_categories=hybrid+['OTHER'],bond_type_categories=bondtypes+['OTHER'],stereo_categories=stereo+['OTHER'],
        source_files={str(RAW/f):sha256(RAW/f) for f in files},source_sha256=sha256(__file__),
        clean_graph_sha256=sha256(CLEAN/'graph.pt'),molecular_inputs_sha256=sha256(path),
        original_drug_order_verified=True,original_smiles_all_components_retained=True,
        categorical_feature_ranges_verified=True,
        association_node_types=['miRNA','drug'],molecular_atoms_are_internal_drug_descriptors=True,
        labels_used=False,external_data_used=False,pretrained_weights_used=False,lncrna_used=False)
    (HERE/'input_manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:report[k] for k in ['status','n_drugs','n_atoms','n_directed_bonds','atom_counts','multicomponent_drugs']},indent=2))


if __name__=='__main__':main()
