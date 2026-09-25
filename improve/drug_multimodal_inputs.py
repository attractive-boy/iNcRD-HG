"""Encode the fixed drug dictionary using the public M2UMol encoder.

Run in the isolated M2UMol environment. Does not read task outcome labels.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
import pandas as pd

from improve import tabular_foundation as base


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[1]
SOURCE = WORKSPACE/'.external/M2UMol_20260914'
WEIGHTS = WORKSPACE/'data_external/m2umol_20260914/pre-trained_M2UMol.pt'
CONSTRUCT = WORKSPACE/'incrd-run/2.construct_network'
DICTIONARY = ROOT/'results/tabicl_pathway_context_inputs_20260913/dictionary.json'
DEFAULT_OUT = ROOT/'results/drug_multimodal_inputs_20260914_v3'


def parameter_digest(model):
    h = hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def encode(args):
    out = Path(args.out_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new drug representation directory')
    download = base.read_json(WEIGHTS.parent/'DOWNLOAD.json')
    if base.digest(WEIGHTS)!=download['sha256'] or download['sha256']!='7a61f9a9b6222585c685051b5c56435801cf54e032d7dd612dda3dbf70320fcd':
        raise ValueError('Downloaded weights disagree with the public LFS record')
    source_record = base.read_json(SOURCE/'FETCH.json')
    pretrain_record = base.read_json(SOURCE/'PRETRAIN_FETCH.json')
    if source_record['commit']!=pretrain_record['commit']:
        raise ValueError('M2UMol sources use different commits')
    source_files = dict(source_record['files'],**pretrain_record['files'])
    for name,record in source_files.items():
        if base.digest(SOURCE/name)!=record['sha256']:
            raise ValueError('Downloaded encoder source changed')
    inputs = [WEIGHTS,WEIGHTS.parent/'DOWNLOAD.json',WEIGHTS.parent/'checkpoint_lfs_pointer.txt',
        WEIGHTS.parent/'LFS_VERIFICATION.json',SOURCE/'FETCH.json',SOURCE/'PRETRAIN_FETCH.json',DICTIONARY,
        ROOT/'m2umol_frozen_encoder.py',ROOT/'test_m2umol_frozen_encoder.py',
        CONSTRUCT/'Drug.csv',CONSTRUCT/'drug_smiles.csv',CONSTRUCT/'drug_features-chemBertA.csv']
    inputs.extend(SOURCE/name for name in source_files)
    protocol = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        github_commit=source_record['commit'],checkpoint_sha256=download['sha256'],
        checkpoint_repository_commit='ca30b0fca6a5018fe0c1b4711ec1770c4ee8610c',
        inputs={str(p):base.digest(p) for p in inputs},implementation_sha256=base.digest(__file__),
        representation='128D feats2d from the original pretraining_M2UMol/layer.py forward, cut exactly after global_add_pool. OGB AtomEncoder/BondEncoder and all three GraphGPS layers loaded strictly. No downstream fusion head or modality-output selection.',
        compatibility_amendment='Before any drug encoding or task scoring, the public package example failed strict checkpoint checks: its 103D/6D linear input layers and task fusion head are absent from the checkpoint. The revised extraction uses the exact original pretraining graph functions and forward prefix with its categorical 9D/3D inputs. Earlier failed directories remain archived; neither produced vectors or scores. This is a frozen pretraining-2D representation experiment, not a reproduction of the paper final fine-tuned heads.',
        input_policy='Keep all original SMILES atoms, disconnected components, charges and salts. Extract official pretraining graph functions without running their data-loading module. Unsupported/edgeless graph shapes are left uncovered; never invent bonds, discard fragments or drop candidate rows.',
        loader='torch.load(weights_only=True,map_location=cpu); require every state entry of the extracted 2D encoder to load by exact name and shape. Ignore only unused other-modality state. No fine tuning.',
        descriptors=['m2umol_present','log1p_heavy_atoms','log1p_molecular_weight','log1p_TPSA','fraction_CSP3','formal_charge','log1p_rings','log1p_fragments'],
        controls='Existing CID-aligned ChemBERTa and radius-2 2048-bit Morgan with chirality, each masked to exact M2UMol coverage. Eight structure covariates remain for every parseable drug. Other controls supplied by downstream frozen protocol.',
        inference_device='cpu',seed=20260914,batch_size=16,replay_atol=1e-4,replay_rtol=1e-5,
        outcome_labels_read=False,pretrained_source_disjointness_claimed=False,
        provenance_limitation='Pretraining includes DrugBank mechanism, clinical text and target knowledge. Generated modality vectors are predictions, not measured functional data.'))
    out.mkdir(parents=True)
    base.write_json(out/'ENCODING_PROTOCOL.json',protocol)
    shutil.copy2(__file__,out/'encoding_source.py')
    # Only the original 2D helper definitions/forward prefix are compiled;
    # no 3D/text model or pretraining dataset is instantiated or loaded.
    code = SOURCE/'pretraining_M2UMol'
    import torch
    import torch_geometric
    import rdkit
    from rdkit import Chem,DataStructs
    from rdkit.Chem import AllChem,Descriptors,rdMolDescriptors
    from torch_geometric.data import Batch
    from improve.m2umol_frozen_encoder import M2UMolFrozen2D,graph_builder
    torch.set_num_threads(4)
    torch.manual_seed(protocol['seed'])
    np.random.seed(protocol['seed'])
    model = M2UMolFrozen2D(SOURCE).cpu().eval()
    checkpoint = torch.load(WEIGHTS,map_location='cpu',weights_only=True)
    loading = model.load_pretrained(checkpoint)
    missing = loading['missing']
    model.requires_grad_(False)
    before = parameter_digest(model)
    del checkpoint
    create_graph = graph_builder(SOURCE)
    dictionary = base.read_json(DICTIONARY)
    drugs = pd.read_csv(CONSTRUCT/'Drug.csv',dtype=str).set_index('Drug_Name')
    names = np.asarray(dictionary['drug_names'])
    cids = drugs.loc[names,'CID'].to_numpy(str)
    smiles_table = pd.read_csv(CONSTRUCT/'drug_smiles.csv',dtype=str).set_index('drug_cid')
    smiles = smiles_table.loc[cids,'smiles'].to_numpy(str)
    chemical = pd.read_csv(CONSTRUCT/'drug_features-chemBertA.csv',dtype={'drug_cid':str}).set_index('drug_cid')
    if not chemical.index.is_unique or not smiles_table.index.is_unique:
        raise ValueError('Duplicate drug feature CID')
    chemberta = chemical.loc[cids].to_numpy(np.float32)
    n = len(names)
    simple = np.zeros((n,8),np.float32)
    fingerprint = np.zeros((n,2048),np.float32)
    present,parsed = np.zeros(n,bool),np.zeros(n,bool)
    graphs,ids,audits = [],[],[]
    cwd = Path.cwd()
    start = time.monotonic()
    try:
        os.chdir(code)
        for i,(name,cid,smile) in enumerate(zip(names,cids,smiles)):
            record = dict(index=i,name=str(name),cid=str(cid),parsed=False,covered=False,error=None)
            try:
                mol = Chem.MolFromSmiles(smile)
                if mol is None or not mol.GetNumAtoms():
                    raise ValueError('RDKit did not produce a nonempty molecule')
                parsed[i] = True
                fragments = len(Chem.GetMolFrags(mol))
                simple[i,1:] = [np.log1p(mol.GetNumHeavyAtoms()),np.log1p(Descriptors.MolWt(mol)),
                    np.log1p(rdMolDescriptors.CalcTPSA(mol)),rdMolDescriptors.CalcFractionCSP3(mol),
                    Chem.GetFormalCharge(mol),np.log1p(rdMolDescriptors.CalcNumRings(mol)),np.log1p(fragments)]
                bits = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=2048,useChirality=True)
                DataStructs.ConvertToNumpyArray(bits,fingerprint[i])
                record.update(parsed=True,atoms=mol.GetNumAtoms(),bonds=mol.GetNumBonds(),fragments=fragments,
                    metals=sorted(set(a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum() in [3,4,11,12,13,19,20,26,27,28,29,30,46,47,48,78,79,80])))
                graph = create_graph(mol,smile).cpu()
                if tuple(graph.x.shape)!=(mol.GetNumAtoms(),9) or graph.edge_index.ndim!=2 or graph.edge_index.shape[0]!=2 or graph.edge_attr.ndim!=2 or graph.edge_attr.shape[1]!=3:
                    raise ValueError('Unsupported official graph shape (including edgeless molecules)')
                if not torch.isfinite(graph.x).all() or not torch.isfinite(graph.edge_attr).all():
                    raise ValueError('Nonfinite graph features')
                graphs.append(graph)
                ids.append(i)
                present[i] = True
                record['covered'] = True
            except Exception as error:
                record['error'] = type(error).__name__+': '+str(error)
            audits.append(record)
    finally:
        os.chdir(cwd)
    if not len(graphs):
        raise ValueError('No valid graphs for the frozen encoder')
    def predict(order,batch_size):
        result = []
        with torch.inference_mode():
            for offset in range(0,len(order),batch_size):
                batch = Batch.from_data_list([graphs[j].clone() for j in order[offset:offset+batch_size]]).cpu()
                outputs = model(batch)
                if outputs.shape!=(batch.num_graphs,128) or not torch.isfinite(outputs).all():
                    raise ValueError('Invalid official encoder output')
                result.append(outputs.numpy())
        return np.concatenate(result,axis=0)
    order = list(range(len(graphs)))
    values = predict(order,16)
    reversed_values = predict(order[::-1],16)[::-1]
    split_values = predict(order,7)
    for replay in (reversed_values,split_values):
        np.testing.assert_allclose(values,replay,atol=protocol['replay_atol'],rtol=protocol['replay_rtol'])
    after = parameter_digest(model)
    if before!=after:
        raise ValueError('Frozen encoder state changed during inference')
    m2umol = np.zeros((n,128),np.float32)
    m2umol[ids] = values
    simple[:,0] = present
    # Exact coverage matching applies to all encoded controls, not the candidate
    # rows or their available basic molecular descriptors.
    fingerprint[~present] = 0
    chemberta[~present] = 0
    np.savez_compressed(out/'representations.npz',simple=simple,m2umol=m2umol,chemberta=chemberta,
        fingerprint=fingerprint,present=present,parsed=parsed,names=names,cids=cids)
    pd.DataFrame(audits).to_csv(out/'drug_audit.csv',index=False)
    checks = dict(used_parameters_complete=True,missing_unused_state=missing,loading=loading,parameters_unchanged=True,
        parameter_sha256=before,reversed_max_abs=float(np.max(np.abs(values-reversed_values))),
        batch_max_abs=float(np.max(np.abs(values-split_values))),all_candidates_retained=True,
        encoder_states=128,chemberta_dimensions=chemberta.shape[1],fingerprint_dimensions=2048)
    base.write_json(out/'CHECKS.json',checks)
    files = ['representations.npz','drug_audit.csv','CHECKS.json','ENCODING_PROTOCOL.json','encoding_source.py']
    manifest = base.seal(dict(files={name:base.digest(out/name) for name in files},inputs=protocol['inputs'],
        encoding_protocol_sha256=protocol['sha256'],checks=checks,drugs=n,covered=int(present.sum()),parsed=int(parsed.sum()),
        uncovered=[r for r in audits if not r['covered']],seconds=time.monotonic()-start,
        versions=dict(python=sys.version,torch=torch.__version__,pyg=torch_geometric.__version__,rdkit=rdkit.__version__,
            torch_scatter=importlib.metadata.version('torch-scatter'),ogb=importlib.metadata.version('ogb')),
        source_code_sha256=base.digest(__file__),outcome_labels_read=False,pretrained_source_disjointness_claimed=False))
    base.write_json(out/'INPUTS.json',manifest)
    print('M2UMOL_ENCODED',manifest['covered'],'/',n,checks,flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',default=str(DEFAULT_OUT))
    encode(parser.parse_args())
