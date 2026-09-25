"""Audit public encoder/checkpoint compatibility without outcome prediction."""
from pathlib import Path
import sys

import torch

from improve import tabular_foundation as base
from improve.m2umol_frozen_encoder import M2UMolFrozen2D


def main():
    workspace = Path(__file__).resolve().parents[2]
    source = workspace/'.external/M2UMol_20260914'
    weights = workspace/'data_external/m2umol_20260914/pre-trained_M2UMol.pt'
    out = Path(__file__).resolve().parent/'results/frontier_validation_20260914'
    out.mkdir(parents=True,exist_ok=True)
    target = out/'M2UMOL_COMPATIBILITY.json'
    if target.exists():
        raise ValueError('Compatibility audit already recorded')
    sys.path.insert(0,str(source/'M2UMol_package/molecular_representation_learning'))
    from M2UMol import M2UMolencoder
    state = torch.load(weights,map_location='cpu',weights_only=True)
    example = M2UMolencoder().cpu().eval()
    expected = example.state_dict()
    missing = sorted(set(expected)-set(state))
    absent_used = [key for key in missing if not key.startswith(('linfi.','classification_mlp.'))]
    assert len(missing)==17 and len(absent_used)==13
    encoder = M2UMolFrozen2D(source)
    loading = encoder.load_pretrained(state)
    assert loading['loaded_state_entries']==103 and not loading['missing']
    for key,tensor in encoder.state_dict().items():
        torch.testing.assert_close(tensor,state[key],rtol=0,atol=0)
    base.write_json(target,dict(audit_source_sha256=base.digest(__file__),checkpoint_sha256=base.digest(weights),
        github_commit=base.read_json(source/'FETCH.json')['commit'],
        public_example_state_entries=len(expected),matched_example_state_entries=len(expected)-len(missing),
        missing_example_entries=missing,missing_used_example_entries=absent_used,
        example_input=dict(atom='Linear(103,120)',bond='Linear(6,128)'),
        checkpoint_input=dict(atom='OGB AtomEncoder, nine categorical features',bond='OGB BondEncoder, three categorical features'),
        pretrained_2d=loading,all_extracted_state_equals_checkpoint=True,
        extraction='Exact upstream GPSConv1 class and forward prefix through feats2d = global_add_pool, plus original OGB atom/bond feature functions. No randomly initialized fusion head.',
        encoding_protocol=str(Path(__file__).resolve().parent/'results/drug_multimodal_inputs_20260914_v3/ENCODING_PROTOCOL.json'),
        scope='Compatibility inspection only; no outcome labels, alternate embedding scores or pretrained-task datasets read.',
        final_paper_finetuning_reproduction_claimed=False))
    print('M2UMOL_COMPATIBILITY_AUDITED',target,flush=True)


if __name__=='__main__':
    main()
