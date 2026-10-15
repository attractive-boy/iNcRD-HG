"""Validate molecular invariances and the complete nonzero association branch."""
import copy,json
import numpy as np
import torch
from torch.nn import functional as F
from common import HERE,load_inner,seed_all,read_module
from model import make_adapter,mask_generator

QUERY_CHECK_SOURCE=HERE.parent/'midrug_degree_encoder_20261006/check.py'
checker=read_module('explicit_query_deletion_helper',QUERY_CHECK_SOURCE)
checker.make_adapter=make_adapter
deletion_check=checker.deletion_check


@torch.no_grad()
def permutation_check(a,require_nonzero=True):
    drug=a.model.molecular.drug
    if drug.mode=='chem':return dict(applicable=False)
    drug.eval()
    af,bf,edge,batch=drug.atom_features,drug.bond_features,drug.edge_index,drug.atom_drug
    generator=torch.Generator(device=af.device).manual_seed(909)
    perm=torch.randperm(len(af),device=af.device,generator=generator)
    inverse=torch.empty_like(perm);inverse[perm]=torch.arange(len(perm),device=af.device)
    eperm=torch.randperm(edge.shape[1],device=af.device,generator=generator)
    original=drug.molecular_features(af,bf,edge,batch)
    actual=drug.molecular_features(af[perm],bf[eperm],inverse[edge[:,eperm]],batch[perm])
    torch.testing.assert_close(original,actual,atol=2e-5,rtol=2e-5)
    # Even isolated one-atom drugs have a defined representation.
    assert torch.isfinite(original).all() and original.shape==(216,64)
    absent=drug.molecular_features(af,bf[:0],edge[:,:0],batch)
    delta=float(abs(absent-original).max())
    if drug.mode=='gnn' and require_nonzero:assert delta>1e-5
    if drug.mode=='bag':
        torch.testing.assert_close(absent,original,atol=2e-5,rtol=2e-5)
        # CUDA index_add pooling is not bitwise deterministic; verify exact bond independence on CPU too.
        cpu=copy.deepcopy(drug).cpu()
        cp=cpu.molecular_features(af.cpu(),bf.cpu(),edge.cpu(),batch.cpu())
        cp0=cpu.molecular_features(af.cpu(),bf[:0].cpu(),edge[:,:0].cpu(),batch.cpu())
        torch.testing.assert_close(cp,cp0,atol=0,rtol=0)
    return dict(applicable=True,max_embedding_permutation_error=float(abs(original-actual).max()),
                without_bonds_max_difference=delta,isolated_atom_finite=True)


def main():
    data=load_inner(0);seed_all(42);base=make_adapter(data,'unmasked10','cuda:0');base.model.eval()
    with torch.no_grad():baseline=base.predict('val')
    records=[];shared=None;gnn_state=None
    for name in ['gnn_wd4','bag_wd4','chem_wd4']:
        seed_all(42);a=make_adapter(data,name,'cuda:0');a.model.eval()
        for k,v in base.model.state_dict().items():torch.testing.assert_close(v,a.model.state_dict()[k],atol=0,rtol=0)
        current={k:v.clone() for k,v in a.model.molecular.state_dict().items() if k.startswith(('rna.','pair.'))}
        if shared is not None:
            for k,v in current.items():torch.testing.assert_close(v,shared[k],atol=0,rtol=0)
        else:shared=current
        if name=='gnn_wd4':gnn_state={k:v.clone() for k,v in a.model.molecular.state_dict().items()}
        if name=='bag_wd4':
            for k,v in a.model.molecular.state_dict().items():torch.testing.assert_close(v,gnn_state[k],atol=0,rtol=0)
        with torch.no_grad():
            prediction=a.predict('val');assert a.model.branch_logits.eq(0).all()
        np.testing.assert_allclose(baseline,prediction,atol=1e-6,rtol=1e-5)
        perm=permutation_check(a)
        # Activate the readout before checking gradients and deletion, avoiding a vacuous zero-branch pass.
        with torch.no_grad():a.model.molecular.pair[-1].weight.normal_(std=.01)
        a.model.train();opt=torch.optim.RMSprop(a.model.parameters(),lr=1e-4);opt.zero_grad()
        logits=a.logits('train');branch=a.model.branch_logits
        rec=a.model.reconstruction;indices=rec.masks(a.graph,.5,mask_generator('cuda:0',0))
        aux,_=rec.loss(a.model.encoder,a.graph,indices,False)
        loss=F.binary_cross_entropy_with_logits(logits,a.y)+.1*F.binary_cross_entropy_with_logits(branch,a.y)+10*aux
        loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.model.parameters())
        drug=a.model.molecular.drug
        active=[p for p in drug.parameters() if p.grad is not None]
        norm=float(torch.stack([p.grad.norm() for p in active]).norm());assert norm>0
        if drug.mode=='gnn':assert sum(float(p.weight.grad.norm()) for p in drug.bond_embeddings)>0
        if drug.mode=='bag':assert all(p.weight.grad is None for p in drug.bond_embeddings)
        opt.step();a.model.eval()
        deletion=deletion_check(a,data,name)
        changed=make_adapter({**data,'val_y':1-data['val_y']},name,'cuda:0')
        changed.model.load_state_dict(a.model.state_dict());changed.model.eval()
        with torch.no_grad():pa,pb=a.predict('val'),changed.predict('val')
        np.testing.assert_allclose(pa,pb,atol=2e-5,rtol=2e-5)
        records.append(dict(config=name,zero_branch_max_prediction_error=float(abs(baseline-prediction).max()),
            permutation=perm,drug_encoder_gradient_norm=norm,explicit_deletions=deletion,
            heldout_label_flip_max_prediction_error=float(abs(pa-pb).max())))
        print('PASS',name,flush=True)
    (HERE/'initial_checks.json').write_text(json.dumps(dict(status='passed',records=records,
        shared_pair_head_initialization_exact=True,gnn_bag_initialization_exact=True,
        no_lncrna=True,external_data_used=False,outer_validation_evaluated=False,test_evaluated=False),indent=2))


if __name__=='__main__':main()
