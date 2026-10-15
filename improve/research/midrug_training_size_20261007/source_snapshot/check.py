"""Subset membership, support graphs, gradients and full original query deletion."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from common import HERE,DEGREE,read_module,load_full_inner,load_subset,seed_all,sha256
from model import CONFIGS,SIZES,make_adapter,make_optimizer,mask_generator

QUERY_CHECK_SOURCE=DEGREE/'check.py'
query=read_module('training_size_query_checks',QUERY_CHECK_SOURCE);query.make_adapter=make_adapter
deletion_check=query.deletion_check


def main():
    records=[]
    for fold in range(3):
        full=load_full_inner(fold);previous=set()
        for size,fraction in SIZES.items():
            data=load_subset(fold,size);indices=set(data['train_indices'])
            assert previous.issubset(indices);previous=indices
            assert indices.issubset(set(full['train_indices']))
            assert not indices.intersection(full['val_indices'])
            for k in ['val_pairs','val_y','val_indices','test_pairs']:np.testing.assert_array_equal(data[k],full[k])
            selected=np.isin(full['train_indices'],data['train_indices'])
            np.testing.assert_array_equal(full['train_pairs'][selected],data['train_pairs'])
            np.testing.assert_array_equal(full['train_y'][selected],data['train_y'])
            seed_all(42);plain=make_adapter(data,'control','cuda:0')
            for name,cfg in CONFIGS.items():
                seed_all(42);a=make_adapter(data,name,'cuda:0');m=a.model
                for key,value in plain.model.state_dict().items():torch.testing.assert_close(value,m.state_dict()[key],atol=0,rtol=0)
                expected=torch.zeros_like(a.ps.A);positive=data['train_pairs'][data['train_y']==1]
                expected[positive[:,0],positive[:,1]]=1
                torch.testing.assert_close(expected,a.ps.A,atol=0,rtol=0)
                np.testing.assert_array_equal(a.graph['miRNA','MiDrug','drug'].edge_index.cpu().numpy().T,positive)
                assert set(a.graph.node_types)=={'miRNA','drug'}
                raw=a.ps.qis(*a.pairs['train'])
                torch.testing.assert_close(raw.mean(0,keepdim=True),a.mu,atol=0,rtol=0)
                torch.testing.assert_close(raw.std(0,keepdim=True)+1e-6,a.sd,atol=0,rtol=0)
                opt=make_optimizer(m);m.train();opt.zero_grad()
                loss=F.binary_cross_entropy_with_logits(a.logits('train'),a.y)
                if cfg['reconstruction_weight']:
                    ix=m.reconstruction.masks(a.graph,.5,mask_generator('cuda:0',20261006+fold))
                    rec,_=m.reconstruction.loss(m.encoder,a.graph,ix,False);loss=loss+10*rec
                assert torch.isfinite(loss);loss.backward()
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
                norm=sum(float(p.grad.square().sum()) for p in m.encoder.parameters() if p.grad is not None)**.5;assert norm>0
                opt.step();assert int(m.eb.num_batches_tracked)==1
                deleted=deletion_check(a,data,name,count=4)
                seed_all(42);flipped=make_adapter({**data,'val_y':1-data['val_y']},name,'cuda:0')
                flipped.model.load_state_dict(m.state_dict());m.eval();flipped.model.eval()
                with torch.no_grad():pa,pb=a.predict('val'),flipped.predict('val')
                np.testing.assert_allclose(pa,pb,atol=2e-5,rtol=2e-5)
                records.append(dict(fold=fold,size=size,config=name,training_count=len(data['train_y']),
                    positive_count=int(data['train_y'].sum()),encoder_gradient_norm=norm,batchnorm_updates=1,
                    explicit_deletions=deleted,heldout_label_flip_max_error=float(abs(pa-pb).max())))
                print('PASS',name,size,fold,flush=True);del a,m,opt,flipped,loss;torch.cuda.empty_cache()
    (HERE/'initial_checks.json').write_text(json.dumps(dict(status='passed',records=records,subset_nesting_and_order_passed=True,
        heldout_sets_unchanged=True,support_exactly_selected_positives=True,query_check_source_sha256=sha256(QUERY_CHECK_SOURCE),
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2));print('ALL PASSED',flush=True)

if __name__=='__main__':main()
