"""Independent text preparation plus baseline identity, label isolation and query deletion."""
import ast,json
import numpy as np
import torch
from torch.nn import functional as F
from common import HERE,load_inner,make_graph,seed_all,sha256
from model import CONFIGS,make_adapter,mask_generator

QUERY_CHECK_SOURCE=HERE.parent/'midrug_degree_encoder_20261006/check.py'
tree=ast.parse(QUERY_CHECK_SOURCE.read_text())
node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='deletion_check')
exec(compile(ast.Module(body=[node],type_ignores=[]),str(QUERY_CHECK_SOURCE),'exec'),globals())

def main():
    assert not (HERE/'initial_checks.json').exists()
    data=load_inner(0);original=make_graph(data)
    seed_all(42);baseline=make_adapter(data,'baseline','cuda:0');baseline.model.eval()
    with torch.no_grad():base_logits=baseline.logits('val')
    states=[];records=[]
    for name in CONFIGS:
        seed_all(42);a=make_adapter(data,name,'cuda:0');a.model.eval()
        states.append({k:v.detach().cpu().clone() for k,v in a.model.named_parameters()})
        for key,value in baseline.model.state_dict().items():torch.testing.assert_close(value,a.model.state_dict()[key],atol=0,rtol=0)
        assert set(a.graph.node_types)=={'miRNA','drug'}
        for kind in original.node_types:torch.testing.assert_close(a.graph[kind].x.cpu(),original[kind].x,atol=0,rtol=0)
        for edge in original.edge_types:torch.testing.assert_close(a.graph[edge].edge_index.cpu(),original[edge].edge_index,atol=0,rtol=0)
        with torch.no_grad():initial_logits=a.logits('val')
        torch.testing.assert_close(initial_logits,base_logits,atol=2e-5,rtol=2e-5)
        seed_all(42);b=make_adapter({**data,'val_y':1-data['val_y']},name,'cuda:0');b.model.eval()
        for key,value in a.model.state_dict().items():torch.testing.assert_close(value,b.model.state_dict()[key],atol=0,rtol=0)
        for part in a.qis:
            torch.testing.assert_close(a.qis[part],b.qis[part],atol=0,rtol=0)
            torch.testing.assert_close(a.ease[part],b.ease[part],atol=0,rtol=0)
        with torch.no_grad():flipped=b.logits('val')
        torch.testing.assert_close(initial_logits,flipped,atol=2e-5,rtol=2e-5)
        a.model.train();rec=a.model.reconstruction
        indices=rec.masks(a.graph,.5,mask_generator('cuda:0',0))
        aux,_=rec.loss(a.model.encoder,a.graph,indices,False)
        loss=F.binary_cross_entropy_with_logits(a.logits('train'),a.y)+10*aux
        assert torch.isfinite(loss);loss.backward()
        grad=float(a.model.encoder.refiner.projection.weight.grad.abs().max())
        assert grad>0 if name!='zero' else grad==0
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.model.parameters())
        torch.optim.RMSprop(a.model.parameters(),lr=1e-4).step()
        records.append(dict(config=name,initial_baseline_max_error=float(abs(initial_logits-base_logits).max()),
            label_flip_max_logit_error=float(abs(initial_logits-flipped).max()),name_projection_max_gradient=grad,
            parameters=sum(p.numel() for p in a.model.parameters()),deletions=deletion_check(a,data,name)))
        print('PASS',name,flush=True)
    for state in states[1:]:
        for key in state:torch.testing.assert_close(state[key],states[0][key],atol=0,rtol=0)
    (HERE/'initial_checks.json').write_text(json.dumps(dict(status='passed',same_initial_parameters=True,
        initial_baseline_equivalence=True,original_graph_features_and_reconstruction_targets=True,
        records=records,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))
if __name__=='__main__':main()
