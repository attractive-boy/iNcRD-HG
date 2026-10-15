"""Same state, protected original targets, held-out label isolation and query deletion."""
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
    data=load_inner(0);original=make_graph(data);states=[];records=[]
    for name in CONFIGS:
        seed_all(42);a=make_adapter(data,name,'cuda:0')
        states.append({k:v.detach().cpu().clone() for k,v in a.model.state_dict().items()})
        assert set(a.graph.node_types)=={'miRNA','drug'}
        assert a.graph['miRNA'].x.shape==original['miRNA'].x.shape
        torch.testing.assert_close(a.graph['drug'].x.cpu(),original['drug'].x,atol=0,rtol=0)
        for edge in original.edge_types:
            torch.testing.assert_close(a.graph[edge].edge_index.cpu(),original[edge].edge_index,atol=0,rtol=0)
        for kind in original.node_types:
            torch.testing.assert_close(getattr(a.model.reconstruction,kind+'_target').cpu(),original[kind].x,atol=0,rtol=0)
        seed_all(42);flipped=make_adapter({**data,'val_y':1-data['val_y']},name,'cuda:0')
        for key,value in a.model.state_dict().items():torch.testing.assert_close(value,flipped.model.state_dict()[key],atol=0,rtol=0)
        for kind in original.node_types:torch.testing.assert_close(a.graph[kind].x,flipped.graph[kind].x,atol=0,rtol=0)
        for part in a.qis:
            torch.testing.assert_close(a.qis[part],flipped.qis[part],atol=0,rtol=0)
            torch.testing.assert_close(a.ease[part],flipped.ease[part],atol=0,rtol=0)
        a.model.eval();flipped.model.eval()
        with torch.no_grad():before,after=a.logits('val'),flipped.logits('val')
        torch.testing.assert_close(before,after,atol=2e-5,rtol=2e-5)
        a.model.train();rec=a.model.reconstruction
        indices=rec.masks(a.graph,.5,mask_generator('cuda:0',0))
        aux,_=rec.loss(a.model.encoder,a.graph,indices,False)
        loss=F.binary_cross_entropy_with_logits(a.logits('train'),a.y)+10*aux
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.model.parameters())
        torch.optim.RMSprop(a.model.parameters(),lr=1e-4).step()
        records.append(dict(config=name,label_flip_max_logit_error=float(abs(before-after).max()),
            parameters=sum(p.numel() for p in a.model.parameters()),deletions=deletion_check(a,data,name)))
        print('PASS',name,flush=True)
    for state in states[1:]:
        for key in state:torch.testing.assert_close(state[key],states[0][key],atol=0,rtol=0)
    (HERE/'initial_checks.json').write_text(json.dumps(dict(status='passed',same_initial_state=True,
        original_reconstruction_targets=True,records=records,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))
if __name__=='__main__':main()
