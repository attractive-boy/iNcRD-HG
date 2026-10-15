"""Independent neighbor summation, own-label exclusion, baseline and physical deletion."""
import ast,json
import numpy as np
import torch
from torch.nn import functional as F
from common import HERE,load_inner,seed_all
from model import CONFIGS,make_adapter,NameEvidence,mask_generator

QUERY_CHECK_SOURCE=HERE.parent/'midrug_degree_encoder_20261006/check.py'
tree=ast.parse(QUERY_CHECK_SOURCE.read_text())
node=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='deletion_check')
exec(compile(ast.Module(body=[node],type_ignores=[]),str(QUERY_CHECK_SOURCE),'exec'),globals())

def oracle(data,kernel,pairs):
    result=[]
    for r,d in pairs:
        mask=(data['train_pairs'][:,0]==r)&(data['train_pairs'][:,1]!=d)
        drugs=data['train_pairs'][mask,1];labels=data['train_y'][mask]
        p=sum(float(kernel[j,d]) for j,y in zip(drugs,labels) if y==1)
        n=sum(float(kernel[j,d]) for j,y in zip(drugs,labels) if y==0)
        mass=sum(float(kernel[j,d]) for j in range(216) if j!=d)
        result.append([np.log1p(p),np.log1p(n),p/(p+n+1),np.log1p(p+n),p/(mass+1)])
    return np.asarray(result,dtype=np.float32)

def main():
    assert not (HERE/'initial_checks.json').exists()
    data=load_inner(0);seed_all(42);base=make_adapter(data,'baseline','cuda:0');base.model.eval()
    with torch.no_grad():base_logits=base.logits('val')
    indices=np.unique(np.r_[np.flatnonzero(data['train_y']==1)[:8],np.flatnonzero(data['train_y']==0)[:8],np.linspace(0,len(data['train_y'])-1,20,dtype=int)])
    states=[];records=[]
    for name in CONFIGS:
        seed_all(42);a=make_adapter(data,name,'cuda:0');a.model.eval()
        states.append({k:v.detach().cpu().clone() for k,v in a.model.state_dict().items()})
        with torch.no_grad():initial=a.logits('val')
        torch.testing.assert_close(initial,base_logits,atol=2e-5,rtol=2e-5)
        evidence=a.ps.evidence;query=data['train_pairs'][indices]
        actual=evidence.features(torch.as_tensor(query[:,0],device='cuda:0'),torch.as_tensor(query[:,1],device='cuda:0')).cpu().numpy()
        expected=oracle(data,evidence.kernel.cpu().numpy(),query)
        np.testing.assert_allclose(actual,expected,atol=1e-6,rtol=1e-6)
        for ix in np.r_[np.flatnonzero(data['train_y']==1)[:3],np.flatnonzero(data['train_y']==0)[:3]]:
            changed=data['train_y'].copy();changed[ix]=1-changed[ix]
            other=NameEvidence({**data,'train_y':changed},name,'cuda:0')
            r,d=data['train_pairs'][ix]
            torch.testing.assert_close(evidence.values[r,d],other.values[r,d],atol=0,rtol=0)
        seed_all(42);flipped=make_adapter({**data,'val_y':1-data['val_y']},name,'cuda:0');flipped.model.eval()
        for part in a.qis:
            torch.testing.assert_close(a.qis[part],flipped.qis[part],atol=0,rtol=0)
            torch.testing.assert_close(a.ease[part],flipped.ease[part],atol=0,rtol=0)
        for key,value in a.model.state_dict().items():torch.testing.assert_close(value,flipped.model.state_dict()[key],atol=0,rtol=0)
        a.model.train();rec=a.model.reconstruction
        sample=rec.masks(a.graph,.5,mask_generator('cuda:0',0));aux,_=rec.loss(a.model.encoder,a.graph,sample,False)
        loss=F.binary_cross_entropy_with_logits(a.logits('train'),a.y)+10*aux
        assert torch.isfinite(loss);loss.backward()
        grad=float(a.model.pc[0].weight.grad[:,-5:].abs().max())
        assert grad>0 if name!='zero' else grad==0
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.model.parameters())
        torch.optim.RMSprop(a.model.parameters(),lr=1e-4).step()
        records.append(dict(config=name,baseline_initial_max_error=float(abs(initial-base_logits).max()),
            oracle_max_error=float(abs(actual-expected).max()),own_label_excluded=True,
            appended_input_max_gradient=grad,explicit_deletions=deletion_check(a,data,name)))
        print('PASS',name,flush=True)
    for state in states[1:]:
        for key in state:torch.testing.assert_close(state[key],states[0][key],atol=0,rtol=0)
    (HERE/'initial_checks.json').write_text(json.dumps(dict(status='passed',records=records,
        same_initial_state=True,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))
if __name__=='__main__':main()
