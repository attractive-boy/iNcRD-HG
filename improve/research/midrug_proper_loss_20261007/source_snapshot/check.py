"""Independent proper-loss mathematics and original M123 integration checks."""
import ast,json
from decimal import Decimal,localcontext
import numpy as np
import torch
from scipy.special import expit
from scipy.optimize import minimize_scalar
from torch.nn import functional as F
from common import HERE,load_inner,seed_all
from model import CONFIGS,make_adapter,make_optimizer,mask_generator,proper_loss,gradient_factor,association_loss
QUERY_SOURCE=HERE.parent/'midrug_degree_encoder_20261006/check.py'
tree=ast.parse(QUERY_SOURCE.read_text());body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='deletion_check']
ns=dict(np=np,torch=torch,make_adapter=make_adapter);exec(compile(ast.Module(body=body,type_ignores=[]),str(QUERY_SOURCE),'exec'),ns);deletion_check=ns['deletion_check']

def oracle(z,y,name):
    with localcontext() as ctx:
        ctx.prec=100;zz=Decimal(str(float(z)));p=1/(1+(-zz).exp());q=1-p
        if name=='negative_tail':v=2*(-p.ln()-q) if y else 2*p
        elif name=='positive_tail':v=2*q if y else 2*(-q.ln()-p)
        else:v=2*(q*q if y else p*p)
        return float(v)

def mathematical_check():
    transition=-np.log(np.expm1(.01))
    zz=np.array([-80.,-20.,-5.,0.,transition-1e-5,transition,transition+1e-5,5.,20.,80.]);records=[]
    for name in CONFIGS:
        z=torch.tensor(np.tile(zz,2),dtype=torch.double,requires_grad=True);y=torch.tensor(np.repeat([0.,1.],len(zz)),dtype=torch.double)
        values=proper_loss(z,y,name);g=torch.autograd.grad(values.sum(),z)[0]
        p,q=expit(z.detach().numpy()),expit(-z.detach().numpy());sign=np.where(y.numpy()==1,-q,p)
        factor=2*q if name=='negative_tail' else 2*p if name=='positive_tail' else 4*p*q
        expected=np.array([oracle(a,b,name) for a,b in zip(z.detach().numpy(),y.numpy())])
        np.testing.assert_allclose(values.detach().numpy(),expected,atol=2e-13,rtol=2e-11)
        np.testing.assert_allclose(g.detach().numpy(),factor*sign,atol=2e-13,rtol=2e-10)
        assert (values>=0).all() and torch.isfinite(values).all() and torch.isfinite(g).all()
        assert torch.autograd.gradcheck(lambda x:proper_loss(x,y,name).sum(),(z,),eps=1e-5,atol=1e-7,rtol=1e-5)
        zero=torch.zeros(2,dtype=torch.double,requires_grad=True);yy=torch.tensor([0.,1.],dtype=torch.double)
        gg=torch.autograd.grad(proper_loss(zero,yy,name).sum(),zero)[0]
        torch.testing.assert_close(gg,torch.tensor([.5,-.5],dtype=torch.double),atol=1e-15,rtol=0)
        optimum_errors=[]
        for eta in [.01,.05,.1,.25,.5,.75,.9,.95,.99]:
            def risk(pr):
                if name=='negative_tail':l0,l1=2*pr,2*(-np.log(pr)-(1-pr))
                elif name=='positive_tail':l0,l1=2*(-np.log1p(-pr)-pr),2*(1-pr)
                else:l0,l1=2*pr**2,2*(1-pr)**2
                return (1-eta)*l0+eta*l1
            result=minimize_scalar(risk,bounds=(1e-8,1-1e-8),method='bounded',options={'xatol':1e-13})
            assert result.success and abs(result.x-eta)<1e-7;optimum_errors.append(abs(result.x-eta))
        records.append(dict(config=name,max_value_error=float(np.max(abs(values.detach().numpy()-expected))),
            max_gradient_error=float(np.max(abs(g.detach().numpy()-factor*sign))),
            gradcheck=True,max_population_optimum_error=max(optimum_errors),extreme_values_finite=True))
    z=torch.linspace(-30,30,151,dtype=torch.double);y=(torch.arange(151)%2).double()
    torch.testing.assert_close(proper_loss(z,y,'negative_tail'),proper_loss(-z,1-y,'positive_tail'),atol=0,rtol=0)
    return dict(records=records,reflection_exact=True,all_zero_logit_gradients_match_bce=True,
        strict_propriety_is_not_noise_invariance=True)

def main():
    math=mathematical_check();data=load_inner(0);records=[]
    for name,cfg in CONFIGS.items():
        seed_all(42);a=make_adapter(data,name,'cuda:0');seed_all(42);b=make_adapter(data,'unmasked10','cuda:0')
        for k,v in a.model.state_dict().items():torch.testing.assert_close(v,b.model.state_dict()[k],atol=0,rtol=0)
        a.model.train();b.model.train();opt=make_optimizer(a.model);opt.zero_grad(set_to_none=True)
        loss,stats,_=association_loss(a,cfg,1,0);plain=F.binary_cross_entropy_with_logits(b.logits('train'),b.y)
        loss.backward();plain.backward();errors=[]
        for (k,p),(l,q) in zip(a.model.named_parameters(),b.model.named_parameters()):
            assert k==l and ((p.grad is None)==(q.grad is None))
            if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=2e-6,rtol=2e-5);errors.append(float(abs(p.grad-q.grad).max()))
        assert int(a.model.eb.num_batches_tracked)==1
        del b;opt.zero_grad(set_to_none=True);loss,stats,_=association_loss(a,cfg,60,0)
        idx=a.model.reconstruction.masks(a.graph,.5,mask_generator('cuda:0',20261006));rec,_=a.model.reconstruction.loss(a.model.encoder,a.graph,idx,False)
        (loss+10*rec).backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in a.model.parameters())
        assert any(p.grad is not None and p.grad.abs().max()>0 for p in a.model.encoder.parameters());opt.step();a.model.eval()
        assert int(a.model.eb.num_batches_tracked)==2
        deleted=deletion_check(a,data,name,count=4)
        b=make_adapter({**data,'val_y':1-data['val_y']},name,'cuda:0');b.model.load_state_dict(a.model.state_dict());b.model.eval()
        with torch.no_grad():pa,pb=a.predict('val'),b.predict('val');la,sa,_=association_loss(a,cfg,60,0);lb,sb,_=association_loss(b,cfg,60,0)
        np.testing.assert_allclose(pa,pb,atol=2e-5,rtol=2e-5);assert abs(float(la-lb))<2e-6
        records.append(dict(config=name,zero_transition_gradient_error=max(errors),full_transition_stats=stats,
            physical_deletions=deleted,heldout_flip_probability_error=float(abs(pa-pb).max()),heldout_flip_loss_error=abs(float(la-lb))))
        print('PASS',name,flush=True);del a,b,opt;torch.cuda.empty_cache()
    (HERE/'initial_checks.json').write_text(json.dumps(dict(status='passed',mathematics=math,records=records,no_lncrna=True,
        outer_validation_evaluated=False,test_evaluated=False),indent=2))

if __name__=='__main__':main()

