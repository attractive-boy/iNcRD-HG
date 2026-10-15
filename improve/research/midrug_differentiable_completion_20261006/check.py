"""Forward identity plus independent PyG/deleted-path parameter gradients."""
import gc,json,time
import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from common import HERE,Adapter,read_module,load_inner,seed_all
from model import CONFIGS,PREVIOUS,previous,make_adapter,configure,auxiliary_loss,mask_generator
from incrd_hg_m123.config import ModelConfig

prior_check=read_module('differentiable_prior_checks',PREVIOUS/'check.py')
deletion_check=prior_check.deletion_check
feature_check=prior_check.feature_check
QUERY_CHECK_SOURCE=prior_check.QUERY_CHECK_SOURCE

def explicit_first(a):
    out={};enc=a.model.encoder
    for edge in a.graph.edge_types:
        conv=enc.original.conv1['__'.join(edge)]
        value=conv((a.graph[edge[0]].x,a.graph[edge[2]].x),a.graph[edge].edge_index)
        out[edge[2]]=out.get(edge[2],0)+value
    return {n:F.leaky_relu(v/enc.counts[n],.1) for n,v in out.items()}

def path_oracle(a,r,d):
    h=explicit_first(a);s=a.model.soft_paths.scorer
    P=s.matrix(s.mi(h['miRNA']),s.drug(h['drug'])).sigmoid();U=P*(~a.ps.known)
    A=a.ps.A;left=middle=right=P.sum()*0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if i==r or j==d:continue
            left=left+U[r,j]*A[i,j]*A[i,d]
            middle=middle+A[r,j]*U[i,j]*A[i,d]
            right=right+A[r,j]*A[i,j]*U[i,d]
    return torch.stack((left,middle,right)).log1p()[None]

def gradient_checks():
    old=torch.get_default_dtype();torch.set_default_dtype(torch.float64);records=[]
    try:
        seed_all(123);g=HeteroData();g['miRNA'].x=torch.randn(5,4);g['drug'].x=torch.randn(4,3)
        g['miRNA','MiMi','miRNA'].edge_index=torch.tensor([[0,0],[1,1],[2,2],[3,3],[4,4],[0,1],[1,0],[2,4],[4,2]]).t()
        g['drug','DrugDrug','drug'].edge_index=torch.tensor([[0,0],[1,1],[2,2],[3,3],[0,1],[1,0]]).t()
        pos=np.array([[0,0],[0,1],[1,0],[1,1],[2,2],[3,1]])
        e=torch.as_tensor(pos.T.copy());g['miRNA','MiDrug','drug'].edge_index=e;g['drug','rev_MiDrug','miRNA'].edge_index=e.flip(0)
        pairs=np.r_[pos,[[4,3],[1,3],[4,0]]];y=np.r_[np.ones(len(pos)),np.zeros(3)]
        data=dict(train_pairs=pairs,train_y=y,val_pairs=pairs,val_y=y,test_pairs=np.empty((0,2),dtype=np.int64))
        def build(graph,record,full=True):
            cfg=CONFIGS['full'] if full else previous.CONFIGS['soft']
            a=configure(Adapter(record,graph,'cpu',ModelConfig(hidden=8,ncn_dim=4)),record,cfg);a.model.eval();return a
        a=build(g,data);stopped=build(g,data,False)
        with torch.no_grad():a.model.soft_paths.head[-1].weight.normal_(std=.1)
        stopped.model.load_state_dict(a.model.state_dict())
        for r,d in [(0,0),(3,1),(4,3),(2,1)]:
            pair=torch.tensor([[r],[d]]);deleted=g.clone()
            for edge,s,t in [(('miRNA','MiDrug','drug'),r,d),(('drug','rev_MiDrug','miRNA'),d,r)]:
                ei=deleted[edge].edge_index;deleted[edge].edge_index=ei[:,~((ei[0]==s)&(ei[1]==t))]
            keep=~((pairs[:,0]==r)&(pairs[:,1]==d));b=build(deleted,{**data,'train_pairs':pairs[keep],'train_y':y[keep]})
            b.model.load_state_dict(a.model.state_dict())
            def output(obj):return obj.model(pair,obj.ps,(obj.ps.qis(*pair)-a.mu)/a.sd,obj.ps.ease(*pair),obj.dense)
            u,v,w=output(a),output(b),output(stopped)
            torch.testing.assert_close(u,v,atol=1e-8,rtol=1e-7);torch.testing.assert_close(u,w,atol=1e-10,rtol=1e-9)
            _,fields=a.model.soft_paths.features(a.model.encoder,a.ps,a.dense,*pair)
            _,fixed=stopped.model.soft_paths.features(stopped.model.encoder,stopped.ps,stopped.dense,*pair)
            oracle=path_oracle(b,r,d)
            assert fields.requires_grad and not fixed.requires_grad
            torch.testing.assert_close(fields,oracle,atol=1e-10,rtol=1e-9)
            # This term verifies gradients through the path probabilities,
            # in addition to the complete prediction's gradient.
            gpath=torch.autograd.grad(fields.sum(),list(a.model.soft_paths.scorer.parameters()),retain_graph=True,allow_unused=True)
            path_norm=sum(float(x.abs().sum()) for x in gpath if x is not None)
            la=F.softplus(u).sum()+.01*fields.square().sum()
            lb=F.softplus(v).sum()+.01*oracle.square().sum()
            ga=torch.autograd.grad(la,list(a.model.parameters()),allow_unused=True)
            gb=torch.autograd.grad(lb,list(b.model.parameters()),allow_unused=True);err=0.
            for x,z in zip(ga,gb):
                assert (x is None)==(z is None)
                if x is not None:torch.testing.assert_close(x,z,atol=2e-7,rtol=2e-6);err=max(err,float(abs(x-z).max()))
            records.append(dict(pair=[r,d],max_gradient_error=err,logit_error=float(abs(u-v).max()),
                stopped_forward_error=float(abs(u-w).max()),path_gradient_l1=path_norm))
        assert max(x['path_gradient_l1'] for x in records)>0
    finally:torch.set_default_dtype(old)
    return records

def main():
    gradients=gradient_checks();print('PASS independent full path gradients',flush=True)
    data=load_inner(0);rows=[]
    for name,cfg in CONFIGS.items():
        seed_all(42);a=make_adapter(data,name,'cuda:0');m=a.model;m.eval()
        stopped=make_adapter(data,'soft_rec10' if cfg['reconstruction_weight'] else 'soft','cuda:0')
        stopped.model.load_state_dict(m.state_dict());stopped.model.eval()
        with torch.no_grad():p0=a.predict('val');p1=stopped.predict('val');m.soft_paths.head[-1].weight.normal_(std=.1)
        np.testing.assert_allclose(p0,p1,atol=2e-6,rtol=2e-6)
        stopped.model.load_state_dict(m.state_dict())
        with torch.no_grad():p0=a.predict('val');p1=stopped.predict('val')
        np.testing.assert_allclose(p0,p1,atol=2e-6,rtol=2e-6)
        del stopped;gc.collect();torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats();start=time.time();m.train();opt=torch.optim.RMSprop(m.parameters(),lr=1e-4);opt.zero_grad()
        loss=F.binary_cross_entropy_with_logits(a.logits('train'),a.y)+cfg['auxiliary_weight']*auxiliary_loss(a)
        if cfg['reconstruction_weight']:
            rec=m.reconstruction;idx=rec.masks(a.graph,.5,mask_generator('cuda:0',0));aux,_=rec.loss(m.encoder,a.graph,idx,False);loss=loss+cfg['reconstruction_weight']*aux
        loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
        gn={k:float(p.grad.norm()) for k,p in m.soft_paths.named_parameters()};assert gn['scorer.mi.1.weight']>0 and gn['scorer.drug.1.weight']>0
        opt.step();torch.cuda.synchronize()
        rows.append(dict(config=name,nonzero_head_forward_identity_error=float(abs(p0-p1).max()),step_seconds=time.time()-start,
            peak_cuda_gib=torch.cuda.max_memory_allocated()/2**30,branch_gradients=gn,
            enumerated_features=feature_check(a,data),explicit_deletions=deletion_check(a,data,name)))
        print('PASS',name,flush=True);del a,m,opt;gc.collect();torch.cuda.empty_cache()
    a=make_adapter(data,'full','cuda:0');a.model.eval()
    with torch.no_grad():a.model.soft_paths.head[-1].weight.normal_(std=.1)
    b=make_adapter({**data,'val_y':1-data['val_y']},'full','cuda:0');b.model.load_state_dict(a.model.state_dict());b.model.eval()
    with torch.no_grad():pa,pb=a.predict('val'),b.predict('val')
    np.testing.assert_allclose(pa,pb,atol=2e-6,rtol=2e-6)
    (HERE/'initial_checks.json').write_text(json.dumps(dict(status='passed',gradient_checks=gradients,records=rows,
        heldout_flip_error=float(abs(pa-pb).max()),no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))

if __name__=='__main__':main()
