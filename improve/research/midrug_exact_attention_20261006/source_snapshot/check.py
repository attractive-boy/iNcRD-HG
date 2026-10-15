"""Compare analytic scores AND gradients with independent explicit deletion."""
import gc,json,time
import numpy as np
import torch
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from common import HERE,Adapter,DEGREE,read_module,load_inner,seed_all
from model import AttentionEncoder,AttentionPairStruct,make_adapter,mask_generator
from incrd_hg_m123.config import ModelConfig

QUERY_CHECK_SOURCE=DEGREE/'check.py'
checker=read_module('real_attention_deletion_helper',QUERY_CHECK_SOURCE)
checker.make_adapter=make_adapter
deletion_check=checker.deletion_check


def toy_graph():
    g=HeteroData();g['miRNA'].x=torch.randn(5,4);g['drug'].x=torch.randn(4,3)
    g['miRNA','MiMi','miRNA'].edge_index=torch.tensor([[0,0],[0,0],[1,1],[2,2],[3,3],[4,4],[0,1],[1,0],[2,4],[4,2]]).t()
    g['drug','DrugDrug','drug'].edge_index=torch.tensor([[0,0],[1,1],[2,2],[3,3],[2,2],[0,1],[1,0]]).t()
    positive=np.array([[0,0],[0,1],[1,0],[2,2],[3,1]],dtype=np.int64)
    e=torch.as_tensor(positive.T.copy());g['miRNA','MiDrug','drug'].edge_index=e;g['drug','rev_MiDrug','miRNA'].edge_index=e.flip(0)
    pairs=np.concatenate([positive,np.array([[4,3],[1,3],[4,0]])])
    data=dict(train_pairs=pairs,train_y=np.r_[np.ones(5),np.zeros(3)],val_pairs=pairs.copy(),val_y=np.r_[np.ones(5),np.zeros(3)],test_pairs=np.empty((0,2),dtype=np.int64))
    return g,data


def toy_adapter(graph,data,scope,alpha):
    a=Adapter(data,graph,'cpu',ModelConfig(hidden=8,ncn_dim=4))
    a.model.encoder.__class__=AttentionEncoder;a.model.encoder.configure(scope,alpha);a.ps.__class__=AttentionPairStruct
    a.model.eval();return a


def scalar(a,pair,mu,sd):
    enc=a.model.encoder;z=enc.forward_dense(*a.dense,*pair,a.ps.qvals(*pair))
    pool=a.ps.qncn(*pair,{'miRNA':z[0],'drug':z[1]},z[4])
    qis=(a.ps.qis(*pair)-mu)/sd;ease=a.ps.ease(*pair)
    score=a.model(pair,a.ps,qis,ease,a.dense)
    return F.softplus(score).sum()+.01*sum(v.square().sum() for v in [z[2],z[3],*pool]),score


def gradient_checks():
    old=torch.get_default_dtype();torch.set_default_dtype(torch.float64);records=[]
    try:
        for scope in ('target','all'):
            for alpha in (0.,.5):
                seed_all(123);graph,data=toy_graph();a=toy_adapter(graph,data,scope,alpha)
                with torch.no_grad():
                    for gates in [a.model.encoder.gates1,a.model.encoder.gates2]:
                        for gate in gates.values():
                            if gate.enabled:gate.score.weight.normal_(std=.2)
                for r,d in [(0,0),(3,1),(4,3)]:
                    pair=torch.tensor([[r],[d]])
                    deleted=graph.clone()
                    for edge,first,second in [(('miRNA','MiDrug','drug'),r,d),(('drug','rev_MiDrug','miRNA'),d,r)]:
                        ix=deleted[edge].edge_index;deleted[edge].edge_index=ix[:,~((ix[0]==first)&(ix[1]==second))]
                    keep=~((data['train_pairs'][:,0]==r)&(data['train_pairs'][:,1]==d))
                    other={**data,'train_pairs':data['train_pairs'][keep],'train_y':data['train_y'][keep]}
                    b=toy_adapter(deleted,other,scope,alpha);b.model.load_state_dict(a.model.state_dict());b.model.eval()
                    loss1,score1=scalar(a,pair,a.mu,a.sd);loss2,score2=scalar(b,pair,a.mu,a.sd)
                    torch.testing.assert_close(score1,score2,atol=1e-9,rtol=1e-8)
                    torch.testing.assert_close(loss1,loss2,atol=1e-9,rtol=1e-8)
                    pa=[p for p in a.model.parameters() if p.requires_grad];pb=[p for p in b.model.parameters() if p.requires_grad]
                    ga=torch.autograd.grad(loss1,pa,allow_unused=True);gb=torch.autograd.grad(loss2,pb,allow_unused=True)
                    max_grad=0.
                    for va,vb in zip(ga,gb):
                        assert (va is None)==(vb is None)
                        if va is not None:
                            torch.testing.assert_close(va,vb,atol=2e-8,rtol=2e-7)
                            max_grad=max(max_grad,float(abs(va-vb).max()))
                    records.append(dict(scope=scope,alpha=alpha,pair=[r,d],score_error=float(abs(score1-score2).max()),max_gradient_error=max_grad))
    finally:torch.set_default_dtype(old)
    return records


def main():
    gradients=gradient_checks();print('PASS exact scores and autograd',len(gradients),flush=True)
    data=load_inner(0);seed_all(42);base=make_adapter(data,'control','cuda:0');base.model.eval()
    seed_all(42);identity=make_adapter(data,'identity','cuda:0');identity.model.eval()
    for name,value in base.model.state_dict().items():torch.testing.assert_close(value,identity.model.state_dict()[name],atol=0,rtol=0)
    # A selection of positives and nonedges avoids a large identity-only autograd graph.
    pair=base.pairs['train'][:,:128]
    with torch.no_grad():
        p0=base.model(pair,base.ps,base.qis['train'][:128],base.ease['train'][:128],base.dense)
        p1=identity.model(pair,identity.ps,identity.qis['train'][:128],identity.ease['train'][:128],identity.dense)
    torch.testing.assert_close(p0,p1,atol=2e-4,rtol=5e-5)
    zero_error=float(abs(p0-p1).max());del base,identity;gc.collect();torch.cuda.empty_cache()
    reference=read_module('independent_degree_reference',DEGREE/'model.py')
    seed_all(42);degree=reference.make_adapter(data,'all05','cuda:0');degree.model.eval()
    seed_all(42);scaled=make_adapter(data,'all05','cuda:0');scaled.model.eval()
    with torch.no_grad():
        pd=degree.model(pair,degree.ps,degree.qis['train'][:128],degree.ease['train'][:128],degree.dense)
        ps=scaled.model(pair,scaled.ps,scaled.qis['train'][:128],scaled.ease['train'][:128],scaled.dense)
    torch.testing.assert_close(pd,ps,atol=2e-4,rtol=5e-5)
    degree_identity_error=float(abs(pd-ps).max());del degree,scaled;gc.collect();torch.cuda.empty_cache()
    # Strongest memory configuration: both layers/all relations and reconstruction.
    seed_all(42);a=make_adapter(data,'all_rec10','cuda:0');m=a.model
    torch.cuda.reset_peak_memory_stats();started=time.time();m.train()
    opt=torch.optim.RMSprop(m.parameters(),lr=1e-4);opt.zero_grad()
    bce=F.binary_cross_entropy_with_logits(a.logits('train'),a.y)
    rec=m.reconstruction;indices=rec.masks(a.graph,.5,mask_generator('cuda:0',0))
    aux,_=rec.loss(m.encoder,a.graph,indices,False);loss=bce+10*aux;loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    gate_norm=sum(float(g.score.weight.grad.norm()) for layer in [m.encoder.gates1,m.encoder.gates2] for g in layer.values())
    assert gate_norm>0;opt.step();torch.cuda.synchronize()
    elapsed=time.time()-started;peak=torch.cuda.max_memory_allocated()/2**30
    geometry=m.encoder.events.get(*a.pairs['train'],a.ps.qvals(*a.pairs['train']))
    counts={str(e):dict(events=len(v['query']),receiver_groups=len(v['group_query'])) for e,v in geometry.items()}
    # Deliberately nonuniform scores before all real-data deletion checks.
    with torch.no_grad():
        for layer in [m.encoder.gates1,m.encoder.gates2]:
            for g in layer.values():g.score.weight.normal_(std=.1)
    rows=deletion_check(a,data,'all_rec10')
    m.eval();changed=make_adapter({**data,'val_y':1-data['val_y']},'all_rec10','cuda:0')
    changed.model.load_state_dict(m.state_dict());changed.model.eval()
    with torch.no_grad():pa,pb=a.predict('val'),changed.predict('val')
    np.testing.assert_allclose(pa,pb,atol=2e-5,rtol=2e-5)
    report=dict(status='passed',gradient_checks=gradients,zero_attention_max_logit_error=zero_error,
        alpha05_uniform_reference_error=degree_identity_error,
        full_batch_peak_cuda_gib=peak,first_training_step_seconds=elapsed,gate_gradient_norm=gate_norm,
        event_geometry=counts,explicit_deletions=rows,heldout_label_flip_error=float(abs(pa-pb).max()),
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    (HERE/'initial_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
