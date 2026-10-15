"""Independent token-loop output/gradient oracle and full graph information-boundary checks."""
import ast,json,math,time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from common import HERE,DEGREE,load_inner,seed_all,sha256
from model import DrugTokens,CONFIGS,make_adapter,make_optimizer,mask_generator
QUERY_CHECK_SOURCE=DEGREE/'check.py'
tree=ast.parse(QUERY_CHECK_SOURCE.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='deletion_check')
ns=dict(torch=torch,np=np,make_adapter=make_adapter);exec(compile(ast.Module(body=[node],type_ignores=[]),str(QUERY_CHECK_SOURCE),'exec'),ns)
deletion_check=ns['deletion_check']
def seq(layers,x):
 for layer in layers:
  if isinstance(layer,nn.Linear):x=F.linear(x,layer.weight,layer.bias)
  elif isinstance(layer,nn.LayerNorm):x=F.layer_norm(x,layer.normalized_shape,layer.weight,layer.bias,layer.eps)
  elif isinstance(layer,nn.LeakyReLU):x=torch.where(x>=0,x,.1*x)
  else:raise TypeError(type(layer))
 return x

def oracle(m,z,mask,xr,r,d):
 q=seq(m.query,xr[r]);qa=torch.stack([seq(m.query,x) for x in xr]).mean(0) if m.mode=='population' else q
 keys=[];values=[];scores=[]
 for j in torch.where(mask[d])[0].tolist():
  x=F.layer_norm(z[d,j],m.norm.normalized_shape,m.norm.weight,m.norm.bias,m.norm.eps)
  k=F.linear(x,m.key.weight);v=F.linear(x,m.value.weight)
  keys.append(k);values.append(v);scores.append((k*qa).sum()/math.sqrt(m.hidden))
 w=torch.stack(scores).softmax(0) if m.mode!='pooled' else q.new_ones(len(scores))/len(scores)
 context=sum(a*b for a,b in zip(w,values));km=torch.stack(keys).mean(0)
 return seq(m.readout,torch.cat([context,q*km]))

def toy_checks():
 records=[]
 for mode in CONFIGS:
  seed_all(51);m=DrugTokens(6,7,4,mode,2).double()
  z=torch.randn(3,5,6,dtype=torch.double);mask=torch.tensor([[1,1,0,0,0],[1,1,1,1,0],[1,1,1,1,1]],dtype=torch.bool)
  xr=torch.randn(4,7,dtype=torch.double);r=torch.tensor([0,2,3]);d=torch.tensor([0,1,2])
  actual=m(z,mask,xr,r,d);ref=torch.stack([oracle(m,z,mask,xr,int(i),int(j)) for i,j in zip(r,d)])
  torch.testing.assert_close(actual,ref,atol=1e-10,rtol=1e-10)
  c=torch.linspace(-.9,.8,actual.numel(),dtype=torch.double).reshape_as(actual);params=list(m.parameters())
  g1=torch.autograd.grad((actual*c).sum(),params);g2=torch.autograd.grad((ref*c).sum(),params)
  for a,b in zip(g1,g2):torch.testing.assert_close(a,b,atol=2e-9,rtol=2e-9)
  perm=torch.tensor([4,1,3,0,2]);poison=z.clone();poison[~mask]=12345.
  with torch.no_grad():
   shuffled=m(z[:,perm],mask[:,perm],xr,r,d);clean=m(poison,mask,xr,r,d)
   m.chunk_size=1;chunk=m(z,mask,xr,r,d)
  for x in [shuffled,clean,chunk]:torch.testing.assert_close(actual,x,atol=1e-10,rtol=1e-10)
  k,v,q,km,vm=m.encode(z,mask,xr);_,w=m.attend(k,v,q,km,vm,mask,r,d)
  assert (w[~mask[d]]==0).all();torch.testing.assert_close(w.sum(1),torch.ones(3,dtype=torch.double),atol=1e-12,rtol=1e-12)
  records.append(dict(mode=mode,output_error=float(abs(actual-ref).max()),gradient_error=max(float(abs(a-b).max()) for a,b in zip(g1,g2)),padding_invariance=True,permutation_invariance=True,chunk_invariance=True))
 return records

@torch.no_grad()
def branch_check(a):
 m=a.model.token_context;z=a.model.drug_tokens;mask=a.model.drug_mask;xr=a.dense[0];r,d=a.pairs['val'][:,:3]
 actual=m(z,mask,xr,r,d);ref=torch.stack([oracle(m,z,mask,xr,int(i),int(j)) for i,j in zip(r,d)])
 torch.testing.assert_close(actual,ref,atol=2e-4,rtol=5e-5)
 k,v,q,km,vm=m.encode(z,mask,xr);_,w=m.attend(k,v,q,km,vm,mask,r,d)
 assert (w[~mask[d]]==0).all();torch.testing.assert_close(w.sum(1),torch.ones(3,device=w.device),atol=2e-6,rtol=2e-6)
 # Local attention weights are invariant across candidate miRNAs in both controls.
 rr=torch.tensor([0,1],device=r.device);dd=d[0].expand(2);_,ww=m.attend(k,v,q,km,vm,mask,rr,dd)
 if m.mode!='mirna':torch.testing.assert_close(ww[0],ww[1],atol=1e-7,rtol=1e-7)
 return dict(oracle_error=float(abs(actual-ref).max()),attention_candidate_delta=float(abs(ww[0]-ww[1]).max()),mask_zero=True)

@torch.no_grad()
def batch_check(a):
 a.model.eval();whole=a.logits('val')[:3]
 pieces=torch.cat([a.model(a.pairs['val'][:,i:i+1],a.ps,a.qis['val'][i:i+1],a.ease['val'][i:i+1],a.dense) for i in range(3)])
 torch.testing.assert_close(whole,pieces,atol=2e-4,rtol=5e-5);return float(abs(whole-pieces).max())

def main():
 toy=toy_checks();data=load_inner(0);records=[];counts={}
 for name,cfg in CONFIGS.items():
  seed_all(42);plain=make_adapter(data,'unmasked10','cuda:0')
  seed_all(42);a=make_adapter(data,name,'cuda:0');m=a.model
  for k,v in plain.model.state_dict().items():torch.testing.assert_close(v,m.state_dict()[k],atol=0,rtol=0)
  m.eval();plain.model.eval()
  with torch.no_grad():
   p0,p1=plain.predict('val'),a.predict('val');m.token_projection.weight.normal_(std=.005)
  np.testing.assert_allclose(p0,p1,atol=2e-5,rtol=2e-5)
  m.train();opt=make_optimizer(m,cfg);opt.zero_grad();start=time.time();torch.cuda.reset_peak_memory_stats()
  bce=F.binary_cross_entropy_with_logits(a.logits('train'),a.y)
  indices=m.reconstruction.masks(a.graph,.5,mask_generator('cuda:0',20261006));rec,_=m.reconstruction.loss(m.encoder,a.graph,indices,False)
  loss=bce+10*rec;loss.backward();grads={}
  for label,module in m.token_context.named_children():
   for p in module.parameters():assert p.grad is not None and torch.isfinite(p.grad).all()
   norm=sum(float(p.grad.square().sum()) for p in module.parameters())**.5;assert norm>0,label;grads[label]=norm
  opt.step();torch.cuda.synchronize();elapsed=time.time()-start;peak=torch.cuda.max_memory_allocated()/2**30
  assert int(m.eb.num_batches_tracked)==1 and not m.lin1._forward_hooks
  assert set(a.graph.node_types)=={'miRNA','drug'} and all('lnc' not in k.lower() for k in m.state_dict())
  assert not m.drug_tokens.requires_grad
  counts[name]=sum(p.numel() for p in m.token_context.parameters())+m.token_projection.weight.numel()
  m.eval();records.append(dict(config=name,initial_prediction_error=float(abs(p0-p1).max()),nonzero_gradients=grads,
   elapsed_seconds=elapsed,peak_cuda_gib=peak,branch=branch_check(a),batch_error=batch_check(a),explicit_deletions=deletion_check(a,data,name,count=4)))
  print('PASS',name,elapsed,peak,flush=True)
  del a,m,plain,opt,loss;torch.cuda.empty_cache()
 assert len(set(counts.values()))==1
 seed_all(42);a=make_adapter(data,'mirna','cuda:0')
 seed_all(42);b=make_adapter({**data,'val_y':1-data['val_y']},'population','cuda:0')
 for k,v in a.model.state_dict().items():torch.testing.assert_close(v,b.model.state_dict()[k],atol=0,rtol=0)
 b.model.token_context.mode='mirna'
 with torch.no_grad():a.model.token_projection.weight.normal_(std=.005)
 b.model.load_state_dict(a.model.state_dict());a.model.eval();b.model.eval()
 with torch.no_grad():pa,pb=a.predict('val'),b.predict('val')
 np.testing.assert_allclose(pa,pb,atol=2e-5,rtol=2e-5)
 out=dict(status='passed',toy=toy,records=records,matched_parameter_counts=counts,matched_initial_parameters=True,heldout_labels_invariant=True,
  heldout_label_error=float(abs(pa-pb).max()),query_check_source_sha256=sha256(QUERY_CHECK_SOURCE),no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
 (HERE/'initial_checks.json').write_text(json.dumps(out,indent=2));print('ALL PASSED',flush=True)
if __name__=='__main__':main()
