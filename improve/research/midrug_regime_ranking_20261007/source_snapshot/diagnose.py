"""Frozen-checkpoint comparison-set errors and score-gradient allocation."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,load_inner,seed_all,sha256
from model import CONFIGS,make_adapter,ranking


def conditional_auc(labels,prob,groups,kind):
    error=0.;count=0
    for group in range(4):
        p=prob[(labels==1)&(groups==group)]
        take=labels==0
        if kind=='within':take=take&(groups==group)
        elif kind=='cross':take=take&(groups!=group)
        n=np.sort(prob[take]);lo=np.searchsorted(n,p,'left');hi=np.searchsorted(n,p,'right')
        error+=(len(n)-hi+.5*(hi-lo)).sum();count+=len(n)*len(p)
    return dict(auc=1-error/count if count else None,count=count,error_mass=float(error))


def score_gradient(logits,labels,groups,kind):
    s=logits.detach().double().requires_grad_();y=labels.double()
    bce=F.binary_cross_entropy_with_logits(s,y);rank=.1*ranking(s,y,groups,kind)
    b=torch.autograd.grad(bce,s)[0];r=torch.autograd.grad(rank,s)[0]
    bygroup={}
    for name,value in [('bce',b),('weighted_rank',r)]:
        bygroup[name]=[float(value[groups==i].abs().sum()/value.abs().sum()) for i in range(4)]
    return dict(weighted_rank_norm_over_bce=float(r.norm()/b.norm()),cosine=float((r@b)/(r.norm()*b.norm())),
                group_l1_gradient_fraction=bygroup)


def main():
    frozen=sha256(HERE/'selection.json');assert json.loads((HERE/'verification.json').read_text())['status']=='passed'
    records=[]
    for fold in range(3):
        data=load_inner(fold)
        for name in CONFIGS:
            path=HERE/'results'/f'{name}_f{fold}_s42.pt';meta=json.loads(path.with_suffix('.json').read_text());assert sha256(path)==meta['checkpoint_sha256']
            seed_all(42);a=make_adapter(data,name,'cuda:0');a.model.load_state_dict(torch.load(path,map_location='cuda:0',weights_only=True));a.model.eval()
            before={k:v.detach().clone() for k,v in a.model.state_dict().items()};row=dict(config=name,fold=fold)
            for part in ['train','val']:
                with torch.no_grad():logits=a.logits(part);prob=logits.sigmoid().cpu().numpy()
                labels=data[part+'_y'];r,d=a.pairs[part];q=a.ps.qvals(r,d)
                groups=((a.ps.deg_r[r]-q>a.rank_cuts[0]).long()*2+(a.ps.deg_d[d]-q>a.rank_cuts[1]).long()).cpu().numpy()
                if part=='train':np.testing.assert_array_equal(groups,a.rank_groups.cpu().numpy())
                cond={k:conditional_auc(labels,prob,groups,k) for k in CONFIGS}
                assert abs(cond['all']['auc']-roc_auc_score(labels,prob))<1e-12
                assert cond['all']['count']==cond['cross']['count']+cond['within']['count']
                assert cond['all']['error_mass']==cond['cross']['error_mass']+cond['within']['error_mass']
                row[part]=dict(auc=float(roc_auc_score(labels,prob)),ap=float(average_precision_score(labels,prob)),
                    bce=float(F.binary_cross_entropy_with_logits(logits,torch.as_tensor(labels,device=logits.device,dtype=logits.dtype))),conditional_auc=cond)
                if part=='train':row['training_score_gradients']=score_gradient(logits,a.y,a.rank_groups,name)
            for m in ['auc','ap']:assert abs(row['val'][m]-meta['inner_'+m])<2e-6
            for k,v in before.items():torch.testing.assert_close(v,a.model.state_dict()[k],atol=0,rtol=0)
            records.append(row);print('PASS',name,fold,flush=True);del a,before;torch.cuda.empty_cache()
    means=[]
    for name in CONFIGS:
        rows=[r for r in records if r['config']==name];item=dict(config=name)
        for part in ['train','val']:
            item[part]={m:float(np.mean([r[part][m] for r in rows])) for m in ['auc','ap','bce']}
            item[part]['conditional_auc']={k:float(np.mean([r[part]['conditional_auc'][k]['auc'] for r in rows])) for k in CONFIGS}
        item['weighted_rank_norm_over_bce']=float(np.mean([r['training_score_gradients']['weighted_rank_norm_over_bce'] for r in rows]))
        means.append(item)
    assert sha256(HERE/'selection.json')==frozen
    (HERE/'component_diagnostics.json').write_text(json.dumps(dict(status='passed',records=records,means=means,selection_sha256=frozen,
        selection_changed=False,no_parameter_or_buffer_updates=True,no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False),indent=2))
    table='\n'.join(f"| {r['config']} | {r['train']['auc']:.6f}/{r['train']['ap']:.6f} | {r['val']['auc']:.6f}/{r['val']['ap']:.6f} | {r['val']['conditional_auc']['cross']:.6f} | {r['val']['conditional_auc']['within']:.6f} | {r['weighted_rank_norm_over_bce']:.4f} |" for r in means)
    (HERE/'COMPONENT_DIAGNOSTICS.md').write_text(f'''# 排名集合选型冻结后的诊断

| 配置 | 训练全局AUC/AP | 内部全局AUC/AP | 跨组条件AUC | 组内条件AUC | .1排名梯度/BCE梯度范数 |
|---|---:|---:|---:|---:|---:|
{table}

条件AUC的分母是各自集合中正负比较数，与全局AUC不同；JSON保存数目和逆序质量，
两集合之和精确还原全局。梯度是在已选模型eval模式训练logits上的分数梯度，
不是模型参数梯度或历史RMSprop步长，不据此声称训练因果解释。全部参数、缓冲
与冻结选型保持不变；没有按条件AUC或梯度重新选择配置、权重或温度。
''')


if __name__=='__main__':main()
