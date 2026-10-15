"""Frozen full-ensemble versus individual-member diagnostics; no subset selection."""
import json
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,load_inner,seed_all,sha256
from model import CONFIGS,make_adapter,member_logits,classification_loss


def metrics(y,p):
    return dict(auc=float(roc_auc_score(y,p)),ap=float(average_precision_score(y,p)))


@torch.no_grad()
def examine(a,data,part):
    logits=member_logits(a,part);prob=logits.sigmoid();mean=prob.mean(1)
    y=torch.as_tensor(data[part+'_y'],dtype=logits.dtype,device=logits.device)
    individual=[metrics(data[part+'_y'],prob[:,i].cpu().numpy()) for i in range(a.model.members)]
    votes=(prob>=.5).to(prob.dtype).mean(1)
    return dict(ensemble=metrics(data[part+'_y'],mean.cpu().numpy()),
                individual=individual,mean_individual={k:float(np.mean([r[k] for r in individual])) for k in ['auc','ap']},
                fixed_first_member=individual[0],mean_logit_prediction=metrics(data[part+'_y'],logits.mean(1).sigmoid().cpu().numpy()),
                mean_individual_bce=float(classification_loss(logits,y,'member_bce')),
                mixture_bce=float(classification_loss(logits,y,'mixture_bce')),
                mean_member_probability_std=float(prob.std(1,unbiased=False).mean()),
                pair_vote_disagreement=float((2*votes*(1-votes)).mean()),
                mean_absolute_first_member_difference=float(abs(prob[:,0]-mean).mean())),mean.cpu().numpy()


def main():
    selection_hash=sha256(HERE/'selection.json');plan=json.loads((HERE/'results/plan.json').read_text())
    assert json.loads((HERE/'verification.json').read_text())['status']=='passed'
    rows=[]
    for fold in range(3):
        data=load_inner(fold)
        for name in CONFIGS:
            path=HERE/'results'/f'{name}_f{fold}_s42.pt';metadata=json.loads(path.with_suffix('.json').read_text())
            assert sha256(path)==metadata['checkpoint_sha256'];seed_all(42);a=make_adapter(data,name,'cuda:0')
            a.model.load_state_dict(torch.load(path,map_location='cuda:0',weights_only=True));a.model.eval()
            before={k:v.detach().clone() for k,v in a.model.state_dict().items()}
            tr,_=examine(a,data,'train');va,prob=examine(a,data,'val')
            with np.load(path.with_name(path.stem+'_inner.npz')) as z:np.testing.assert_allclose(prob,z['probability'],atol=2e-5,rtol=2e-5)
            for key,value in before.items():torch.testing.assert_close(value,a.model.state_dict()[key],atol=0,rtol=0)
            rows.append(dict(config=name,fold=fold,checkpoint_sha256=sha256(path),train=tr,val=va));print('PASS',name,fold,flush=True)
            del a,before;torch.cuda.empty_cache()
    means=[]
    for name in CONFIGS:
        records=[r for r in rows if r['config']==name];item=dict(config=name)
        for part in ['train','val']:
            item[part]={}
            for key in ['ensemble','mean_individual','fixed_first_member','mean_logit_prediction']:
                item[part][key]={metric:float(np.mean([r[part][key][metric] for r in records])) for metric in ['auc','ap']}
            for key in ['mean_individual_bce','mixture_bce','mean_member_probability_std','pair_vote_disagreement','mean_absolute_first_member_difference']:
                item[part][key]=float(np.mean([r[part][key] for r in records]))
        means.append(item)
    assert sha256(HERE/'selection.json')==selection_hash
    for path,digest in plan['source_hashes'].items():assert sha256(path)==digest
    result=dict(status='passed',records=rows,means=means,selection_sha256=selection_hash,selection_changed=False,
                no_parameter_or_buffer_updates=True,no_member_subset_selection=True,outer_validation_evaluated=False,test_evaluated=False,
                limitation='A fixed first member and averaging logits are post-training diagnostics, not independently trained controls or new candidate choices.')
    (HERE/'component_diagnostics.json').write_text(json.dumps(result,indent=2))
    table='\n'.join(f"| {r['config']} | {r['val']['ensemble']['auc']:.9f}/{r['val']['ensemble']['ap']:.9f} | {r['val']['mean_individual']['auc']:.9f}/{r['val']['mean_individual']['ap']:.9f} | {r['val']['fixed_first_member']['auc']:.9f}/{r['val']['fixed_first_member']['ap']:.9f} | {r['val']['mean_member_probability_std']:.6f} | {r['val']['mean_individual_bce']:.6f}/{r['val']['mixture_bce']:.6f} |" for r in means)
    (HERE/'COMPONENT_DIAGNOSTICS.md').write_text(f'''# 冻结选型后的多成员预测头诊断

相同检查点，报告全部 16 个成员，不按留出成绩选择成员子集。
单成员指标先分别计算再取平均，不与平均概率指标混同。

| 训练目标 | 整体 AUC/AP | 平均个体 AUC/AP | 固定第一个成员 AUC/AP | 平均成员概率标准差 | 个体 BCE 均值/混合 BCE |
|---|---:|---:|---:|---:|---:|
{table}

JSON 同时记录训练配对的相同指标、按成员完整列表、投票分歧及平均 logits 的
预测指标。固定第一个成员/平均 logits 只作事后诊断，不是重新训练的对照，
不用于再次挑选候选。全部参数/缓冲逐项保持不变，冻结选择哈希不变。
''')

if __name__=='__main__':main()
