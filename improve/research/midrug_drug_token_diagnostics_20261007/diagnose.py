"""Post-selection fixed-model interventions, no metric selection or training."""
import sys,json,math
from pathlib import Path
HERE=Path(__file__).resolve().parent
EXP=HERE.parent/'midrug_drug_tokens_20261007'
sys.path.insert(0,str(EXP))
import torch
import numpy as np
from common import load_inner,seed_all,sha256
from model import make_adapter,CONFIGS

def main():
 plan=json.loads((HERE/'plan.json').read_text())
 for p,h in plan['hashes'].items():assert sha256(p)==h,p
 selection=json.loads((EXP/'selection.json').read_text());assert not selection['promote_to_outer_validation']
 rows=[]
 for name in CONFIGS:
  for fold in range(3):
   p=EXP/'results'/f'{name}_f{fold}_s42.pt';seed_all(42);a=make_adapter(load_inner(fold),name,'cuda:0');a.model.load_state_dict(torch.load(p,map_location='cuda:0',weights_only=True));a.model.eval();m=a.model.token_context
   with torch.no_grad():
    prob=a.predict('val');enc=m.encode(a.model.drug_tokens,a.model.drug_mask,a.dense[0]);r,d=a.pairs['val']
    ent=[];divergence=[];dynamic=[];lengths=[]
    original=m.mode
    for s in range(0,len(r),256):
     rr,dd=r[s:s+256],d[s:s+256];mask=a.model.drug_mask
     actual,w=m.attend(*enc,mask,rr,dd)
     try:
      m.mode='mirna';_,wc=m.attend(*enc,mask,rr,dd)
      m.mode='population';_,wm=m.attend(*enc,mask,rr,dd)
     finally:m.mode=original
     entropy=-(w*w.clamp_min(1e-30).log()).sum(-1);count=mask[dd].sum(-1)
     ent.extend(entropy.cpu().tolist());lengths.extend(count.cpu().tolist())
     divergence.extend((wc*(wc.clamp_min(1e-30).log()-wm.clamp_min(1e-30).log())).sum(-1).cpu().tolist())
     dynamic.extend((wc-wm).abs().sum(-1).cpu().tolist())
    try:
     m.mode='population';population=a.predict('val')
     m.mode='pooled';pooled=a.predict('val')
    finally:m.mode=original
    projection=a.model.token_projection.weight.clone()
    try:
     a.model.token_projection.weight.zero_();zero=a.predict('val')
    finally:a.model.token_projection.weight.copy_(projection)
    def without_global(module,args):
     x=args[0].clone();x[:,m.hidden:]=0;return (x,)
    hook=m.readout.register_forward_pre_hook(without_global)
    try:no_global=a.predict('val')
    finally:hook.remove()
    restored=a.predict('val');np.testing.assert_allclose(prob,restored,atol=2e-5,rtol=2e-5)
   e=np.array(ent);n=np.array(lengths);normalized=e[n>1]/np.log(n[n>1]);effective=np.exp(e)
   row=dict(config=name,fold=fold,checkpoint_sha256=sha256(p),mean_valid_tokens=float(n.mean()),mean_effective_tokens=float(effective.mean()),
    normalized_entropy_mean=float(normalized.mean()),conditional_vs_population_attention_l1=float(np.mean(dynamic)),conditional_vs_population_kl=float(np.mean(divergence)),
    population_probability_shift=float(abs(prob-population).mean()),pooled_probability_shift=float(abs(prob-pooled).mean()),zero_projection_probability_shift=float(abs(prob-zero).mean()),
    zero_global_probability_shift=float(abs(prob-no_global).mean()),restored_probability_error=float(abs(prob-restored).max()))
   rows.append(row);print('PASS',name,fold,flush=True);del a,m;torch.cuda.empty_cache()
 mean=[dict(config=n,**{k:float(np.mean([r[k] for r in rows if r['config']==n])) for k in rows[0] if k not in ['config','fold','checkpoint_sha256']}) for n in CONFIGS]
 out=dict(status='passed',records=rows,means=mean,plan_sha256=sha256(HERE/'plan.json'),selection_sha256=sha256(EXP/'selection.json'),no_new_fit=True,no_new_metric_selection=True,outer_validation_evaluated=False,test_evaluated=False)
 (HERE/'results.json').write_text(json.dumps(out,indent=2))
 table='\n'.join(f"| {r['config']} | {r['normalized_entropy_mean']:.6f} | {r['conditional_vs_population_attention_l1']:.6f} | {r['population_probability_shift']:.6f} | {r['pooled_probability_shift']:.6f} | {r['zero_projection_probability_shift']:.6f} | {r['zero_global_probability_shift']:.6f} |" for r in mean)
 report=f'''# 药物token分支诊断（冻结选择后）

9个固定模型，未训练、未计算任何干预后的AUC/AP，不改变原失败选择。
仅内部留出配对，原验证/测试未使用。干预后恢复模型并重放预测，全部通过。

| 配置 | 归一熵 | 条件vs共享权重L1 | 共享query概率变化 | 均值汇总概率变化 | 零分支概率变化 | 零全局交互概率变化 |
|---|---:|---:|---:|---:|---:|---:|
{table}

概率变化均为配对绝对差平均，再平均三折。条件vs共享权重比较使用每个模型自身固定
参数；对population/pooled亦可计算虚拟条件权重，但不把它作为新模型选型。
归一熵只对有效长度>1的配对计算，1为均匀。平均有效token数量由exp(熵)定义。
零全局交互在readout之前把q⊙mean(k)那一半置零，会影响其后的LayerNorm，
因此是整体消融，不能解释为可加因果效应。零分支不撤销主干已受到的联合训练。

本轮候选三折选择轮数56/71/169，比unmasked10的244/512/327早，但单凭早停和
以上消融不能证明过拟合来源。诊断只确认分支实际作用及注意力变化程度；
不据此调整温度/宽度/头数或提升对照为候选。
'''
 (HERE/'REPORT.md').write_text(report)
 (HERE/'archive_manifest.json').write_text(json.dumps(dict(artifact_hashes={p.name:sha256(p) for p in HERE.iterdir() if p.is_file() and p.name not in ['archive_manifest.json','archive_recheck.json','run.log']},frozen_dependencies=plan['hashes']),indent=2))
 print(json.dumps(mean,indent=2))
if __name__=='__main__':main()
