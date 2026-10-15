"""Freeze candidate decision, then diagnose head/branch contributions and archive."""
import csv,json,shutil
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,average_precision_score
from common import HERE,RELEASE,load_inner,seed_all,sha256
from model import make_adapter,source_record,HEAD_PREFIXES

def write(path,value):path.write_text(json.dumps(value,indent=2))
def scores(y,p):return dict(auc=roc_auc_score(y,p),ap=average_precision_score(y,p))

def main():
    plan=json.loads((HERE/'results/plan.json').read_text());v=json.loads((HERE/'verification.json').read_text())
    assert v['status']=='passed' and v['new_checkpoint_replays']==9 and v['final_state_inspections']==9
    assert len(v['records'])==21 and v['immutable_reused_control_checks']==12
    dependencies={**plan['source_hashes'],**plan['evidence_hashes']}
    for p,h in dependencies.items():assert sha256(p)==h
    runs={n:[json.loads(Path(c[str(f)]['path']).read_text()) for f in range(3)] for n,c in plan['controls'].items()}
    runs.update({n:[json.loads((HERE/'results'/f'{n}_f{f}_s42.json').read_text()) for f in range(3)] for n in plan['configs']})
    rows=[dict(config=n,mean_inner_auc=float(np.mean([r['inner_auc'] for r in rs])),mean_inner_ap=float(np.mean([r['inner_ap'] for r in rs]))) for n,rs in runs.items()]
    assert plan['eligible_candidates']==['full_head'];best=next(r for r in rows if r['config']=='full_head')
    best['comparisons']={n:{m:dict(mean_delta=float(np.mean([r['inner_'+m]-b['inner_'+m] for r,b in zip(runs['full_head'],runs[n])])),
        fold_deltas=[r['inner_'+m]-b['inner_'+m] for r,b in zip(runs['full_head'],runs[n])]) for m in ['auc','ap']} for n in plan['comparators']}
    best['selection_score']=min(c[m]['mean_delta'] for c in best['comparisons'].values() for m in ['auc','ap'])
    promote=best['selection_score']>=plan['promotion_minimum_both_gains_over_each_control']
    selection=dict(status='frozen_inner_selection',rows=rows,best=best,promote_to_outer_validation=promote,
        plan_sha256=sha256(HERE/'results/plan.json'),verification_sha256=sha256(HERE/'verification.json'),
        result_hashes={p.name:sha256(p) for p in sorted((HERE/'results').glob('*_s42.json'))},
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False)
    sp=HERE/'selection.json'
    if sp.exists():assert json.loads(sp.read_text())==selection
    else:write(sp,selection)
    diagnostics=[]
    for name in plan['configs']:
        for fold in range(3):
            r=runs[name][fold];seed_all(42);data=load_inner(fold);a=make_adapter(data,name,'cuda:0');m=a.model
            m.load_state_dict(torch.load(HERE/'results'/f"{r['tag']}.pt",map_location='cuda:0',weights_only=True));m.eval()
            source,_=source_record(fold);original=torch.load(source.with_suffix('.pt'),map_location='cuda:0',weights_only=True)
            with np.load(HERE/'replay_predictions'/f"{r['tag']}.npz") as z:train,val=z['train'],z['val']
            d=dict(config=name,fold=fold,best_epoch=r['best_epoch'],epochs_run=r['epochs_run'],train=scores(data['train_y'],train),val=scores(data['val_y'],val))
            curve=json.loads((HERE/'results'/f"{r['tag']}_curves.json").read_text())
            d['final_inner_from_training_curve']={k:curve[-1]['inner_'+k] for k in ['auc','ap']}
            d['final_training_bce_before_update']=curve[-1]['loss']
            head=[(p,original[n]) for n,p in m.named_parameters() if n.startswith(HEAD_PREFIXES)]
            d['head_relative_parameter_change']=float((sum((p-o).square().sum() for p,o in head)/sum(o.square().sum() for p,o in head)).sqrt())
            if hasattr(m,'feedback'):
                with torch.no_grad():
                    addition=m.feedback_projection(m.feedback(*a.dense[:2],a.pairs['val']))
                    saved=m.feedback_projection.weight.clone();m.feedback_projection.weight.zero_()
                    no_branch=a.predict('val');m.feedback_projection.weight.copy_(saved)
                d['val_added_lin1_rms']=float(addition.square().mean().sqrt())
                d['zero_projection_intervention']=scores(data['val_y'],no_branch)
                d['trained_minus_zero_projection']={k:d['val'][k]-d['zero_projection_intervention'][k] for k in ['auc','ap']}
                d['zero_projection_logit_rms']=float((torch.logit(torch.as_tensor(val).clamp(1e-6,1-1e-6))-torch.logit(torch.as_tensor(no_branch).clamp(1e-6,1-1e-6))).square().mean().sqrt())
            else:d['val_added_lin1_rms']=0.
            diagnostics.append(d);del a,m,original;torch.cuda.empty_cache()
    means={n:dict(train_auc=float(np.mean([r['train']['auc'] for r in diagnostics if r['config']==n])),
                 train_ap=float(np.mean([r['train']['ap'] for r in diagnostics if r['config']==n])),
                 final_inner_auc=float(np.mean([r['final_inner_from_training_curve']['auc'] for r in diagnostics if r['config']==n])),
                 final_inner_ap=float(np.mean([r['final_inner_from_training_curve']['ap'] for r in diagnostics if r['config']==n])),
                 head_relative_parameter_change=float(np.mean([r['head_relative_parameter_change'] for r in diagnostics if r['config']==n])),
                 val_added_lin1_rms=float(np.mean([r['val_added_lin1_rms'] for r in diagnostics if r['config']==n]))) for n in plan['configs']}
    write(HERE/'component_diagnostics.json',dict(status='passed',selection_sha256=sha256(sp),records=diagnostics,means=means,
        scope='Postselection parameter and zero-projection intervention diagnostics; no retraining or coefficient search.',no_further_selection=True))
    flat=[dict(config=n,fold=f,**{k:r[k] for k in ['inner_auc','inner_ap','best_epoch','epochs_run']}) for n,rs in runs.items() for f,r in enumerate(rs)]
    with (HERE/'inner_summary.csv').open('w') as out:
        writer=csv.DictWriter(out,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    original=HERE.parent/'midrug_internal_20261006/plmf_protection.json';protected=json.loads(original.read_text())
    for p,h in protected['source_hashes'].items():assert sha256(RELEASE/p)==h
    frozen=RELEASE/'data/frozen';manifest=json.loads((frozen/'manifest.json').read_text());refs=HERE.parent/'baselines_m123_20261004/results';protection=[]
    for r in protected['records']:
        ds,fold=r['dataset'],r['fold']
        assert sha256(refs/f'incrd_m123_{ds}_f{fold}_s42.pt')==r['checkpoint_sha256']
        assert sha256(frozen/f'{ds}_fold{fold}.npz')==r['split_sha256']
        assert sha256(frozen/f'{ds}_graph.pt')==manifest['datasets'][ds]['graph_sha256']
        protection.append(dict(dataset=ds,fold=fold,checkpoint_sha256=r['checkpoint_sha256'],split_sha256=r['split_sha256'],graph_sha256=manifest['datasets'][ds]['graph_sha256']))
    write(HERE/'plmf_hash_recheck.json',dict(status='passed',records=protection,source_unchanged=True,prior_full_replay_sha256=sha256(original),new_gpu_replays=0))
    table='\n'.join(f"| {r['config']} | {r['mean_inner_auc']:.9f} | {r['mean_inner_ap']:.9f} |" for r in rows)
    paired='\n'.join(f"| {ref} | {c['auc']['mean_delta']:+.9f} | {c['ap']['mean_delta']:+.9f} |" for ref,c in best['comparisons'].items())
    stages={n:[r['best_epoch'] for r in runs[n]] for n in plan['configs']}
    new=[r for r in v['records'] if not r['reused_control']]
    deleted=max(x['errors']['full_logit'] for r in new for x in r['explicit_deletions'])
    diagtable='\n'.join(f"| {n} | {r['train_auc']:.6f}/{r['train_ap']:.6f} | {r['final_inner_auc']:.6f}/{r['final_inner_ap']:.6f} | {r['head_relative_parameter_change']:.6f} | {r['val_added_lin1_rms']:.6f} |" for n,r in means.items())
    report=f'''# 固定编码器的分类头适配：内部实验结果

**原测试 AUC/AP 同时绝对 +.02 尚未证实。** 完成9次新拟合和9个所选检查点完整预测重放，
检查9个最终状态及所选/最终优化器，另对12个旧模型引用不可变完整重放并核对哈希。
只用原训练内部三折 seed42，没有新原验证或测试评估。

| 配置 | 内部 AUC | 内部 AP |
|---|---:|---:|
{table}

| full_head 的规定对照 | ΔAUC | ΔAP |
|---|---:|---:|
{paired}

唯一候选 full_head 最小规定平均增益 {best['selection_score']:+.9f}；推进原验证：{promote}。
须相对全部六个对照的三折平均 AUC/AP 均≥+.002。不事后改选 mean_head 或 head_only。
最终仍须原测试三种子均值 AUC≥.9733929274189016、AP≥.933683373732677，PLMF 两集不降。

## 本轮检验和边界

复用前轮 [GeoF 两轮反馈局部适配](../midrug_feedback_geometry_20261007/README.md)，
模块源码逐字相同；本轮没有新论文搜索或新几何超参数。既有全模型微调失败不证明
分类头无法适配，本轮只放开原 lin1/lin2/lin3，固定编码器、M2/M3 投影、重构及所有缓冲。
full_head/mean_head 从同一原 unmasked10 权重及新分支相同初始化开始，投影初值零；
head_only 仅改变原分类头。三者没有从上轮已选反馈权重继续。

头 lr1e-5/wd0，新增分支 lr1e-4/wd1e-4；RMSprop alpha.99，空优化器，BCE，
set_to_none=True。max300/min100/patience100，含 stage0 的最早最大双指标均值选轮。
所选阶段轮次：{stages}。同一内折源模型已选轮，本轮不是独立验证。
预检真实更新后头改变，所有固定参数和 BN 缓冲逐位不变；九个所选/最终状态再核对。
所选预测、曲线、逐参数优化器步数、分支分块梯度及物理删图通过；
完整 M1/M2/M3 加分支的最终 logit 物理删查询最大误差 {deleted:.3g}。
相同反馈模块的独立 NumPy/有限差分数学核验引用前轮哈希，没有重复计算并虚报次数。

## 选型冻结后的诊断

| 配置 | 所选训练 AUC/AP | 最后一步内部 AUC/AP | 所选头参数相对位移 | 所选留出新增 lin1 激活 RMS |
|---|---:|---:|---:|---:|
{diagtable}

固定所选权重将新增投影置零的推理干预见 component_diagnostics.json；仅用于描述
当前头和反馈的共同作用，不作为新候选、权重搜索或独立证据。训练收益不能代替留出收益。
最后一步内部指标来自冻结训练曲线；最后状态检查不计为额外完整预测重放。
本轮固定训练配方的结果不证明所有头微调方法无效，也不能外推为性能上限。

没有 lncRNA、外部数据、注释、额外序列或新权重。PLMF 十个旧检查点、原输入和发布源码
哈希保持不变；本轮没有新 PLMF 训练或 GPU 重放。
'''
    (HERE/'REPORT.md').write_text(report)
    state=dict(goal_status='active',completion_proven=False,stage='completed_and_audited',best=best,promote_to_outer_validation=promote,
        new_fits=9,new_checkpoint_replays=9,final_state_inspections=9,reused_control_checks=12,no_lncrna=True,new_candidate_test_evaluations=0,
        processes_expected_live=[],component_diagnostics=means,next_action='Use the frozen matched head/feedback comparisons to decide a distinct next hypothesis; do not interpret freezing as an established bottleneck or retrospectively sweep head learning rates or feedback constants.')
    write(HERE/'CONTINUATION.json',state)
    progress=HERE.parent/'MIDRUG_DUAL_METRIC_PROGRESS.md';prefix=progress.read_text().split('最新完成')[0]
    progress.write_text(prefix+f'''最新完成[固定编码器的分类头适配](midrug_feedback_head_20261007/REPORT.md)：9次新训练与9个所选检查点预测重放，最终状态和优化器另查。
full_head 内部 AUC/AP 为 {best['mean_inner_auc']:.9f}/{best['mean_inner_ap']:.9f}，推进原验证：{promote}。
没有新原验证/测试成绩；PLMF 十个旧模型及冻结输入/发布源码哈希未变。
''')
    snapshot=HERE/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,snapshot/p.name)
    files=[p for p in HERE.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'source_snapshot' not in p.parts
        and p.name not in ['archive_manifest.json','archive_recheck.json','summarize.log']]
    archive=dict(status='archived',source_hashes={p.name:sha256(p) for p in sorted(HERE.glob('*.py'))},
        artifact_hashes={str(p.relative_to(HERE)):sha256(p) for p in sorted(files)},frozen_dependencies=dependencies)
    write(HERE/'archive_manifest.json',archive);count=0
    for p,h in archive['source_hashes'].items():assert sha256(HERE/p)==h and sha256(snapshot/p)==h;count+=2
    for p,h in archive['artifact_hashes'].items():assert sha256(HERE/p)==h;count+=1
    for p,h in archive['frozen_dependencies'].items():assert sha256(p)==h;count+=1
    write(HERE/'archive_recheck.json',dict(status='passed',hash_checks=count,new_checkpoint_replays=9,final_state_inspections=9,reused_control_checks=12,
        protected_plmf_records=10,selection_unchanged=True,archive_sha256=sha256(HERE/'archive_manifest.json'),selection_sha256=sha256(sp),
        no_lncrna=True,outer_validation_evaluated=False,test_evaluated=False))
    scopepath=HERE.parent/'ACTIVE_SCOPE.json';scope=json.loads(scopepath.read_text());scope.update(active_directory=str(HERE),
        latest_continuation=str(HERE/'CONTINUATION.json'),latest_archive_check=str(HERE/'archive_recheck.json'),
        latest_experiment_archive_check=str(HERE/'archive_recheck.json'),latest_research_review=str(HERE/'README.md'),
        latest_research_continuation=str(HERE/'CONTINUATION.json'),next_action=state['next_action']);write(scopepath,scope)
    print(json.dumps(dict(best=best,promote=promote,diagnostics=means),indent=2))

if __name__=='__main__':main()
