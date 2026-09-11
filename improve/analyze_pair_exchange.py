"""Verify inner selection/refits and report global and within-endpoint ranking."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from improve.analyze_expression_consensus import within_drug
from improve.expression_scale_mechanism import state_digest
from improve.first_round_features import sha256, write_json
from improve.network_feature_ablation import score
from improve.pair_exchange import (ARMS, TASKS, build, checked_result, load_config,
    load_selection, load_stage, pick_config, predict, result_dir)


def ranking_metrics(labels, pairs, predictions):
    auc, ap = score(labels, predictions)
    drug = within_drug(labels, pairs, predictions)
    rna = within_drug(labels, pairs[::-1], predictions)
    return dict(auc=auc, aupr=ap, criterion=(auc+ap)/2,
                macro_drug_auc=drug['macro_drug_auc'], macro_drug_aupr=drug['macro_drug_aupr'],
                eligible_drugs=drug['eligible_drugs'], drug_covered_pairs=drug['covered_pairs'],
                macro_rna_auc=rna['macro_drug_auc'], macro_rna_aupr=rna['macro_drug_aupr'],
                eligible_rnas=rna['eligible_drugs'], rna_covered_pairs=rna['covered_pairs'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = Path(args.out_dir)
    protocol = load_config(out)
    selection = load_selection(out, protocol)
    inner_rows, outer_rows, components, controls, replay_errors = [], [], [], [], []
    initial, encoders = {}, {}
    for task in TASKS:
        for stage in ('inner','outer'):
            arrays, graph = load_stage(out,protocol,task,stage)
            if stage == 'outer':
                degree = np.bincount(arrays['pairs_train'][1,arrays['y_train']==1],minlength=len(arrays['x_drug']))
                control = ranking_metrics(arrays['y_val'],arrays['pairs_val'],np.log1p(degree[arrays['pairs_val'][1]]))
                controls.append(dict(task=task,control='training_drug_degree',**control))
            for arm in ARMS:
                jobs = [(config['id'],protocol['tuning_seed'],config,protocol['training'][task]['train_epochs'])
                        for config in protocol['configurations']]
                if stage == 'outer':
                    choice = selection['choices'][task][arm]
                    cfg = next(config for config in protocol['configurations'] if config['id']==choice['config_id'])
                    jobs = [('s'+str(seed),seed,cfg,choice['epochs']) for seed in protocol['refit_seeds']]
                for identifier,seed,config,epochs in jobs:
                    directory = result_dir(out,task,arm,stage,identifier)
                    result = checked_result(directory,protocol)
                    for name,expected in dict(task=task,arm=arm,stage=stage,seed=seed,config_id=config['id'],trained_epochs=epochs).items():
                        if result[name] != expected:
                            raise ValueError('Mismatched result '+name)
                    if stage=='outer' and result['selection_sha256']!=selection['sha256']:
                        raise ValueError('Refit used a different selection')
                    with np.load(directory/'scores.npz') as source:
                        saved = {k:source[k].copy() for k in source.files}
                    expected_fields={'val','y_val','pairs_val'} | ({'marginal','interaction'} if arm.startswith('factor_') else set())
                    if set(saved) != expected_fields:
                        raise ValueError('Unexpected stored prediction fields')
                    for name in ('y_val','pairs_val'):
                        np.testing.assert_array_equal(saved[name],arrays[name])
                    auc,ap=score(saved['y_val'],saved['val'])
                    np.testing.assert_allclose([auc,ap,(auc+ap)/2],[result['auc'],result['aupr'],result['criterion']],rtol=0,atol=1e-12)
                    trace=pd.read_csv(directory/'trace.csv')
                    np.testing.assert_array_equal(trace.epoch,np.arange(1,epochs+1))
                    if not np.isfinite(trace.select_dtypes(include=[np.number]).values).all():
                        raise ValueError('Nonfinite training trace')
                    if stage=='inner':
                        if int(trace.loc[trace.criterion.idxmax(),'epoch'])!=result['selected_epoch']:
                            raise ValueError('Inner checkpoint selection differs')
                        selected=trace[trace.epoch==result['selected_epoch']].iloc[0]
                        np.testing.assert_allclose([selected.inner_auc,selected.inner_aupr],[auc,ap],atol=1e-12,rtol=0)
                    else:
                        if result['selected_epoch']!=epochs or set(trace.columns)!={'epoch','loss','supervised','auxiliary','queries','exchanges','ramp'}:
                            raise ValueError('Outer refit must use a fixed epoch count without validation selection')
                    group='factor' if arm.startswith('factor_') else 'pca'
                    ikey=(task,stage,seed,group)
                    state=(result['initial_state_sha256'],result['params'])
                    if ikey in initial and initial[ikey]!=state:
                        raise ValueError('Unequal initialization inside an architecture family')
                    initial[ikey]=state
                    ekey=(task,stage,seed)
                    if ekey in encoders and encoders[ekey]!=result['initial_encoder_sha256']:
                        raise ValueError('Backbone initialization differs between arms')
                    encoders[ekey]=result['initial_encoder_sha256']
                    if arm.startswith('factor_'):
                        np.testing.assert_allclose(saved['val'],saved['marginal']+saved['interaction'],atol=4e-5,rtol=4e-5)
                        if stage=='outer':
                            for part in ('marginal','interaction','val'):
                                components.append(dict(task=task,arm=arm,seed=seed,part=part,
                                    **ranking_metrics(saved['y_val'],saved['pairs_val'],saved[part])))
                    if args.replay:
                        model,data,pairs,weights=build(arrays,graph,task,arm,seed,protocol,args.device)
                        if state_digest(model)!=result['initial_state_sha256']:
                            raise ValueError('Replay initialization differs')
                        model.load_state_dict(torch.load(directory/'checkpoint.pt',map_location='cpu'))
                        actual=predict(model,data,pairs['val'],weights)
                        # Repeated FP32 CUDA scatter evaluations showed up to
                        # 1.14e-4 component drift (recorded separately), with
                        # at most 2.84e-6 AUPR drift. Check both raw scores and
                        # their probability/ranking consequences explicitly.
                        for name,prediction in actual.items():
                            np.testing.assert_allclose(prediction,saved[name],atol=2e-4,rtol=4e-5)
                        replay_auc,replay_ap=score(arrays['y_val'],actual['val'])
                        np.testing.assert_allclose([replay_auc,replay_ap],[auc,ap],atol=2e-5,rtol=0)
                        probability_error=float((torch.sigmoid(torch.from_numpy(actual['val']))-torch.sigmoid(torch.from_numpy(saved['val']))).abs().max())
                        if probability_error>5e-5:
                            raise ValueError('Replay probability drift exceeds its bound')
                        replay_errors.append(dict(task=task,arm=arm,stage=stage,identifier=identifier,
                            max_abs=float(np.max(np.abs(actual['val']-saved['val']))),
                            max_probability_abs=probability_error,delta_auc=replay_auc-auc,delta_aupr=replay_ap-ap))
                    if stage=='inner':
                        inner_rows.append(result)
                    else:
                        outer_rows.append(dict(result,**{k:v for k,v in ranking_metrics(saved['y_val'],saved['pairs_val'],saved['val']).items() if k not in result}))
                print('VERIFIED',task,stage,arm,len(jobs),flush=True)
    for task in TASKS:
        for arm in ARMS:
            rows=[row for row in inner_rows if row['task']==task and row['arm']==arm]
            best=pick_config(rows)
            chosen=selection['choices'][task][arm]
            if (chosen['config_id'],chosen['epochs'])!=(best['config_id'],best['selected_epoch']):
                raise ValueError('Frozen hyperparameter winner differs from verified inner results')
    expected_inner=len(TASKS)*len(ARMS)*len(protocol['configurations'])
    expected_outer=len(TASKS)*len(ARMS)*len(protocol['refit_seeds'])
    if (len(inner_rows),len(outer_rows))!=(expected_inner,expected_outer):
        raise ValueError('Incomplete experiment')
    frame=pd.DataFrame(outer_rows)
    frame.to_csv(out/'results.csv',index=False)
    pd.DataFrame(inner_rows).to_csv(out/'verified_tuning_results.csv',index=False)
    pd.DataFrame(components).to_csv(out/'score_components.csv',index=False)
    pd.DataFrame(controls).to_csv(out/'degree_controls.csv',index=False)
    metrics=['auc','aupr','criterion','macro_drug_auc','macro_drug_aupr','macro_rna_auc','macro_rna_aupr']
    summary=frame.groupby(['task','arm'])[metrics].agg(['mean','std'])
    summary.to_csv(out/'summary.csv')
    indexed=frame.set_index(['task','arm','seed'])
    contrasts=[('pca_query','pca_full'),('factor_bce','pca_query'),
               ('factor_pairwise','factor_bce'),('factor_exchange','factor_bce'),
               ('factor_exchange','factor_pairwise'),('factor_exchange','pca_full')]
    deltas=[]
    for task in TASKS:
        for candidate,control in contrasts:
            for metric in metrics:
                delta=np.array([indexed.loc[(task,candidate,seed),metric]-indexed.loc[(task,control,seed),metric]
                                for seed in protocol['refit_seeds']])
                deltas.append(dict(task=task,candidate=candidate,control=control,metric=metric,
                    mean_delta=float(delta.mean()),std_delta=float(delta.std(ddof=1)) if len(delta)>1 else 0.,
                    positive_seeds=int((delta>0).sum()),seeds=len(delta)))
    pd.DataFrame(deltas).to_csv(out/'paired_deltas.csv',index=False)
    def contrast(task,control,metric):
        return next(row for row in deltas if row['task']==task and row['candidate']=='factor_exchange'
                    and row['control']==control and row['metric']==metric)
    performance,mechanism={},{}
    for task in TASKS:
        performance[task]=all(contrast(task,'pca_full',metric)['mean_delta']>0 for metric in ('auc','aupr')) and contrast(task,'pca_full','criterion')['positive_seeds']>=4
        mechanism[task]=all(contrast(task,control,'macro_drug_aupr')['mean_delta']>0 and contrast(task,control,'macro_drug_aupr')['positive_seeds']>=4
                            for control in ('factor_bce','factor_pairwise'))
    eligible=not protocol['smoke'] and len(protocol['refit_seeds'])==5
    screens=dict(primary='factor_exchange',performance_passes=eligible and all(performance.values()),
                 mechanism_passes=eligible and all(mechanism.values()),per_task_performance=performance,
                 per_task_mechanism=mechanism,not_a_significance_test=True,new_split_confirmation=False,
                 literature_novelty_established=False)
    write_json(out/'SCREENS.json',screens)
    write_json(out/'VALIDATION.json',dict(passed=True,smoke=protocol['smoke'],tuning_runs=len(inner_rows),
        refit_runs=len(outer_rows),completed_runs=len(inner_rows)+len(outer_rows),test_evaluations=0,
        replayed_runs=len(replay_errors),max_replay_abs=max((r['max_abs'] for r in replay_errors),default=None),
        replay_tolerances=dict(raw_atol=2e-4,raw_rtol=4e-5,probability_atol=5e-5,metric_atol=2e-5),
        replay_roundoff_audit='replay_roundoff_audit.json; repeated FP32 CUDA inference before setting expanded component tolerance',
        protocol_sha256=protocol['sha256'],selection_sha256=selection['sha256'],
        checks=['input/source hashes','equal tuning counts','same initial backbone','same decoder initialization within family',
                'inner-only configuration/epoch selection','fixed refit epoch budgets','stored prediction metrics',
                'no test arrays','both directions of held-out target edges absent',
                'checkpoint replay' if args.replay else 'checkpoint restore at training']))
    pd.DataFrame(replay_errors).to_csv(out/'replay.csv',index=False)
    lines=['# 节点倾向与配对交互分解：交换学习实验','',
        '**主候选性能筛选：'+('通过' if screens['performance_passes'] else '未通过')+'；交换机制筛选：'+('通过' if screens['mechanism_passes'] else '未通过')+'。**','',
        '每个方案各用相同的学习率/权重衰减组合，在原训练标签内部的 80/20 划分上选择一个配置及一个轮次，再用全部原训练标签重训。外层验证不选择配置或轮次，测试集未使用。外层验证曾被前几轮实验使用，因此仍属于探索结果。','',
        '| 任务 | 方案 | Val AUC | Val AUPR | 按药物宏平均 AUPR | 按 RNA 宏平均 AUPR |',
        '|---|---|---:|---:|---:|---:|']
    for task in TASKS:
        for arm in ARMS:
            rows=frame[(frame.task==task)&(frame.arm==arm)]
            lines.append('| %s | %s | %.6f | %.6f | %.6f | %.6f |'%(task,arm,rows.auc.mean(),rows.aupr.mean(),rows.macro_drug_aupr.mean(),rows.macro_rna_aupr.mean()))
    lines+=['','## 主候选相对各自调参后的原模型','',
            '| 任务 | Δ AUC | Δ AUPR | 平均 AUC/AUPR 上升的种子 |','|---|---:|---:|---:|']
    for task in TASKS:
        lines.append('| %s | %+.6f | %+.6f | %d/%d |'%(task,contrast(task,'pca_full','auc')['mean_delta'],contrast(task,'pca_full','aupr')['mean_delta'],contrast(task,'pca_full','criterion')['positive_seeds'],len(protocol['refit_seeds'])))
    lines+=['','## 固定的内部调参结果','',
            '| 任务 | 方案 | 学习率倍数 | 权重衰减 | 重训轮数 |','|---|---|---:|---:|---:|']
    for task in TASKS:
        for arm in ARMS:
            choice=selection['choices'][task][arm]
            config=next(c for c in protocol['configurations'] if c['id']==choice['config_id'])
            lines.append('| %s | %s | %.1f | %g | %d |'%(task,arm,config['lr_multiplier'],config['weight_decay'],choice['epochs']))
    lines+=['','## 核验与解释范围','',
        '- 完成 %d 次内部调参训练、%d 次固定轮次重训；独立重放 %d 个检查点。'%(len(inner_rows),len(outer_rows),len(replay_errors)),
        '- pca_full 使用完整训练行；其他方案每轮抽取一半的正负训练行并移除查询正边。pca_query 与 pca_full 的差异同时包含查询抽样和边屏蔽，不能单独归因为边屏蔽。',
        '- 因子分解、普通排序、交换排序三组共享相同架构与查询抽样；初始化相同，并分别获得相同调参预算。',
        '- 交换项对加性 RNA/药物分数具有代数不变性；这不等于去除了所有研究偏倚，也不是因果效应。',
        '- 交换后的未知配对可能是真实但未发现的关联，辅助目标只施加弱排序约束。',
        '- 宏平均仅覆盖验证中同时含正负样本的节点，且仍使用既有抽样负例；不能解释为全候选空间的筛选精度。',
        '- 五个初始化种子不能替代独立划分；没有独立冷启动或癌种留出验证。',
        '- 使用已清理且含既有扩展的训练图，不直接与论文成绩相减。','',
        '独立重放存在 FP32 CUDA 浮点归约差异；除原分数容差外，同时约束最大概率变化不超过 5e-5、AUC/AUPR 变化不超过 2e-5。实际误差保存在 replay.csv，先前的重复推理排查保存在 replay_roundoff_audit.json。','',
        '[方法与复现](../../PAIR_EXCHANGE.md) · [文献核对](../../research/pair_exchange_20260909/NOVELTY.md)','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print(frame.groupby(['task','arm'])[metrics].mean().to_string(),flush=True)
    print('SCREENS',screens,flush=True)


if __name__=='__main__':
    main()
