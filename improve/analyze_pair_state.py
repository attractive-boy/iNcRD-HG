"""Verify every saved run, replay checkpoints, and report paired contrasts."""
import argparse
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

from improve import pair_exchange as parent
from improve import pair_state as experiment
from improve.expression_scale_mechanism import state_digest
from improve.first_round_features import sha256, write_json
from improve.network_feature_ablation import score


def macro_metrics(y, logits, pairs):
    aucs, aps = [], []
    for drug in np.unique(pairs[1]):
        keep = pairs[1] == drug
        if len(np.unique(y[keep])) != 2:
            continue
        aucs.append(roc_auc_score(y[keep], logits[keep]))
        aps.append(average_precision_score(y[keep], logits[keep]))
    return dict(macro_drug_auc=float(np.mean(aucs)), macro_drug_aupr=float(np.mean(aps)),
                macro_drug_count=len(aps))


def numerical_metric_envelope(y, saved, max_score_error):
    """Bounds on binary ranking metrics under an observed uniform score error.

    Class-opposed shifts bound AUC. AP also depends on splitting ties between
    positives, so bound each positive's threshold precision using counts of
    guaranteed/possible positives and negatives above it. Original logits and
    probabilities still have to pass fixed elementwise tolerances.
    """
    values = np.asarray(saved, dtype=np.float64)
    direction = 2*np.asarray(y, dtype=np.float64)-1
    epsilon = max_score_error + np.finfo(np.float64).eps*max(1.,float(np.abs(values).max()))*8
    lower = np.asarray(score(y, values-epsilon*direction))
    upper = np.asarray(score(y, values+epsilon*direction))
    positive = np.sort(values[np.asarray(y)==1])
    negative = np.sort(values[np.asarray(y)==0])
    guaranteed_positive = len(positive)-np.searchsorted(positive, positive+2*epsilon, side='left')+1
    possible_positive = len(positive)-np.searchsorted(positive, positive-2*epsilon, side='left')
    guaranteed_negative = len(negative)-np.searchsorted(negative, positive+2*epsilon, side='left')
    possible_negative = len(negative)-np.searchsorted(negative, positive-2*epsilon, side='left')
    lower[1] = np.mean(guaranteed_positive/(guaranteed_positive+possible_negative))
    upper[1] = np.mean(possible_positive/(possible_positive+guaranteed_negative))
    return lower, upper


def main():
    global experiment
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--experiment-module', default='improve.pair_state')
    args = parser.parse_args()
    if args.experiment_module not in ('improve.pair_state', 'improve.dual_pair'):
        raise ValueError('Unsupported experiment module')
    experiment = importlib.import_module(args.experiment_module)
    torch.set_num_threads(4)
    out = Path(args.out_dir)
    p = experiment.load_protocol(out)
    selection = experiment.load_selection(out, p)
    records, replays, interventions = [], [], []
    initial_backbones, initial_branches = {}, {}
    for stage in ('inner', 'outer'):
        for task in p['tasks']:
            arrays, graph = parent.load_stage(out, p, task, stage)
            arms = p['tuned_arms'] if stage == 'inner' else p['arms']
            for arm in arms:
                jobs = [(p['tuning_seed'], c, p['training'][task]['train_epochs'], c['id']) for c in p['configurations']] if stage == 'inner' else [
                    (seed, selection['chosen'][task][arm]['config'], selection['chosen'][task][arm]['epochs'], 's'+str(seed))
                    for seed in p['refit_seeds']]
                for seed, config, epochs, identifier in jobs:
                    path = experiment.directory(out, task, arm, stage, identifier)
                    result = experiment.checked_result(path, p)
                    with np.load(path / 'scores.npz') as saved:
                        for name in ('y_val', 'pairs_val'):
                            np.testing.assert_array_equal(saved[name], arrays[name])
                        scores = {k:saved[k].copy() for k in ('val', 'base', 'correction')}
                    np.testing.assert_allclose(scores['val'], scores['base']+scores['correction'], atol=1e-6, rtol=1e-6)
                    observed = score(arrays['y_val'], scores['val'])
                    np.testing.assert_allclose(observed, [result['auc'], result['aupr']], atol=1e-12, rtol=0)
                    trace = pd.read_csv(path / 'trace.csv')
                    np.testing.assert_array_equal(trace.epoch, np.arange(1, epochs+1))
                    if stage == 'inner':
                        if int(trace.loc[trace.criterion.idxmax(), 'epoch']) != result['selected_epoch']:
                            raise AssertionError('Incorrect inner checkpoint selection')
                        low, high = numerical_metric_envelope(arrays['y_val'], scores['val'], result['restore_max_abs'])
                        if not low.mean()-1e-12 <= trace.criterion.max() <= high.mean()+1e-12:
                            raise AssertionError('Restored checkpoint disagrees beyond recorded numerical error')
                    else:
                        if result['selected_epoch'] != epochs or set(trace.columns) != {'epoch','loss','queries'}:
                            raise AssertionError('Outer labels entered epoch selection')
                    model, data, pairs, weights = experiment.build(arrays, graph, task, arm, seed, p, args.device)
                    if state_digest(model) != result['initial_state_sha256']:
                        raise AssertionError('Cannot reproduce initialization')
                    base_key = (stage, task, seed)
                    old = initial_backbones.setdefault(base_key, result['initial_backbone_sha256'])
                    if old != result['initial_backbone_sha256']:
                        raise AssertionError('Backbones were initialized differently')
                    if model.branch is not None:
                        signature = (result['initial_state_sha256'], result['params'])
                        old = initial_branches.setdefault(base_key, signature)
                        if old != signature:
                            raise AssertionError('Pair controls have different initial parameters')
                    model.load_state_dict(torch.load(path / 'checkpoint.pt', map_location='cpu'))
                    output = experiment.predict(model, data, pairs['val'], weights)
                    for name in scores:
                        np.testing.assert_allclose(output[name], scores[name], atol=2e-4, rtol=2e-5)
                    probability_error = float(np.max(np.abs(expit(output['val'])-expit(scores['val']))))
                    if probability_error > 5e-5:
                        raise AssertionError('Excessive replay probability drift')
                    replay_metrics = score(arrays['y_val'], output['val'])
                    score_error = float(np.max(np.abs(output['val']-scores['val'])))
                    low, high = numerical_metric_envelope(arrays['y_val'], scores['val'], score_error)
                    if np.any(np.asarray(replay_metrics)<low-1e-12) or np.any(np.asarray(replay_metrics)>high+1e-12):
                        raise AssertionError('Replay metrics exceed the score-roundoff envelope')
                    replays.append(dict(stage=stage, task=task, arm=arm, seed=seed, config_id=config['id'],
                        score_max_abs=score_error,
                        probability_max_abs=probability_error,
                        auc_abs=abs(replay_metrics[0]-observed[0]), aupr_abs=abs(replay_metrics[1]-observed[1]),
                        auc_envelope_width=float(high[0]-low[0]), aupr_envelope_width=float(high[1]-low[1])))
                    result.update(macro_metrics(arrays['y_val'], scores['val'], arrays['pairs_val']))
                    records.append(result)
                    if stage == 'outer' and arm == p['primary']:
                        for condition in p.get('interventions', ('branch_off', 'no_transport', 'no_seed')):
                            pred = scores['base'] if condition == 'branch_off' else experiment.predict(
                                model, data, pairs['val'], weights, intervention=condition)['val']
                            auc, ap = score(arrays['y_val'], pred)
                            interventions.append(dict(task=task, seed=seed, condition=condition, auc=auc, aupr=ap,
                                **macro_metrics(arrays['y_val'], pred, arrays['pairs_val'])))
                    del model, data, pairs, weights
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            print('VERIFIED', stage, task, flush=True)
    frame = pd.DataFrame(records)
    frame.to_csv(out / 'results.csv', index=False)
    replay_frame = pd.DataFrame(replays)
    replay_frame.to_csv(out / 'replay.csv', index=False)
    pd.DataFrame(interventions).to_csv(out / 'interventions.csv', index=False)
    outer = frame[frame.stage == 'outer']
    summary = outer.groupby(['task','arm'])[['auc','aupr','criterion','macro_drug_auc','macro_drug_aupr']].agg(['mean','std'])
    summary.to_csv(out / 'summary.csv')
    indexed = outer.set_index(['task','arm','seed'])
    contrasts = []
    primary = p['primary']
    primary_controls = p.get('primary_controls', ('pca_default','pca_query','pair_local','pair_features'))
    comparisons = [(primary,c) for c in primary_controls]
    comparisons += [(a,c) for a in p['tuned_arms'] if a != primary for c in ('pca_default','pca_query') if a != c]
    for task in p['tasks']:
        for candidate, control in comparisons:
            for metric in ('auc','aupr','criterion','macro_drug_auc','macro_drug_aupr'):
                delta = np.array([indexed.loc[(task,candidate,s),metric]-indexed.loc[(task,control,s),metric]
                                  for s in p['refit_seeds']])
                contrasts.append(dict(task=task,candidate=candidate,control=control,metric=metric,
                    mean_delta=float(delta.mean()), std_delta=float(delta.std(ddof=1)) if len(delta)>1 else 0.,
                    positive_seeds=int((delta>0).sum()), seeds=len(delta)))
    comparisons_frame = pd.DataFrame(contrasts)
    comparisons_frame.to_csv(out / 'contrasts.csv', index=False)
    def get(task, control, metric):
        return next(r for r in contrasts if r['task']==task and r['candidate']==primary
                    and r['control']==control and r['metric']==metric)
    def passes(controls):
        return all(get(t,c,'auc')['mean_delta']>0 and get(t,c,'aupr')['mean_delta']>0
                   and get(t,c,'criterion')['positive_seeds']>=4 for t in p['tasks'] for c in controls)
    performance = False if p['smoke'] else passes(('pca_default','pca_query'))
    transport = False if p['smoke'] else passes(p.get('mechanism_controls', ('pair_local',)))
    boundary_control = p.get('boundary_control', 'pair_features')
    boundary = None if boundary_control is None else (False if p['smoke'] else all(
        get(t,boundary_control,'aupr')['mean_delta']>0 and get(t,boundary_control,'criterion')['mean_delta']>0 for t in p['tasks']))
    expected = len(p['tasks'])*(len(p['tuned_arms'])*len(p['configurations']) + len(p['arms'])*len(p['refit_seeds']))
    if len(frame) != expected:
        raise AssertionError('Missing runs')
    write_json(out / 'VALIDATION.json', dict(passed=True, smoke=p['smoke'], completed_runs=len(frame),
        tuning_runs=int((frame.stage=='inner').sum()), refit_runs=int((frame.stage=='outer').sum()),
        independent_checkpoint_replays=len(replays), test_evaluations=0,
        performance_screen_passed=performance, transport_screen_passed=transport, boundary_screen_passed=boundary,
        protocol_sha256=p['sha256'],
        analysis_source_sha256=sha256(Path(__file__)), experiment_module=args.experiment_module,
        near_tie_sensitive_replays=int((replay_frame.aupr_abs>1e-5).sum()),
        max_replay_errors={k:float(replay_frame[k].max()) for k in ('score_max_abs','probability_max_abs','auc_abs','aupr_abs')},
        checks=['source and input hashes','no test arrays','stage support isolation','saved prediction metrics',
                'internal checkpoint selection','fixed outer epochs','equal backbone initialization',
                'equal pair-branch allocated parameters and initialization','independent checkpoint replay',
                'fixed score/probability tolerances and score-roundoff ranking envelope']))
    lines = ['# 节点与配对联合编码：结构实验', '',
             '原关联预测任务、现有输入；内部训练划分调参，冻结后重训并评价既有外层验证集。该验证集曾反复使用，结果属于探索。', '',
             '| 任务 | 结构 | AUC 均值 | AUPR 均值 | 药物内宏平均 AUPR |', '|---|---|---:|---:|---:|']
    for task in p['tasks']:
        for arm in p['arms']:
            subset = outer[(outer.task==task)&(outer.arm==arm)]
            lines.append('| %s | %s | %.6f | %.6f | %.6f |' % (task,arm,subset.auc.mean(),subset.aupr.mean(),subset.macro_drug_aupr.mean()))
    lines += ['', '预设性能筛选通过：%s；配对传播筛选通过：%s；显式关联边界增量筛选通过：%s。' % (performance,transport,boundary), '',
              '| 任务 | '+primary+' − 对照 | Δ AUC | Δ AUPR | 平均指标提升种子 |', '|---|---|---:|---:|---:|']
    for task in p['tasks']:
        for control in primary_controls:
            a,b,c=get(task,control,'auc'),get(task,control,'aupr'),get(task,control,'criterion')
            lines.append('| %s | %s | %+.6f | %+.6f | %d/%d |' % (task,control,a['mean_delta'],b['mean_delta'],c['positive_seeds'],c['seeds']))
    lines += ['', '参数由内部验证选择，外层验证不用于选轮数。所有结构共用 BCE、固定标签、PCA 与节点骨干；pca_default 使用预先固定的原设置。',
              '配对分支的三组分配参数相同，但禁用传播或边界值会使部分参数不参与对照的输出。',
              '训练后结构干预的诊断见 interventions.csv；它们衡量模型依赖，不是生物因果证据。',
              '重放保留固定的逐分数/概率误差检查；近乎同分样本的排序变化用实际分数漂移推导的指标范围核对。原始预测和训练记录不变。',
              '本轮完成 %d 次训练及 %d 个检查点重放，测试集评价为 0。' % (len(frame),len(replays)), '']
    (out / 'REPORT.md').write_text('\n'.join(lines))
    print(summary.to_string(), flush=True)
    print('VALIDATED',len(frame),'performance',performance,'transport',transport,'boundary',boundary,flush=True)


if __name__ == '__main__':
    main()
