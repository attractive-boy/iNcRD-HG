"""Audit the complete hyperparameter rescue, including independent replay."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
import torch

from improve import dual_pair_tuning as experiment
from improve import pair_exchange as parent
from improve.analyze_pair_state import macro_metrics, numerical_metric_envelope
from improve.expression_scale_mechanism import state_digest
from improve.first_round_features import sha256, write_json
from improve.network_feature_ablation import score


def read_trace(path):
    return pd.read_csv(path/'trace.csv',float_precision='round_trip')


def verify_decisions(out,p,short,selection):
    if selection['shortlist_sha256']!=short['sha256']:
        raise AssertionError('Wrong shortlist used in selection')
    for task in p['tasks']:
        for arm in p['search_arms']:
            records=[]
            for cfg in p['configurations'][arm]:
                path=experiment.directory(out,task,arm,'inner',cfg['id'],p['screening_seed'])
                records.append(experiment.checked_result(path,p))
            ranked=sorted(records,key=lambda r:(-r['selection_score'],r['config_id']))
            expected=[r['config'] for r in ranked[:p['finalists_per_arm']]]
            if expected!=short['chosen'][task][arm]:
                raise AssertionError('Shortlist does not follow prespecified ranking')
            candidates=[]
            for cfg in expected:
                traces=[read_trace(experiment.directory(out,task,arm,'inner',cfg['id'],seed))
                        for seed in [p['screening_seed']]+p['confirmation_seeds']]
                candidates.append(dict(config=cfg,**experiment.choose_epoch(traces,p['smoothing_window'])))
            chosen=sorted(candidates,key=lambda r:(-r['selection_score'],r['config']['id']))[0]
            if chosen!=selection['chosen'][task][arm] or candidates!=selection['candidates'][task][arm]:
                raise AssertionError('Selection does not match multi-seed inner curves')
        for arm in ('matched_local','matched_transport'):
            if selection['chosen'][task][arm]!=selection['chosen'][task]['dual_pair']:
                raise AssertionError('Structural control has different hyperparameters')
        if selection['chosen'][task]['dual_previous']!=p['previous_settings'][task]:
            raise AssertionError('Previous-setting reference changed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',required=True)
    parser.add_argument('--device',default='cuda:0')
    args=parser.parse_args()
    torch.set_num_threads(4)
    out=Path(args.out_dir)
    p=experiment.load_protocol(out)
    short=experiment.load_decision(out,'SHORTLIST.json',p)
    selection=experiment.load_decision(out,'SELECTION.json',p)
    verify_decisions(out,p,short,selection)
    records,replays,interventions=[],[],[]
    bases,matched={},{}
    for phase in ('screen','confirm','outer'):
        stage='outer' if phase=='outer' else 'inner'
        cache={}
        for task,arm,cfg,seed,epochs in experiment.jobs(out,p,phase):
            if task not in cache:
                cache[task]=parent.load_stage(out,p,task,stage)
            arrays,graph=cache[task]
            path=experiment.directory(out,task,arm,stage,cfg['id'],seed)
            result=experiment.checked_result(path,p)
            for key,value in dict(task=task,arm=arm,config=cfg,seed=seed,stage=stage,
                                   trained_epochs=epochs).items():
                if result[key]!=value:
                    raise AssertionError('Wrong run metadata: '+key)
            with np.load(path/'scores.npz') as saved:
                for key in ('y_val','pairs_val'):
                    np.testing.assert_array_equal(saved[key],arrays[key])
                scores={key:saved[key].copy() for key in ('val','base','correction')}
            np.testing.assert_allclose(scores['val'],scores['base']+scores['correction'],atol=1e-6,rtol=1e-6)
            observed=score(arrays['y_val'],scores['val'])
            np.testing.assert_allclose(observed,[result['auc'],result['aupr']],atol=1e-12,rtol=0)
            trace=read_trace(path)
            np.testing.assert_array_equal(trace.epoch,np.arange(1,epochs+1))
            if stage=='inner':
                decision=experiment.choose_epoch([trace],p['smoothing_window'])
                if decision['epochs']!=result['selected_epoch']:
                    raise AssertionError('Wrong saved inner checkpoint epoch')
                np.testing.assert_allclose(decision['selection_score'],result['selection_score'],atol=1e-12,rtol=0)
                low,high=numerical_metric_envelope(arrays['y_val'],scores['val'],result['restore_max_abs'])
                value=trace.iloc[result['selected_epoch']-1][['inner_auc','inner_aupr']].to_numpy(dtype=float)
                if np.any(value<low-1e-12) or np.any(value>high+1e-12):
                    raise AssertionError('Selected trace metrics disagree beyond numerical roundoff')
            elif result['selected_epoch']!=epochs or set(trace.columns)!={'epoch','loss','queries'}:
                raise AssertionError('Outer labels influenced checkpoint selection')
            model,data,pairs,weights=experiment.build(arrays,graph,task,arm,seed,cfg,p,args.device)
            if state_digest(model)!=result['initial_state_sha256'] or state_digest(model.base)!=result['initial_backbone_sha256']:
                raise AssertionError('Initialization cannot be reproduced')
            key=(stage,task,seed)
            old=bases.setdefault(key,result['initial_backbone_sha256'])
            if old!=result['initial_backbone_sha256']:
                raise AssertionError('Different backbone initialization')
            if stage=='outer' and arm in ('dual_pair','matched_local','matched_transport'):
                value=(result['initial_state_sha256'],result['params'],cfg,result['selected_epoch'])
                if matched.setdefault(key,value)!=value:
                    raise AssertionError('Matched controls are not matched')
            optimizer=experiment.make_optimizer(model,p['training'][task]['lr'],cfg)
            groups=[dict(name=g['name'],lr=g['lr'],weight_decay=g['weight_decay'],
                         params=sum(t.numel() for t in g['params'])) for g in optimizer.param_groups]
            if groups!=result['optimizer_groups']:
                raise AssertionError('Incorrect optimizer parameter groups')
            del optimizer
            model.load_state_dict(torch.load(path/'checkpoint.pt',map_location='cpu'))
            output=experiment.predict(model,data,pairs['val'],weights)
            for name in scores:
                np.testing.assert_allclose(output[name],scores[name],atol=2e-4,rtol=2e-5)
            error=float(np.max(np.abs(output['val']-scores['val'])))
            probability_error=float(np.max(np.abs(expit(output['val'])-expit(scores['val']))))
            if probability_error>5e-5:
                raise AssertionError('Excessive probability drift')
            metrics=score(arrays['y_val'],output['val'])
            low,high=numerical_metric_envelope(arrays['y_val'],scores['val'],error)
            if np.any(np.asarray(metrics)<low-1e-12) or np.any(np.asarray(metrics)>high+1e-12):
                raise AssertionError('Replay ranking metrics exceed score-roundoff bounds')
            replays.append(dict(phase=phase,task=task,arm=arm,config_id=cfg['id'],seed=seed,
                                score_max_abs=error,probability_max_abs=probability_error,
                                auc_abs=abs(metrics[0]-observed[0]),aupr_abs=abs(metrics[1]-observed[1])))
            result.update(phase=phase,**macro_metrics(arrays['y_val'],scores['val'],arrays['pairs_val']))
            records.append(result)
            if stage=='outer' and arm==p['primary']:
                for condition in p['interventions']:
                    pred=scores['base'] if condition=='branch_off' else experiment.predict(
                        model,data,pairs['val'],weights,intervention=condition)['val']
                    auc,ap=score(arrays['y_val'],pred)
                    interventions.append(dict(task=task,seed=seed,condition=condition,auc=auc,aupr=ap,
                        **macro_metrics(arrays['y_val'],pred,arrays['pairs_val'])))
            del model,data,pairs,weights
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print('VERIFIED',phase,flush=True)
    frame=pd.DataFrame(records)
    expected=sum(len(experiment.jobs(out,p,phase)) for phase in ('screen','confirm','outer'))
    if len(frame)!=expected:
        raise AssertionError('Missing or duplicated runs')
    frame.to_csv(out/'results.csv',index=False)
    replay=pd.DataFrame(replays)
    replay.to_csv(out/'replay.csv',index=False)
    pd.DataFrame(interventions).to_csv(out/'interventions.csv',index=False)
    outer=frame[frame.stage=='outer']
    metrics=('auc','aupr','criterion','macro_drug_auc','macro_drug_aupr')
    summary=outer.groupby(['task','arm'])[list(metrics)].agg(['mean','std'])
    summary.to_csv(out/'summary.csv')
    index=outer.set_index(['task','arm','seed'])
    contrasts=[]
    for task in p['tasks']:
        for control in ('dual_previous','pca_query','matched_local','matched_transport'):
            for metric in metrics:
                delta=np.asarray([index.loc[(task,'dual_pair',seed),metric]-index.loc[(task,control,seed),metric]
                                  for seed in p['refit_seeds']])
                contrasts.append(dict(task=task,control=control,metric=metric,mean_delta=float(delta.mean()),
                    std_delta=float(delta.std(ddof=1)) if len(delta)>1 else 0.,
                    positive_seeds=int((delta>0).sum()),seeds=len(delta)))
    pd.DataFrame(contrasts).to_csv(out/'contrasts.csv',index=False)
    def get(task,control,metric):
        return next(r for r in contrasts if r['task']==task and r['control']==control and r['metric']==metric)
    def passed(controls):
        return not p['smoke'] and all(get(t,c,'auc')['mean_delta']>0 and get(t,c,'aupr')['mean_delta']>0
            and get(t,c,'criterion')['positive_seeds']>=4 for t in p['tasks'] for c in controls)
    performance,rescue,mechanism=passed(('pca_query',)),passed(('dual_previous',)),passed(('matched_local','matched_transport'))
    write_json(out/'VALIDATION.json',dict(passed=True,smoke=p['smoke'],completed_runs=len(frame),
        screening_runs=int((frame.phase=='screen').sum()),confirmation_runs=int((frame.phase=='confirm').sum()),
        refit_runs=len(outer),independent_checkpoint_replays=len(replay),test_evaluations=0,
        performance_screen_passed=performance,rescue_screen_passed=rescue,mechanism_screen_passed=mechanism,
        protocol_sha256=p['sha256'],analysis_source_sha256=sha256(Path(__file__)),
        ranking_envelope_source_sha256=sha256(Path(__file__).with_name('analyze_pair_state.py')),
        max_replay_errors={k:float(replay[k].max()) for k in ('score_max_abs','probability_max_abs','auc_abs','aupr_abs')},
        checks=['frozen source/input hashes','no test arrays','stage support isolation',
                'shortlist reconstructed from all screening results','three-seed smoothed selection reconstructed',
                'outer epochs fixed before evaluation','same initialization and fixed-setting structural controls',
                'optimizer group assignments and learning rates','saved metrics and independent checkpoint replay',
                'fixed numerical tolerances and score-roundoff ranking bounds']))
    lines=['# 双流结构调参实验','',
        '固定原任务、图和架构，比较分组学习率、权重衰减、配对输出缩放与传播步长初始化。参数只从内部验证选择。',
        '两个搜索组各 12 个配置；前三名各增加两个初始化种子，以三个种子的五轮平滑平均指标共同选配置与轮数。',
        '匹配结构对照沿用主模型配置，没有独立调参；已有内外层验证均被重复使用，结果属于探索。','',
        '| 任务 | 结构 | AUC 均值 | AUPR 均值 | 药物内宏平均 AUPR |',
        '|---|---|---:|---:|---:|']
    for task in p['tasks']:
        for arm in p['refit_arms']:
            subset=outer[(outer.task==task)&(outer.arm==arm)]
            lines.append('| %s | %s | %.6f | %.6f | %.6f |' %
                         (task,arm,subset.auc.mean(),subset.aupr.mean(),subset.macro_drug_aupr.mean()))
    lines+=['','超过重新调参原结构：%s；超过旧双流设置：%s；匹配结构筛选：%s。' %
            (performance,rescue,mechanism),'',
            '| 任务 | dual_pair 减去对照 | Δ AUC | Δ AUPR | 平均指标增加的种子 |',
            '|---|---|---:|---:|---:|']
    for task in p['tasks']:
        for control in ('dual_previous','pca_query','matched_local','matched_transport'):
            a,b,c=get(task,control,'auc'),get(task,control,'aupr'),get(task,control,'criterion')
            lines.append('| %s | %s | %+.6f | %+.6f | %d/%d |' %
                         (task,control,a['mean_delta'],b['mean_delta'],c['positive_seeds'],c['seeds']))
    lines+=['','| 任务 | 模型 | 骨干 LR | 分支 LR | L2 | 配对输出缩放 | 步长初始化 | 轮数 |',
            '|---|---|---:|---:|---:|---:|---:|---:|']
    for task in p['tasks']:
        for arm in ('pca_query','dual_previous','dual_pair'):
            chosen=selection['chosen'][task][arm]
            cfg=chosen['config']
            lr=p['training'][task]['lr']*cfg['lr_multiplier']
            branch='—' if arm=='pca_query' else '%g' % (lr*cfg['branch_lr_multiplier'])
            lines.append('| %s | %s | %g | %s | %g | %g | %g | %d |' %
                         (task,arm,lr,branch,cfg['weight_decay'],cfg['correction_scale'],cfg['step_init'],chosen['epochs']))
    lines+=['','AUPR 为 average precision。药物内宏平均仅包括验证中同时含正、负例的药物。',
            '分支关闭及传播关闭诊断见 interventions.csv。固定检查点干预衡量模型依赖，不能替代重新训练的结构对照。',
            '本轮完成 %d 次训练和 %d 次检查点重放，测试集评价为 0。' % (len(frame),len(replay)),'']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print(summary.to_string(),flush=True)
    print('VALIDATED',len(frame),'performance',performance,'rescue',rescue,'mechanism',mechanism,flush=True)


if __name__=='__main__':
    main()
