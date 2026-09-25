"""Package completed context-adapter results without making a new selection."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from improve import tabular_foundation as base
from improve import tabicl_context_program as experiment
from improve import tabicl_context_program_modules as features
from improve import tabicl_pathway as operations
from improve.tabicl_pathway_modules import assess as mean_assess


def summarize(args):
    out = Path(args.out_dir).resolve()
    p = experiment.protocol(out)
    if p['smoke']:
        raise ValueError('Cannot deliver a smoke experiment')
    selection = base.check_seal(base.read_json(out/'SELECTION.json'))
    if selection['protocol_sha256'] != p['sha256']:
        raise ValueError('Selection/protocol mismatch')
    for name, expected in selection['artifacts'].items():
        if base.digest(out/name) != expected:
            raise ValueError('Selected inner artifact changed')
    phase = 'outer' if selection['eligible'] else 'inner'
    validation = base.read_json(out/('VALIDATION_'+phase+'.json'))
    if validation['protocol_sha256'] != p['sha256']:
        raise ValueError('Final validation/protocol mismatch')
    for name, expected in validation['artifacts'].items():
        if base.digest(out/name) != expected:
            raise ValueError('Final result changed after report')
    rows = pd.read_csv(out/(phase+'_metrics.csv'))
    summary = pd.read_csv(out/(phase+'_summary.csv'))
    inner = pd.read_csv(out/'inner_metrics.csv')
    chosen = selection['recipe']
    assessments = base.read_json(out/(phase+'_ASSESSMENTS.json'))
    final = next((r for r in assessments if r['recipe']['id']==chosen['id']), None)
    accepted = bool(phase=='outer' and validation['eligible'] and final and final['eligible'])

    replay, counts = [], {}
    for checked_phase in ('inner','outer') if phase=='outer' else ('inner',):
        variants = p['variants'] if checked_phase=='inner' else [chosen['variant']]+(
            ['context_permuted'] if chosen['variant']=='context' else [])
        count = 0
        for task in p['tasks']:
            for variant in variants:
                for seed in p['seeds']:
                    for kind in operations.kinds(p,task):
                        dest = operations.directory(out,checked_phase,task,variant,seed,kind)
                        r = operations.checked_result(dest,p)
                        if r['parameter_sha256'] != p['parameter_sha256']:
                            raise ValueError('Neural parameter digest mismatch')
                        replay.append(r['replay_max_abs'])
                        count += 1
        counts[checked_phase] = count
    if counts['inner'] != 84:
        raise ValueError('Incomplete formal inner experiment')
    if phase=='inner' and (out/'outer').exists():
        raise ValueError('Ineligible run unexpectedly has outer artifacts')
    paired, activity = [], []
    if chosen['variant']!='baseline':
        for task in p['tasks']:
            for seed in p['seeds']:
                baseline = rows[(rows.task==task)&(rows.seed==seed)&(rows.recipe=='baseline')].iloc[0]
                candidate = rows[(rows.task==task)&(rows.seed==seed)&(rows.recipe==chosen['id'])].iloc[0]
                paired.append(dict(task=task,seed=seed,**{
                    'delta_'+metric:float(candidate[metric]-baseline[metric])
                    for metric in ('auc','ap','macro_drug_ap')}))
                with np.load(out/'predictions'/phase/task/('s'+str(seed))/'scores.npz',allow_pickle=False) as archive:
                    delta = archive[chosen['id']]-archive['baseline']
                    activity.append(dict(task=task,seed=seed,rows=len(delta),
                                         changed_rows=int(np.count_nonzero(delta)),
                                         max_absolute_change=float(np.max(np.abs(delta)))))
    pd.DataFrame(paired,columns=['task','seed','delta_auc','delta_ap','delta_macro_drug_ap']).to_csv(out/'FINAL_PAIRED_DELTAS.csv',index=False)
    base.write_json(out/'FINAL_SHARED_BRANCH_ACTIVITY.json',dict(phase=phase,recipe=chosen,activity=activity))

    # Report how the stricter seed-wise gate differs from the previous mean gate;
    # this does not reopen model selection or authorize an outer run.
    mean_only = [mean_assess(inner.to_dict('records'),cfg,p['seeds'])
                 for cfg in p['recipes'] if cfg['variant'] in features.CANDIDATES]
    mean_eligible = [r for r in mean_only if r['eligible']]
    strict_eligible = [r for r in base.read_json(out/'inner_ASSESSMENTS.json') if r['eligible']]
    base.write_json(out/'GATE_COMPARISON.json',dict(
        registered_rule='Every LncDrug inference seed must preserve AUC/AP',
        strict_eligible_count=len(strict_eligible),mean_rule_eligible_count=len(mean_eligible),
        mean_rule_eligible=[r['recipe']['id'] for r in mean_eligible],
        selection_changed=False,outer_runs_triggered_by_this_diagnostic=False))

    models = []
    if phase=='outer':
        for task in p['tasks']:
            for seed in p['seeds']:
                for kind in operations.kinds(p,task):
                    dest = operations.directory(out,'outer',task,chosen['variant'],seed,kind)
                    models.append(dict(task=task,seed=seed,kind=kind,file=str(dest/'fitted.pkl'),
                                       sha256=base.digest(dest/'fitted.pkl')))
    candidate = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),accepted=accepted,
        status='accepted_on_historical_validation' if accepted else 'not_accepted',
        protocol_sha256=p['sha256'],selected_inner_recipe=chosen,final_phase=phase,
        final_eligibility=bool(final and final['eligible']),baseline_parent=p['parent_dir'],
        formula='(1-alpha)*verified_graph_context_score + alpha*augmented_graph_score',
        same_new_rule_for_both_tasks=True,task_specific_new_bypass=False,
        extra_feature_count=len(features.names(chosen['variant'])) if chosen['variant']!='baseline' else 0,
        original_graph_feature_count=36,selected_uses_depmap_expression=chosen['variant']=='context',
        parameter_sha256=p['parameter_sha256'],neural_parameter_updates=0,
        summary_type='Mean of per-inference-seed metrics; not a new ensemble of seed probabilities.',
        metrics=[{k:(float(v) if isinstance(v,(np.floating,np.integer)) else v) for k,v in row.items()}
                 for row in summary[summary.recipe.isin(['baseline',chosen['id']])].to_dict('records')],
        models=models,no_original_test_evaluation=True,
        limitations=p['limitations'])
    base.write_json(out/'CANDIDATE.json',candidate)

    data = base.read_json(out/'INPUT_COVERAGE.json')
    lines = ['# 下载数据接入后的 TabICLv2 实验结果','']
    if accepted:
        lines += [f"共享 `{chosen['variant']}` 分支、alpha={chosen['alpha']} 在历史内外层均通过本轮验收。",'']
    elif selection['eligible']:
        lines += [f"内部选择了 `{chosen['id']}`，但固定设置在历史外层未通过验收，不能作为接受的新版本。",'']
    else:
        lines += ['已完成数据接入与内部三种子对照；没有候选通过本轮逐种子的 LncDrug 不下降规则，因此保留基线，未运行外层。','']
    lines += [f"最终报告阶段：{phase}。以下为种子 666/0/1 指标的平均值。",'',
              '|任务|方案|AUC|AP|按药物宏 AP|ΔAUC|ΔAP|Δ宏 AP|',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    if chosen['variant']=='baseline':
        show = summary[summary.recipe=='baseline']
    else:
        show = summary[summary.recipe.isin(['baseline',chosen['id']])]
    for _,row in show.iterrows():
        lines.append(f"|{row['task']}|{row['recipe']}|{row['auc']:.9f}|{row['ap']:.9f}|{row['macro_drug_ap']:.9f}|{row['delta_auc']:+.9f}|{row['delta_ap']:+.9f}|{row['delta_macro_drug_ap']:+.9f}|")
    if paired:
        lines += ['','|任务|种子|ΔAUC|ΔAP|Δ宏 AP|','|---|---:|---:|---:|---:|']
        for row in paired:
            lines.append(f"|{row['task']}|{row['seed']}|{row['delta_auc']:+.12f}|{row['delta_ap']:+.12f}|{row['delta_macro_drug_ap']:+.12f}|")
    lines += ['', '## 内部消融与错配对照','',
              '下表给出完整新分支 alpha=1 的内部均值增量，用于观察分支方向；它不是事后重新选择的结果。','',
              '|结构|MiDrug ΔAUC|MiDrug ΔAP|LncDrug ΔAUC|LncDrug ΔAP|',
              '|---|---:|---:|---:|---:|']
    inners = pd.read_csv(out/'inner_summary.csv').set_index(['task','recipe'])
    for variant in p['variants']:
        cfg = next(c for c in p['recipes'] if c['variant']==variant and c['alpha']==1.)
        mi,lnc = inners.loc[('MiDrug',cfg['id'])],inners.loc[('LncDrug',cfg['id'])]
        lines.append(f"|{variant}|{mi['delta_auc']:+.9f}|{mi['delta_ap']:+.9f}|{lnc['delta_auc']:+.9f}|{lnc['delta_ap']:+.9f}|")
    lines += ['', f'27 个候选设置中，逐种子规则通过 {len(strict_eligible)} 个；如果仅检查旧的种子均值规则，则通过 {len(mean_eligible)} 个。',
              '上述规则对比只作说明，没有据此重选模型或开启额外外层运行。',
              '表达错配分支与 context 的维度相同，并保留注释特征；它不参加选择。一次错配不能等同于全面的随机化检验。','',
              '## 数据与核验','',
              f"DepMap 25Q2 匹配 {data['depmap']['matched_cells']}/938 个细胞系；缺失 ACH-000534。",'',
              '|实体|任一基因证据|功能证据|通路证据|有变化的表达谱|',
              '|---|---:|---:|---:|---:|']
    for task in p['tasks']:
        item=data['coverage'][task]
        lines.append(f"|{task}|{item['union_rnas']}/{item['total']}|{item['tiers']['functional']['covered_rnas']}/{item['total']}|{item['pathway_rnas']}/{item['total']}|{item['variable_expression_rnas']}/{item['total']}|")
    lines += ['', 'lncRNA 覆盖增长同时包括本地较低支持 ENCORI 证据和新上传调控表，不能全部归功于新下载数据。药物端仍为 93/216 有基因注释。',
              f"13 项工程检查通过；28 个小样本状态通过重载和换序/分批核验。本轮正式状态数：{counts}，最大重载预测误差 {max(replay):.3g}。",'',
              '三个种子是推理随机性，不是独立数据划分；历史内外层已经反复参与开发。未访问原始测试集，不能声称独立泛化或统计显著优势。',
              '这是对冻结 TabICLv2 的共享证据/表达特征适配器，未训练新的神经主干。表达相关性和聚合通路分数不能解释为已验证的因果调控。','',
              '输入、结构、复现入口和数据过滤细节见 [方法说明](TABICL_PATHWAY_CONTEXT.md)。',
              f"[候选及状态路径](results/{out.name}/CANDIDATE.json)、[内部完整结果](results/{out.name}/REPORT_inner.md)、[验收规则对比](results/{out.name}/GATE_COMPARISON.json)。"]
    if phase=='outer':
        lines += [f"[外层完整结果](results/{out.name}/REPORT_outer.md)、[外层实际验收](results/{out.name}/VALIDATION_outer.json)。"]
    destination = Path(__file__).with_name('TABICL_PATHWAY_CONTEXT_FINDINGS.md')
    destination.write_text('\n'.join(lines)+'\n')
    base.write_json(out/'DELIVERY_AUDIT.json',dict(passed=True,accepted=accepted,phase=phase,
        protocol_sha256=p['sha256'],state_counts=counts,max_replay_error=max(replay),
        inner_selection_reopened=False,no_original_test_evaluation=True,
        report_source_sha256=base.digest(__file__),findings_sha256=base.digest(destination),
        generated_artifacts={name:base.digest(out/name) for name in ('CANDIDATE.json','FINAL_PAIRED_DELTAS.csv','GATE_COMPARISON.json','FINAL_SHARED_BRANCH_ACTIVITY.json')}))
    print('DELIVERED',destination,'accepted',accepted,'phase',phase,'recipe',chosen,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',default=experiment.DEFAULT_OUT)
    summarize(parser.parse_args())
