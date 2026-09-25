"""Verify completed stage-A artifacts and write the integrated research finding."""
from pathlib import Path
import json

import numpy as np
import pandas as pd

from improve import tabular_foundation as base
from improve import rna_functional_experiment as experiment
from improve import rna_functional_inputs as inputs


RESULTS = Path('improve/results')
OUT = RESULTS/'rna_functional_stage_a_20260914'
ASSOCIATION = RESULTS/'rna_functional_experiment_20260914'
CONTEXT = RESULTS/'context_functional_20260914'
SOURCES = RESULTS/'context_source_20260914'


def check_artifacts(directory, validation, key):
    for name,digest in validation[key].items():
        if base.digest(directory/name)!=digest:
            raise ValueError('Changed artifact: '+str(directory/name))


def main():
    if (OUT/'COMPLETION.json').exists():
        raise ValueError('Stage A already has a frozen completion report')
    p = experiment.protocol(ASSOCIATION)
    validation = base.read_json(ASSOCIATION/'VALIDATION.json')
    if validation['protocol_sha256']!=p['sha256'] or not validation['passed']:
        raise ValueError('Association experiment is incomplete')
    check_artifacts(ASSOCIATION,validation,'artifacts')
    cv = base.read_json(CONTEXT/'EVALUATION.json')
    check_artifacts(CONTEXT,cv,'artifacts')
    sv = base.read_json(SOURCES/'EVALUATION.json')
    check_artifacts(SOURCES,sv,'files')
    for name in ('windows','parnet','mirbind','generic_sites','assembled'):
        inputs.verify(inputs.DEFAULT_OUT/name)
    smoke = base.read_json(RESULTS/'rna_functional_smoke_20260914/SMOKE_VALIDATION.json')
    check_artifacts(RESULTS/'rna_functional_smoke_20260914',smoke,'artifacts')
    if not smoke['passed']:
        raise ValueError('Smoke checks did not pass')
    tests = (OUT/'tests.log').read_text()
    if 'Ran 20 tests' not in tests or '\nOK\n' not in tests:
        raise ValueError('Required tests did not pass')
    state_paths = sorted(ASSOCIATION.glob('inner/**/result.json'))
    states = [base.read_json(x) for x in state_paths]
    if len(states)!=105 or not all(r['replay_passed'] and r['input_columns']==86 for r in states):
        raise ValueError('Not all registered formal states completed')
    unique_jobs = {(r['task'],r['variant'],r['seed'],r['kind']) for r in states}
    if len(unique_jobs)!=105:
        raise ValueError('Duplicate formal job identities')
    raw = pd.read_csv(ASSOCIATION/'metrics.csv')
    means = raw.groupby(['task','variant'])[['auc','ap','macro_drug_ap']].mean()
    diagnostic = pd.read_csv(ASSOCIATION/'diagnostics.csv')
    matched = diagnostic[diagnostic.subset=='covered_drug_matched'].groupby(['task','variant'])[['auc','ap']].mean()
    context = pd.read_csv(CONTEXT/'metrics.csv')
    primary = context[context.subset=='novel_triple_source_heldout'].set_index(['task','variant'])
    bootstrap = base.read_json(CONTEXT/'BOOTSTRAP.json')
    coverage = base.read_json(inputs.DEFAULT_OUT/'assembled/VALIDATION.json')['coverage']
    source_audit = base.read_json(SOURCES/'DATA_AUDIT.json')
    independent = base.read_json(CONTEXT/'INDEPENDENT_SOURCE_AUDIT.json')
    source_filter = base.read_json(CONTEXT/'SOURCE_FILTER.json')
    projection = base.read_json(CONTEXT/'PROJECTIONS.json')
    for values in independent['joint_cross_task_pmid_overlap'].values():
        if values:
            raise ValueError('Source overlap remained')
    summary = dict(association_performance_gate=validation['performance_gate'],
        association_information_gate=validation['information_gate'],reference_replaced=False,
        original_association_test_evaluated=False,historical_outer_evaluated=False,source_holdout_evaluated=False,
        tests=20,smoke_states=smoke['states'],formal_tabicl_states=len(states),
        context_adapter_states=sum(len(x) for x in projection.values()),
        formal_max_reload_error=max(r['replay_max_abs'] for r in states),
        smoke_max_query_invariance_error=smoke['max_query_invariance_error'],coverage=coverage,
        source_filter_counts=source_filter['filter_stats'],source_overlap=independent['joint_cross_task_pmid_overlap'],
        data_audit=source_audit['tasks'])
    OUT.mkdir(parents=True,exist_ok=True)
    base.write_json(OUT/'SUMMARY.json',summary)
    lines = [
        '**阶段 A 已完成：RNA 功能表示已扩展并进行了同预算对照，带来源的上下文评价也已实际跑完。当前继续保留已接受的 TabICLv2 分层证据参考。**',
        '', '日期：2026-09-14。承接 [2026-09-13 研究建议](research/frontier_next_steps_20260913/REPORT.md)。本轮只检验既有数据能支持的功能表示和来源泛化，不把方向描述改造成未经验证的 RNA 干预标签。',
        '', '关联实验的共同性能条件通过：**'+str(validation['performance_gate'])+'**；覆盖及频率匹配后的功能信息描述性条件通过：**'+str(validation['information_gate'])+'**。来源方向任务中，miRNA 有有限的 AP 信号，但尚不能确认优于 RNA-FM；lncRNA 没有稳定收益。',
        '', '此次改变的是信息表示与验证协议。没有换 HGT、增加 gene 节点或搜索更多混合系数。',
        '', '|功能输入|上轮 pilot|本轮|', '|---|---:|---:|',
        '|miRBind2 覆盖 miRNA|27/605|546/605（90.2%）|',
        '|真实 50 nt 靶窗口|35|6,975，涉及 523 个基因|',
        '|PARNET 覆盖 lncRNA|32/1,322|1,288/1,322（97.4%）|',
        '|lncRNA 的匹配编码窗口|试运行全转录本|7,021 个窗口、82,401 个局部单元|',
        '', 'miRNA 的通用分支读取与 miRBind 完全相同的成熟序列和靶窗口。lncRNA 的两类编码器使用相同的窗口、局部位置和权重；最长转录本仍是采样局部单元，不是完整全碱基表示。34 条未覆盖 lncRNA 包含 15 条旧部分序列、15 条未解析序列和 4 条不足 200 nt 的完整注释序列，全部保留在评价中。',
        '', '**固定参考的关联实验。**',
        '', '五个新分支均为 86 列输入：36 个图特征、26 个原分层证据、8 个简单统计、16 个 RNA 表示坐标。每个支持上下文只在其有覆盖 RNA 上拟合标准化/PCA，样本、32 个推理成员及 666/0/1 三个种子相同。保持 0.5 图分支权重；没有混合强度搜索。',
        '', '|任务|表示|AUC|AP|药物宏 AP|ΔAP（百分点）|', '|---|---|---:|---:|---:|---:|']
    order = ['baseline','simple','generic','functional_mean','functional_moments','functional_shuffled']
    for task in ('MiDrug','LncDrug'):
        for variant in order:
            r = means.loc[(task,variant)]
            lines.append('|%s|%s|%.6f|%.6f|%.6f|%+.4f|'%(task,variant,r.auc,r.ap,r.macro_drug_ap,
                100*(r.ap-means.loc[(task,'baseline'),'ap'])))
    lines += ['', '预先指定的功能均值＋SD 分支，MiDrug 和 LncDrug 平均 AP 分别增加 0.0089 和 0.0191 个百分点。但 LncDrug 的 seed=0 出现 AUC −0.000007796、AP −0.000083544，未通过其每个种子都不下降的既定共同条件。没有放宽容差、另选混合系数或让某个任务绕过该分支。',
        '', '`simple` 只有简单统计，表示坐标为零；`generic` 为 RNA-FM 均值；`functional_mean` 为功能局部向量均值；`functional_moments` 为同一批功能向量的均值加标准差；`functional_shuffled` 将后者在覆盖/支持成员/长度/GC/单元数分层内错配。',
        '', '覆盖 RNA 配对内部，再按药物匹配正例/未标注配对边际，得到以下 AP。它们只可在同一行任务的这个匹配分布内比较，不能与上表全体配对 AP 直接作差。',
        '', '|任务|参考|简单统计|RNA-FM|功能均值|功能均值＋SD|功能打乱|', '|---|---:|---:|---:|---:|---:|---:|']
    for task in ('MiDrug','LncDrug'):
        lines.append('|'+task+'|'+'|'.join('%.6f'%matched.loc[(task,v),'ap'] for v in order)+'|')
    lines += ['', '功能均值＋SD 在两个任务的覆盖 RNA AP 和覆盖 RNA 的药物匹配 AP 上均高于通用表示与打乱，因而通过预先登记的描述性信息条件。但差值很小，与简单统计相比也并非每项都占优，不能升级为显著性或来源泛化结论。',
        '', '完整逐种子差值、覆盖组和低频药物结果见 [关联实验](results/rna_functional_experiment_20260914/REPORT.md)、[验收](results/rna_functional_experiment_20260914/ASSESSMENT.json)及 [诊断 CSV](results/rna_functional_experiment_20260914/diagnostics.csv)。这些仍是反复开发过的历史内部数据；三个推理种子不代表三次独立实验。',
        '', '**来源分组的方向任务。**',
        '', '原始数据库记录已逐条回连并保留 Detection_Method、Support、PMID、年份和 Condition。先按 PMID 联合分组两个 RNA 任务，再在每个分区分别汇总三元组；原始记录与 PMID 在训练、验证和留出之间的交集均为 0。没有全局合并标签后再拆分，避免查询标签影响训练共识。',
        '', '|方向数据|MiDir|LncDir|', '|---|---:|---:|',
        '|原始数据库记录|10,506|2,451|',
        '|全局无冲突映射三元组，仅用于数据审计|8,667|2,128|',
        '|训练侧内无冲突三元组|5,693|1,709|',
        '|来源验证侧内无冲突三元组|2,205|284|',
        '|主验证：排除训练中已见三元组后|1,885|256|',
        '|主验证涉及的来源连通组|296|86|',
        '', '来源组按合并原始记录数量目标 60%/20%/20% 分配，不是每任务分别严格等比例。主验证还排除训练侧冲突记录中已经出现过的三元组身份。新来源的留出分区仍未评分。',
        '', '此前 PMID 24145123 的已核查表达比较在完整 MiDir 数据中对应 229 条记录（前一报告的 33 条是优先候选子集口径），Support 原文一致，均未升级为干预标签。另有 6,458 条 miRNA、1,705 条 lncRNA 高通量表达筛查记录标记为需要原始记录核查；自动关键词命中的干预候选也仍是未验证状态。',
        '', '辅助位点额外排除了 **39 个** 与任一验证/留出来源共享 PMID 的位点，余 **6,936 个**，miRNA 覆盖仍为 546。这个零交集保证针对本地辅助位点证据，不证明公共预训练模型的全部训练来源独立。',
        '', '五组表示进入相同的线性上下文交互模型：RNA/药物/癌种身份、24 个数值特征及其与药物/癌种的交互。固定 C=1，无参数搜索。下表 AP 的正类为数据库 resistant 描述；不能解释为 RNA 干预导致耐药的概率。',
        '', '|任务|模型|AUC|resistant AP|药物宏 AP|', '|---|---|---:|---:|---:|']
    for task in ('MiDir','LncDir'):
        for variant in ['context_logistic',*order[1:]]:
            r = primary.loc[(task,variant)]
            lines.append('|%s|%s|%.6f|%.6f|%.6f|'%(task,variant,r.auc,r.ap,r.macro_drug_ap))
    lines += ['', 'MiDir 功能均值＋SD 的 AP 相对打乱为 +0.056989，500 次来源连通组配对重采样区间为 [0.011834, 0.109843]。但相对 RNA-FM 仅 +0.006536，区间为 [-0.035458, 0.052837]，其 AUC 也低于 RNA-FM；不能确认功能编码器整体更好。功能均值 AP 为 0.500197，低于 RNA-FM 的 0.502483，也不能把均值＋SD 的变化完全归因于功能监督。',
        '', 'LncDir 功能均值＋SD 的 AP 为 0.728823，低于简单统计的 0.748527 和功能打乱的 0.730204；各相关差值区间包含零。仅 6 个药物满足药物宏 AP 的双类纳入条件，因此该宏指标不足以支持广泛药物泛化。',
        '', '这些区间条件于当前模型及单次来源划分，并非重新训练的多次独立数据实验。补充的未见实体诊断中，功能均值＋SD 对未见 RNA 的 AUC 为 MiDir 0.505（74 条）和 LncDir 0.470（106 条），也没有提供强冷 RNA 证据；这是事后分层诊断，不参与选模。',
        '', '详见 [来源功能报告](results/context_functional_20260914/REPORT.md)、[配对区间](results/context_functional_20260914/BOOTSTRAP.json)、[来源审计](results/context_source_20260914/DATA_AUDIT.json)和 [未见实体诊断](results/context_functional_20260914/SOURCE_SCENARIOS.csv)。',
        '', '**接下来优先做什么。**',
        '', '保留当前参考模型。miRNA 的来源验证 AP 对简单和打乱对照出现有限信号，值得沿记录语义与药物相关局部功能继续；现有证据不支持把整套 RNA-FM 直接换成功能编码器，也不支持统一加维度作为已验证改进。',
        '', '下一轮先把已保留的 Support、检测方法、原始记录和 PMID 整理成经原文核实的实验语义小集，区分耐药株表达差异与真实 RNA 干预药敏。随后只在可可靠对齐的小子集检验具体药物靶点/基因身份及状态依赖性，保留 RNA-FM、同输入均值和基因/状态打乱对照。LncRNA 分支需要单独诊断功能目标、转录本及上下文，不凭 miRNA 的局部信号自动推进。',
        '', '**工程核验。**',
        '', '20 项针对来源边界、版本号、缺失、压缩和对照不变量的测试通过；6 个小样本 TabICL 状态通过重载、换序、分批检查；105 个正式 TabICL 拟合状态全部重载核对，最大预测误差为 0；10 个上下文适配器重载/换序误差为 0。预训练表示另有抽样重算与权重不变检查。',
        '', '旧参考产物保持不变。本轮未评价历史外层、原始关联测试集或新来源留出分区，没有替换已接受模型。公开 RefSeq 请求中版本号解析的工程修正已归档，初次未完成窗口目录与完整 PARNET 来源快照保留。',
        '', '方法、脚本与完整复现命令见 [RNA_FUNCTIONAL_SOURCE.md](RNA_FUNCTIONAL_SOURCE.md)；[机器可读摘要](results/rna_functional_stage_a_20260914/SUMMARY.json)和 [测试日志](results/rna_functional_stage_a_20260914/tests.log)可复核。']
    report = '\n'.join(lines)+'\n'
    target = Path('improve/RNA_FUNCTIONAL_SOURCE_FINDINGS.md')
    target.write_text(report)
    # The result-directory entry points to the tracked report, avoiding broken relative links.
    (OUT/'REPORT.md').write_text('阶段 A 的完整研究结果见 [RNA_FUNCTIONAL_SOURCE_FINDINGS.md](../../RNA_FUNCTIONAL_SOURCE_FINDINGS.md)。\n\n'+
        '本轮完成了功能表示扩展、五组同预算关联对照、来源分组与同源位点排除后的方向对照。旧参考保留，所有正式拟合状态及来源检查已通过。\n')
    completion = dict(passed=True,summary=summary,files={str(f):base.digest(f) for f in
        [target,Path('improve/RNA_FUNCTIONAL_SOURCE.md'),OUT/'SUMMARY.json',OUT/'tests.log',OUT/'REPORT.md',
         ASSOCIATION/'VALIDATION.json',CONTEXT/'EVALUATION.json',SOURCES/'EVALUATION.json',
         CONTEXT/'INDEPENDENT_SOURCE_AUDIT.json',CONTEXT/'SOURCE_SCENARIOS.csv',
         CONTEXT/'REPORT.md',SOURCES/'REPORT.md',CONTEXT/'REPORT_WORDING_REVISION.json',
         SOURCES/'REPORT_WORDING_REVISION.json']},
        implementation_sha256=base.digest(Path(__file__)))
    base.write_json(OUT/'COMPLETION.json',completion)
    print(json.dumps({key:value for key,value in summary.items() if key!='data_audit'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
