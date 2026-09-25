"""Summarize completed, frozen Q1/Q2/Q3 validation without model selection."""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from improve import tabular_foundation as base


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT/'results'
OUT = RESULTS/'frontier_validation_20260914'


def fixed(path):
    value = base.read_json(path)
    if 'artifacts' in value:
        for name,digest in value['artifacts'].items():
            if base.digest(path.parent/name)!=digest:
                raise ValueError('Completed artifact changed: '+str(path.parent/name))
    return value


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'COMPLETION.json').exists():
        raise ValueError('Completed validation summary is immutable')
    site = RESULTS/'site_evidence_association_20260914'
    source = RESULTS/'site_evidence_source_20260914'
    drug = RESULTS/'drug_multimodal_association_20260914'
    site_inputs = fixed(RESULTS/'site_evidence_inputs_20260914/INPUTS.json')
    drug_inputs = fixed(RESULTS/'drug_multimodal_inputs_20260914_v3/INPUTS.json')
    sv,dv,cv = fixed(site/'VALIDATION.json'),fixed(drug/'VALIDATION.json'),fixed(source/'EVALUATION.json')
    amendment = base.read_json(drug/'REPORTING_AMENDMENT.json')
    if not amendment.get('completed') or amendment['validation_sha256']!=base.digest(drug/'VALIDATION.json'):
        raise ValueError('Reporting-only diagnostic amendment is incomplete')
    if base.digest(ROOT/'frontier_report_guard.py')!=amendment['reporting_source_sha256'] or base.digest(ROOT/'test_frontier_report_guard.py')!=amendment['test_source_sha256']:
        raise ValueError('Reporting guard or its tests changed')
    boundary = base.read_json(OUT/'BOUNDARY_AUDIT.json')
    if not boundary['passed'] or boundary['checked_contexts']!={'site':144,'drug':105,'source':8}:
        raise ValueError('Actual saved support-context audit is incomplete')
    for path,digest in boundary['files'].items():
        if base.digest(path)!=digest:
            raise ValueError('Boundary audit inputs changed')
    compatibility = base.read_json(OUT/'M2UMOL_COMPATIBILITY.json')
    if not compatibility['all_extracted_state_equals_checkpoint']:
        raise ValueError('M2UMol pretraining weights are incomplete')
    sm,dm,cm = (pd.read_csv(p/'metrics.csv') for p in (site,drug,source))
    smean = sm.groupby(['task','variant'])[['auc','ap','macro_drug_ap']].mean()
    dmean = dm.groupby(['task','variant'])[['auc','ap','macro_drug_ap']].mean()
    primary = cm[cm['subset']=='novel_triple_source_heldout'].set_index('variant')
    boot = base.read_json(source/'BOOTSTRAP.json')
    source_assessment = base.read_json(source/'ASSESSMENT.json')
    gates = dict(site=sv['gates'],source=cv['gates'],drug=dv['gates'],
                 q3_joint_performance_passed=dv['joint_performance_passed'],accepted_reference_replaced=False)
    base.write_json(OUT/'ASSESSMENT.json',gates)
    dictionary = base.read_json(RESULTS/'tabicl_pathway_context_inputs_20260913/dictionary.json')
    names = dictionary['drug_names']
    cids = pd.read_csv(ROOT.parents[1]/'incrd-run/2.construct_network/Drug.csv',dtype=str).set_index('Drug_Name').loc[names,'CID'].to_numpy(str)
    named = []
    for family,directory in [('site',site),('drug',drug)]:
        per = pd.read_csv(directory/'per_drug.csv')
        mean = per.groupby(['task','variant','drug_index'],as_index=False).agg(
            auc=('auc','mean'),ap=('ap','mean'),train_positive_degree=('train_positive_degree','first'),
            query_rows=('query_rows','first'),query_positives=('query_positives','first'))
        baseline = mean[mean.variant=='baseline'][['task','drug_index','auc','ap']].rename(columns={'auc':'reference_auc','ap':'reference_ap'})
        mean = mean.merge(baseline,on=['task','drug_index'],validate='many_to_one')
        mean['drug_name'] = mean.drug_index.map(dict(enumerate(names)))
        mean['cid'] = mean.drug_index.map(dict(enumerate(cids)))
        mean['delta_auc'],mean['delta_ap'] = mean.auc-mean.reference_auc,mean.ap-mean.reference_ap
        mean['experiment'] = family
        named.append(mean)
    pd.concat(named,ignore_index=True).to_csv(OUT/'per_drug_association_means.csv',index=False)
    per = pd.read_csv(source/'per_drug.csv',dtype={'cid':str})
    per['drug_name'] = per.cid.map(dict(zip(cids,names)))
    per.to_csv(OUT/'source_per_drug_named.csv',index=False)
    lines = ['**Q1/Q2/Q3 实际验证结果（2026-09-14）**','',
        '三个有限实验已完成：位点证据分层、药物特定靶位点池化、冻结 M2UMol 预训练 2D 药物表示。已接受参考模型保持原样。',
        '', '|实验|关联性能门槛|表示信息对照|来源分组验证|', '|---|---|---|---|']
    for candidate,label in [('evidence','Q1 证据分层'),('target','Q2 药物特定池化')]:
        gate = sv['gates'][candidate]['MiDrug']
        source_gate = cv['gates'][candidate]['source_information_gate']
        lines.append('|%s|MiDrug：%s|%s|%s|'%(label,'通过' if gate['performance_gate'] else '未通过',
            '通过' if gate['information_gate'] else '未通过','通过' if source_gate else '未通过'))
    q3info = all(g['information_gate'] for g in dv['gates']['m2umol'].values())
    lines.append('|Q3 M2UMol 预训练 2D|两任务共同条件：%s|%s|本轮未做；预训练来源独立性未建立|'%( 
        '通过' if dv['joint_performance_passed'] else '未通过','通过' if q3info else '未通过'))
    lines += ['', '这里的“通过”仅指预先登记的开发验证条件。Q1/Q2 只使用 miRNA 位点，不能证明 lncRNA 任务共同改进。三个种子是推理随机性，不是三个独立生物实验。',
        '', '**关联任务：同预算结果**','',
        '所有新分支均为 86 列：36 个固定图特征、26 个原分层证据、24 个附加特征。参考为既有 0.5 图上下文＋0.5 分层证据；每个新分支也固定与图分支等权。使用同一支持样本、查询样本、32 个 TabICLv2 成员和种子 666/0/1，未更新预训练参数，未搜索混合系数。',
        '', 'Q1/Q2 的 24 列为 8 个原 RNA 统计、明确的池化覆盖/基因数、14 个主成分；generic 和 flat 也按这套预算重做。Q1 在唯一覆盖支持 RNA 上拟合压缩，Q2 在唯一覆盖支持 RNA–drug 配对上拟合；查询独有实体不参与拟合。',
        '', '|MiDrug 分支|AUC|AP|药物宏 AP|ΔAP（百分点，相对接受参考）|','|---|---:|---:|---:|---:|']
    variants = base.read_json(site/'protocol.json')['variants']
    for variant in ['baseline',*variants]:
        r = smean.loc[('MiDrug',variant)]
        lines.append('|%s|%.9f|%.9f|%.9f|%+.6f|'%(variant,r.auc,r.ap,r.macro_drug_ap,100*(r.ap-smean.loc[('MiDrug','baseline'),'ap'])))
    sd = pd.read_csv(site/'diagnostics.csv').groupby(['variant','subset'])[['pairs','auc','ap']].mean()
    lines += ['', 'Q1 三个种子的 AUC/AP 均高于接受参考，因而通过 MiDrug 性能条件；但总体 AP 的统计量对照更高，覆盖组的药物频率匹配 AP 为 %.9f，低于 flat 的 %.9f。其药物宏 AP 相对 flat 的差为 %+.6f 个百分点，功能信息条件未通过。'%(
        sd.loc[('evidence','evidence_covered_drug_matched'),'ap'],sd.loc[('flat','evidence_covered_drug_matched'),'ap'],
        100*sv['gates']['evidence']['MiDrug']['macro_ap_delta']),
        '', 'Q2 在关联验证集有 149/1,943 个直接连接配对；只有 42 个配对所属药物在该子集中同时有正例和未标注对照，可纳入频率匹配。这一小子集的 target 与打乱表示 AP 完全相同；全体和有连接子集也没有支持主分支替换 flat。它限制的是当前稀疏池化方案，不能推断所有局部生物学关系无效。']
    lines += ['', 'Q1 的统计对照保留通道数量/覆盖/基因数而置零隐藏向量；位点打乱在证据类别与支持成员状态内进行，RNA 的位点数和简单统计留在原实体。Q2 分开原机制靶点与新增结合靶点，在同一 RNA、同一证据通道内交换向量与基因对应。没有把 Non-Functional 标记当成 RNA–drug 真阴性，也没有从结合边推断药物作用符号。',
        '', '**来源分组 MiDir：方向描述任务**','',
        '既有来源排除表移除了 39 个共享 PMID 的辅助位点；保留 6,936 个位点。主评价是 1,885 个训练中未出现的 RNA–drug–disease 三元组，共 296 个来源连通组，最大组包含 194 条。八个适配器使用同一 24 列数值预算、RNA/药物/癌种身份及线性交互，固定 LogisticRegression C=1。来源或结论文本没有进入预测器。',
        '', '|分支|AUC|耐药描述 AP|敏感描述 AP|药物宏 AP|','|---|---:|---:|---:|---:|']
    for variant in ['context_logistic',*variants]:
        r = primary.loc[variant]
        lines.append('|%s|%.9f|%.9f|%.9f|%.9f|'%(variant,r.auc,r.ap,r.sensitive_ap,r.macro_drug_ap))
    for candidate,label in [('evidence','证据分层'),('target','药物特定池化')]:
        delta = primary.loc[candidate,'ap']-primary.loc['flat','ap']
        ci = boot[candidate]['flat']['intervals']['ap']
        lines += ['', '%s相对 flat 的 AP 差值为 %+.6f 个百分点；500 次来源连通组配对重采样的 95%% 区间为 [%+.6f, %+.6f]。这反映当前划分/模型条件下的来源不确定性，未作多重比较调整。'%(label,100*delta,100*ci[0],100*ci[1])]
    source_coverage = source_assessment['coverage']
    lines += ['', 'Q2 在 %d/%d 个主评价三元组有直接靶基因连接；其余候选全部保留。投影实际只可在 218 个有连接的唯一训练配对上拟合。静态靶基因局部表示不能替代细胞依赖性或干预终点。'%(source_coverage['target_covered_novel'],source_coverage['novel_triple_source_heldout']),
        '', '来源任务的标签是数据库方向描述，不能把 resistant/sensitive 一律解释为 RNA 干预造成的药敏改变。它与关联任务的标签和候选分布不同，两类任务之间的 AP 差不能直接称为泛化下降。',
        '', '**Q3：药物表示与兼容性**','',
        '公开包的编码示例与链接检查点结构不一致：检查点使用 OGB 的 9 个原子类别特征和 3 个键类别特征，示例使用 103/6 维线性输入层；另外还有未训练的融合头。严格检查发现 13 个用于前向计算的状态条目缺失，因此该示例没有生成实验向量或任务成绩。',
        '', '在任何药物任务评分之前，协议改为作者预训练代码的 feats2d：原始 GraphGPS 前向到 global_add_pool 的 128 维输出。103 个所需状态条目全部按名称、形状加载；没有随机融合层。两次不同初始化加载后状态及测试输出一致，实际编码期间参数未变。这是冻结预训练 2D 分支的实验，不能写成论文最终微调模型的复现。',
        '', '214/216 个药物条目成功编码；PLATINUM 和 RUTHENIUM 为单原子无键记录，保留为未覆盖。原 SMILES 的电荷、盐和多组分均保留。ChemBERTa 和半径 2、2048 位且含手性的 Morgan 指纹使用完全相同覆盖，8 个基本分子统计对所有可解析药物保留；每个表示在支持药物上压缩到 16 维。',
        '', '|任务|药物表示|AUC|AP|药物宏 AP|ΔAP（百分点）|','|---|---|---:|---:|---:|---:|']
    drug_protocol = base.read_json(drug/'protocol.json')
    for task in drug_protocol['tasks']:
        for variant in ['baseline',*drug_protocol['variants']]:
            r = dmean.loc[(task,variant)]
            lines.append('|%s|%s|%.9f|%.9f|%.9f|%+.6f|'%(task,variant,r.auc,r.ap,r.macro_drug_ap,100*(r.ap-dmean.loc[(task,'baseline'),'ap'])))
    lines += ['', 'M2UMol 的 LncDrug 在三个种子中 AUC/AP 都增加，但 MiDrug 平均 AUC/AP 下降，因此两任务共同性能条件未通过。LncDrug 的药物频率匹配 AP 和药物宏 AP 有正向信号，但其覆盖组原始 AP 仍略低于打乱表示，预设信息条件也未完全通过。这些结果不等于证明所有多模态药物表示无效。']
    lines += ['', '检查点来自公开 M2UMol 仓库所链接的预训练文件，Git LFS SHA-256 已核对；代码固定提交 533697815e767cac95986f820582fec11bd30ff4。预训练用到 DrugBank 机制、临床描述和靶点知识，不能把结构输入推断的表示当成新测得的功能数据，也没有证明它与评价来源完全独立。',
        '', '**门槛、边界与可复核文件**','',
        '性能条件保持不变：MiDrug 的平均 AUC/AP 同时增加、每项至少 2/3 种子增加；Q3 还要求 LncDrug 的 AUC/AP 在每个种子都不下降。判断使用完整精度，不用四舍五入掩盖下降。表示信息条件另看覆盖子集、药物频率匹配 AP 和药物宏 AP；未把一个总体分数变好当成机制信息成立。',
        '', '完成 144 个位点关联配置、105 个药物关联配置和 8 个来源模型。31 个小规模配置通过重载、查询反序及分批一致性检查。最终审计逐个核对实际保存的投影拟合集合，全部等于唯一覆盖支持实体/配对；没有查询独有实体拟合。正式 TabICLv2 配置均完成模型保存、重载和预训练参数核验。',
        '', 'LncDrug 的未覆盖诊断子集只有 4 条记录，均为未标注类，原匹配辅助函数无法计算。报告阶段将此类匹配权重置零、指标记为 NA；双类别子集仍调用原函数。两项边界测试验证该处理。这个修正没有改变模型预测、主比较或门槛，原失败日志与 [REPORTING_AMENDMENT.json](../drug_multimodal_association_20260914/REPORTING_AMENDMENT.json) 已保留。',
        '', '历史外层关联数组、原始关联测试数组及保留来源分区的结果标签均未打开；已有 PMID 排除元数据可以复用。当前接受模型没有替换。此次结果仍来自已用于开发的验证集，不能替代新的独立验证。',
        '', '汇总门槛见 [ASSESSMENT.json](ASSESSMENT.json)；实际拟合边界见 [BOUNDARY_AUDIT.json](BOUNDARY_AUDIT.json)；M2UMol 兼容性见 [M2UMOL_COMPATIBILITY.json](M2UMOL_COMPATIBILITY.json)。',
        '', '逐药物汇总：[关联任务](per_drug_association_means.csv)、[来源任务](source_per_drug_named.csv)。全体/覆盖/未覆盖/低频药物和频率匹配结果保存在各实验目录的 diagnostics.csv 或 metrics.csv。',
        '', '完整实验：[位点关联](../site_evidence_association_20260914/REPORT.md)、[来源分组](../site_evidence_source_20260914/REPORT.md)、[药物关联](../drug_multimodal_association_20260914/REPORT.md)。所有支持状态、投影、预测、源代码快照和协议均保留。']
    successful = [candidate for candidate in ('evidence','target') if sv['gates'][candidate]['MiDrug']['performance_gate'] and sv['gates'][candidate]['MiDrug']['information_gate'] and cv['gates'][candidate]['source_information_gate']]
    if not successful and not (dv['joint_performance_passed'] and q3info):
        lines += ['', '本轮尚未得到同时通过相应性能与信息条件的替换方案。建议维持参考模型，暂停扩大局部注意力或 gene 图，把下一步投入放在可核实的 RNA 干预记录、准确细胞状态和独立功能读出。']
    else:
        lines += ['', '通过开发条件的分支仅登记为下一轮候选。后续应在新的独立来源数据和明确实验语义下确认，再判断是否替换当前参考模型。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    finding = ROOT/'FRONTIER_VALIDATION_FINDINGS.md'
    if finding.exists():
        raise ValueError('Findings file already exists; do not overwrite prior work')
    finding.write_text('三项新方向的实际验证已完成。\n\n完整报告见 [REPORT.md](results/frontier_validation_20260914/REPORT.md)。\n\n'+
        '\n'.join(lines[4:10])+'\n\n接受参考未替换；未打开历史外层、原始关联测试或保留来源结果标签。\n')
    artifacts = {name:base.digest(OUT/name) for name in ('REPORT.md','ASSESSMENT.json','BOUNDARY_AUDIT.json',
        'M2UMOL_COMPATIBILITY.json','per_drug_association_means.csv','source_per_drug_named.csv')}
    inputs = [site/'VALIDATION.json',drug/'VALIDATION.json',drug/'REPORTING_AMENDMENT.json',source/'EVALUATION.json',source/'BOOTSTRAP.json',
        RESULTS/'site_evidence_inputs_20260914/INPUTS.json',RESULTS/'drug_multimodal_inputs_20260914_v3/INPUTS.json']
    base.write_json(OUT/'COMPLETION.json',dict(completed_at_utc=datetime.now(timezone.utc).isoformat(),
        passed=True,artifacts=artifacts,inputs={str(p):base.digest(p) for p in inputs},
        summary_source_sha256=base.digest(__file__),findings_sha256=base.digest(finding),
        accepted_reference_replaced=False,historical_outer_evaluated=False,original_test_evaluated=False,
        reserved_source_holdout_evaluated=False,completed_tasks=['Q1','Q2','Q3_pretrained_2d']))
    print('FRONTIER_VALIDATION_COMPLETE',OUT/'REPORT.md',flush=True)


if __name__=='__main__':
    main()
