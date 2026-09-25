"""Audit entity-frequency shortcuts on the frozen inner development partition.

No test arrays, new labels, or resampled held-out pairs are used. Matching
reweights existing pairs within strata containing both observed classes.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from improve import tabular_foundation as base
from improve import tabicl_graph_context as graph


DEFAULT_PARENT = 'improve/results/tabicl_pathway_context_20260913'
DEFAULT_OUT = 'improve/results/sampling_bias_audit_20260913'


def matched_weights(y, groups):
    """Equalize entity distributions across classes, preserving retained prevalence."""
    y, groups = np.asarray(y), np.asarray(groups)
    if y.ndim != 1 or groups.shape != y.shape or set(np.unique(y)) != {0, 1}:
        raise ValueError('Expected binary labels and aligned entity IDs')
    _, inverse = np.unique(groups, return_inverse=True)
    positive = np.bincount(inverse, weights=y)
    negative = np.bincount(inverse, weights=1-y)
    eligible = (positive > 0) & (negative > 0)
    keep = eligible[inverse]
    weights = np.zeros(len(y), dtype=np.float64)
    if not keep.any():
        return weights, dict(retained_pairs=0, retained_entities=0)
    total_pos, total_neg = y[keep].sum(), (1-y[keep]).sum()
    weights[keep & (y == 1)] = 1.
    idx = keep & (y == 0)
    weights[idx] = positive[inverse[idx]]/negative[inverse[idx]] * total_neg/total_pos
    return weights, dict(retained_pairs=int(keep.sum()),
        retained_entities=int(eligible.sum()), positive_pairs=int(total_pos),
        negative_pairs=int(total_neg), retained_prevalence=float(total_pos/keep.sum()),
        effective_sample_size=float(weights.sum()**2/np.square(weights).sum()))


def entity_features(support_pairs, support_y, query_pairs, sizes):
    """Counts and smoothed label rates based exclusively on supplied support."""
    support_pairs, query_pairs = np.asarray(support_pairs), np.asarray(query_pairs)
    y = np.asarray(support_y)
    if support_pairs.shape != (2, len(y)) or query_pairs.shape[0] != 2:
        raise ValueError('Bad pair dimensions')
    prior = float(y.mean())
    columns = []
    counts = []
    for axis, size in enumerate(sizes):
        pos = np.bincount(support_pairs[axis], weights=y, minlength=size)
        neg = np.bincount(support_pairs[axis], weights=1-y, minlength=size)
        ids = query_pairs[axis]
        prob = (pos[ids] + 2*prior)/(pos[ids]+neg[ids]+2)
        columns.extend([np.log1p(pos[ids]), np.log1p(neg[ids]), np.log(prob/(1-prob))])
        counts.append((pos, neg))
    x = np.stack(columns, axis=1)
    return x, counts


def weighted_metrics(y, score, weights):
    keep = weights > 0
    if not keep.any() or len(np.unique(y[keep])) < 2:
        return dict(auc=None, ap=None, prevalence=None)
    return dict(auc=float(roc_auc_score(y[keep], score[keep], sample_weight=weights[keep])),
        ap=float(average_precision_score(y[keep], score[keep], sample_weight=weights[keep])),
        prevalence=float(np.average(y[keep], weights=weights[keep])))


def run(args):
    out, parent = Path(args.out_dir).resolve(), Path(args.parent_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new audit output directory')
    candidate = base.read_json(parent/'CANDIDATE.json')
    pp = base.check_seal(base.read_json(parent/'protocol.json'))
    if not candidate['accepted'] or candidate['protocol_sha256'] != pp['sha256']:
        raise ValueError('Need the accepted, frozen tiered reference')
    validation = base.read_json(parent/'VALIDATION_inner.json')
    gp = graph.protocol(pp['parent_dir'])
    dictionary = base.read_json(Path(pp['inputs_dir'])/'dictionary.json')
    selected = candidate['selected_inner_recipe']['id']
    out.mkdir(parents=True)
    rows, audit, source_files = [], {}, {}
    for task in pp['tasks']:
        a = graph.features(pp['parent_dir'], gp, task, 'inner')
        sizes = (len(dictionary['rna_names'][task]), len(dictionary['drug_names']))
        y, pairs = a['y_val'].astype(int), a['pairs_val']
        val_features, counts = entity_features(a['pairs_train'], a['y_train'], pairs, sizes)
        oof = np.empty((len(a['y_train']), 6), dtype=np.float64)
        for fold in np.unique(a['fold_ids']):
            held = a['fold_ids'] == fold
            oof[held], _ = entity_features(a['pairs_train'][:,~held], a['y_train'][~held],
                                          a['pairs_train'][:,held], sizes)
        model = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=1000, random_state=0))
        model.fit(oof, a['y_train'])
        scores = dict(rna_positive_degree=val_features[:,0], drug_positive_degree=val_features[:,3],
            degree_product=val_features[:,0]+val_features[:,3],
            rna_observed_label_rate=val_features[:,2], drug_observed_label_rate=val_features[:,5],
            joint_frequency_logistic=model.predict_proba(val_features)[:,1])
        views = {'original':np.ones(len(y))}
        matching = {}
        for axis, label in ((0,'rna'), (1,'drug')):
            weight, info = matched_weights(y, pairs[axis])
            views[label+'_matched'] = weight
            # A same-pair unweighted control isolates reweighting from exclusions.
            views[label+'_eligible_unweighted'] = (weight > 0).astype(float)
            matching[label] = info
            pos, neg = counts[axis]
            names = dictionary['rna_names'][task] if axis == 0 else dictionary['drug_names']
            table = pd.DataFrame(dict(entity_index=np.arange(sizes[axis]), name=names,
                support_positive=pos.astype(int), support_unobserved=neg.astype(int),
                validation_positive=np.bincount(pairs[axis],weights=y,minlength=sizes[axis]).astype(int),
                validation_unobserved=np.bincount(pairs[axis],weights=1-y,minlength=sizes[axis]).astype(int)))
            table.to_csv(out/(task+'_'+label+'_frequencies.csv'),index=False)
        for seed in pp['seeds']:
            relative = f'predictions/inner/{task}/s{seed}/scores.npz'
            expected = validation['artifacts'].get(relative)
            if expected is None or base.digest(parent/relative) != expected:
                raise ValueError('Reference predictions changed or are unverified')
            source_files[str(parent/relative)] = expected
            with np.load(parent/relative,allow_pickle=False) as archive:
                np.testing.assert_array_equal(archive['y_val'], y)
                np.testing.assert_array_equal(archive['pairs_val'], pairs)
                scores['tiered_reference_s'+str(seed)] = archive[selected].copy()
        for label, score in scores.items():
            for view, weights in views.items():
                rows.append(dict(task=task,predictor=label,evaluation=view,
                                 **weighted_metrics(y,score,weights)))
        macro = {key:base.metrics(y,score,pairs)['macro_drug_ap'] for key,score in scores.items()}
        audit[task] = dict(support_pairs=len(a['y_train']),validation_pairs=len(y),
            support_positive=int(a['y_train'].sum()),validation_positive=int(y.sum()),
            matching=matching,macro_drug_ap_original=macro,
            zero_support_positive_rna_pairs=int((counts[0][0][pairs[0]]==0).sum()),
            zero_support_positive_drug_pairs=int((counts[1][0][pairs[1]]==0).sum()),
            train_validation_pair_overlap=0,
            logistic_train_features='Five-fold cross-fitted entity statistics; query statistics from all support',
            logistic_coefficients=model[-1].coef_.tolist(),logistic_intercept=model[-1].intercept_.tolist())
        np.savez_compressed(out/(task+'_predictions.npz'),y_val=y,pairs_val=pairs,
                            **scores,**{'weight_'+k:v for k,v in views.items()})
    table = pd.DataFrame(rows)
    table.to_csv(out/'metrics.csv',index=False)
    manifest = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),phase='inner',
        accepted_reference=selected,parent_dir=str(parent),parent_protocol_sha256=pp['sha256'],
        parent_candidate_sha256=base.digest(parent/'CANDIDATE.json'),sources=source_files,
        code_sha256=base.digest(Path(__file__)),no_outer_arrays=True,no_test_evaluation=True,
        label_interpretation='0 = sampled unobserved association, not experimentally demonstrated absence',
        matching_policy='Retain only entities with both classes, match negative entity mass to positive mass, preserve retained overall class prevalence',
        audit=audit,limitations=[
            'Matched metrics describe a changed evaluation distribution; they are not trained-model improvements.',
            'Strata without both classes are excluded and explicitly counted.',
            'Entity frequency can reflect biological prevalence and publication effort; this audit does not identify their causal contributions.',
            'The inner partition was already used for development; inference seeds are not independent datasets.'])
    base.write_json(out/'AUDIT.json',manifest)
    lines=['**负采样与实体频率审计：仅使用历史内部开发集。**','',
        '0 标签是采样未观测关联。匹配评价仅对现有样本重加权，不生成新标签、不读取外层或原始测试集。',
        '匹配前先筛选有两个类别的实体；同时报告同一保留集合的未加权对照。匹配保持保留集合的总体正例比例。','',
        '|任务|预测器|原始 AUC / AP|同药物匹配 AUC / AP|同 RNA 匹配 AUC / AP|',
        '|---|---|---|---|---|']
    for task in pp['tasks']:
        for predictor in ['rna_positive_degree','drug_positive_degree','degree_product','joint_frequency_logistic',
                          *['tiered_reference_s'+str(s) for s in pp['seeds']]]:
            vals=[]
            for view in ['original','drug_matched','rna_matched']:
                r=table[(table.task==task)&(table.predictor==predictor)&(table.evaluation==view)].iloc[0]
                vals.append(f'{r.auc:.6f} / {r.ap:.6f}' if pd.notna(r.auc) else '无可评价样本')
            lines.append('|'+task+'|'+predictor+'|'+'|'.join(vals)+'|')
        lines.extend(['',task+' 匹配覆盖：'+str(audit[task]['matching']),''])
    lines+=['频率逻辑回归在交叉拟合的支持集统计上训练，避免训练正例自身增加度数造成标签泄漏。',
            '完整同集合对照见 metrics.csv；宏 AP、实体覆盖与冻结来源见 AUDIT.json。',
            '重加权指标不可解释为模型提分，也不能据此将任何未知配对认定为真阴性。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(table[table.evaluation.isin(['original','drug_matched','rna_matched'])].to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-dir',default=DEFAULT_PARENT)
    parser.add_argument('--out-dir',default=DEFAULT_OUT)
    run(parser.parse_args())


if __name__=='__main__':
    main()
