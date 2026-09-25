"""Evaluate the preregistered baseline-query-rounding diagnostic.

This keeps the augmented scores and chosen alpha fixed. It does not reselect a
model, replace primary results, or rerun alternate augmented query schedules.
"""
import argparse
from pathlib import Path

import numpy as np

from improve import tabular_foundation as base
from improve import tabicl_graph_context as graph
from improve import tabicl_graph_context_modules as graph_kernels
from improve import tabicl_pathway as experiment
from improve import tabicl_pathway_modules as features


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',default='improve/results/tabicl_pathway_calibration_20260913')
    args=parser.parse_args()
    out=Path(args.out_dir)
    p=experiment.protocol(out)
    choice=experiment.selected(out,p)
    policy=base.check_seal(base.read_json(out/'NUMERICAL_DIAGNOSTIC_PROTOCOL.json'))
    if policy['protocol_sha256']!=p['sha256'] or policy['selection_sha256']!=base.digest(out/'SELECTION.json'):
        raise ValueError('Diagnostic policy or frozen choice changed')
    for path,expected in policy['reference_files'].items():
        if base.digest(path)!=expected:
            raise ValueError('Historical numerical reference changed')
    primary_validation=base.read_json(out/'VALIDATION_outer.json')
    if not primary_validation['all_states_replayed']:
        raise ValueError('Primary outer inference must be completed first')
    recipe=choice['recipe']
    alpha=recipe['alpha']
    primary_rows,diagnostic_rows,details=[],[],[]
    arrays={}
    parent=Path(p['parent_dir'])
    for task in p['tasks']:
        a,gp=experiment.stage_data(out,p,task,'outer')
        metadata=base.read_json(parent/f'QUERY_BATCH_AUDIT_{task}.json')
        if not metadata['passed'] or metadata['protocol_sha256']!=gp['sha256']:
            raise ValueError('Wrong historical numerical audit')
        path=parent/metadata['arrays_file']
        if base.digest(path)!=metadata['arrays_sha256']:
            raise ValueError('Historical single-view arrays changed')
        arrays[task+'_y_val']=a['y_val']
        arrays[task+'_pairs_val']=a['pairs_val']
        with np.load(path,allow_pickle=False) as historical:
            np.testing.assert_array_equal(historical['y_val'],a['y_val'])
            np.testing.assert_array_equal(historical['pairs_val'],a['pairs_val'])
            for seed in p['seeds']:
                full,components,_=graph.components(parent,gp,'outer',task,seed,a)
                original=graph_kernels.combine(full,components,p['graph_recipes'][task])
                np.testing.assert_allclose(original,historical['primary_s'+str(seed)],atol=1e-14,rtol=0)
                alternative=historical['separate_s'+str(seed)].copy()
                augmented,_=experiment.augmented_score(out,p,a,'outer',task,recipe['variant'],seed)
                selected=(1-alpha)*original+alpha*augmented
                alternative_selected=(1-alpha)*alternative+alpha*augmented
                with np.load(out/'predictions/outer'/task/('s'+str(seed))/'scores.npz',allow_pickle=False) as saved:
                    np.testing.assert_array_equal(saved['baseline'],original)
                    np.testing.assert_array_equal(saved[recipe['id']],selected)
                for rows,b,s in [(primary_rows,original,selected),(diagnostic_rows,alternative,alternative_selected)]:
                    rows.append(dict(task=task,seed=seed,recipe='baseline',**base.metrics(a['y_val'],b,a['pairs_val'])))
                    rows.append(dict(task=task,seed=seed,recipe=recipe['id'],**base.metrics(a['y_val'],s,a['pairs_val'])))
                arrays[f'{task}_baseline_s{seed}']=alternative
                arrays[f'{task}_selected_s{seed}']=alternative_selected
                details.append(dict(task=task,seed=seed,
                    baseline_schedule_max_abs=float(np.max(np.abs(alternative-original))),
                    selected_schedule_max_abs=float(np.max(np.abs(alternative_selected-selected))),
                    augmented_scores_held_fixed=True))
    primary=features.assess(primary_rows,recipe,p['seeds'])
    diagnostic=features.assess(diagnostic_rows,recipe,p['seeds'])
    if primary['eligible']!=primary_validation['eligible']:
        raise ValueError('Primary acceptance was not reproduced')
    np.savez_compressed(out/'NUMERICAL_DIAGNOSTIC.npz',**arrays)
    result=dict(protocol_sha256=p['sha256'],diagnostic_protocol_sha256=policy['sha256'],
        source_sha256=base.digest(__file__),primary=primary,baseline_rounding_diagnostic=diagnostic,
        details=details,primary_predictions_unchanged=True,no_reselection=True,
        scope=policy['scope'],arrays_sha256=base.digest(out/'NUMERICAL_DIAGNOSTIC.npz'))
    base.write_json(out/'NUMERICAL_DIAGNOSTIC.json',result)
    print('NUMERICAL_DIAGNOSTIC',{'primary_eligible':primary['eligible'],
                                 'baseline_rounding_eligible':diagnostic['eligible']},flush=True)


if __name__=='__main__':
    main()
