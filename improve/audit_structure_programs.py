"""Reconstruct actual feature caches and frozen selections after a full trial.

The small-graph tests supply independent path-count and permutation oracles;
this audit checks that the recorded experiment used those operators and folds.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from improve import backbone_replacement as parent
from improve import graph_tabm as table
from improve import structure_program_search as experiment
from improve.structure_program_modules import ProgramEncoder, grammar, select_programs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = Path(args.out_dir)
    p = experiment.protocol(out)
    parent.decision(out,'REFIT.json',p)
    verification = json.loads((out/'VALIDATION.json').read_text())
    if not verification['passed'] or verification['smoke']:
        raise ValueError('Audit the verified full trial')
    started, records = time.monotonic(), []
    programs = grammar(p['grammar_max_length'])
    for task in p['tasks']:
        for phase in ('inner','outer'):
            path = out/'features'/phase/task
            experiment.checked_manifest(path,p)
            with np.load(path/'arrays.npz') as data:
                a = {name:data[name].copy() for name in data.files}
            original, graph, _ = parent.stage_inputs(p['input_reference_dir'],task,phase)
            baseline = table.features(p['reference_dir'],table.protocol(p['reference_dir']),task,
                                      'outer' if phase=='outer' else 'screen')
            for key in ('pairs_train','pairs_val','y_train','y_val'):
                np.testing.assert_array_equal(a[key],original[key])
            np.testing.assert_array_equal(a['fold_ids'],baseline['fold_ids'])
            for split in ('train','val'):
                np.testing.assert_array_equal(a['base_x_'+split],baseline['x_'+split])
            rules = json.loads((path/'rules.json').read_text())
            if phase == 'inner':
                rebuilt = select_programs(baseline['raw_train'],a['raw_program_train'],a['y_train'],
                                          programs,p['k'],p['selector_seed'])
                for key,value in rebuilt.items():
                    if value != rules[key]:
                        raise AssertionError('Training-only rule selection does not replay: '+key)
            else:
                if parent.digest(path/'rules.json') != parent.digest(out/'features'/'inner'/task/'rules.json'):
                    raise AssertionError('Outer rules differ from inner-frozen rules')
            encoder = ProgramEncoder(graph,task)
            selected = [programs[i] for i in rules['selected']]
            frozen_audits = json.loads((path/'null_audit.json').read_text())['phases']
            for fold in list(range(p['folds'])) + ['evaluation']:
                if fold == 'evaluation':
                    query_pairs = a['pairs_val']
                    support = a['pairs_train'][:,a['y_train']==1]
                    expected, corrected = a['raw_program_val'],a['excess_program_val']
                    seed = p['null_seed'] + 9000
                    audit_index = p['folds']
                else:
                    query = a['fold_ids']==fold
                    support = a['pairs_train'][:,(~query)&(a['y_train']==1)]
                    query_pairs = a['pairs_train'][:,query]
                    expected,corrected = a['raw_program_train'][query],a['excess_program_train'][query]
                    seed = p['null_seed'] + 1000*fold
                    audit_index = fold
                raw = encoder.evaluate(query_pairs,support,programs)
                np.testing.assert_array_equal(raw,expected)
                _,excess,audit = encoder.with_null(query_pairs,support,selected,p['null_samples'],seed,p['swaps_per_edge'])
                np.testing.assert_array_equal(excess,corrected)
                if dict(fold=fold,**audit) != frozen_audits[audit_index]:
                    raise AssertionError('Null-generation audit changed')
                records.append(dict(task=task,phase=phase,fold=fold,raw_max_abs=0.,
                    excess_max_abs=0.,null_fallback_fraction=audit['fallback_fraction'],
                    all_swaps_completed=audit['all_swaps_completed']))
            print('REBUILT',task,phase,'seconds',round(time.monotonic()-started,1),flush=True)
    report = dict(passed=True,protocol_sha256=p['sha256'],
        all_61_programs_rebuilt=True,all_excess_features_rebuilt=True,
        training_only_selection_replayed=True,outer_rules_identical_to_inner=True,
        baseline_features_identical_to_reference=True,no_test_evaluation=True,
        audit_source_sha256=parent.digest(__file__),elapsed_seconds=time.monotonic()-started,checks=records)
    parent.write_json(out/'FEATURE_RECHECK.json',report)
    print('FEATURE AUDIT PASSED',len(records),'support graphs',flush=True)


if __name__=='__main__':
    main()
