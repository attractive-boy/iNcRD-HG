"""Audit completed graph studies against their declared primary and nulls.

This reader does not select a new primary or change a sealed experiment. It
recomputes paired metrics and fails closed when a registered fit is missing.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from improve import overlap_block_experiment as common
from improve.tabular_foundation import seal

METRICS = ('auc', 'ap')


def evaluate(p, records, seeds):
    primary, controls = p['primary_kind'], p['primary_controls']
    if primary not in p['kinds'] or not controls or any(k not in p['kinds'] or k == primary for k in controls):
        raise ValueError('Invalid declared primary or controls')
    required = {(task, kind, seed) for task in p['tasks']
                for kind in p['kinds'] for seed in seeds}
    if set(records) != required:
        raise ValueError('Missing or extra registered selected fits')
    rows, means, comparisons = [], {}, {}
    has_offset = all('offset_metrics' in r for r in records.values())
    if any('offset_metrics' in r for r in records.values()) != has_offset:
        raise ValueError('Incomplete offset null')
    if p.get('method', '').startswith('residual_') and not has_offset:
        raise ValueError('Residual studies require an offset null')
    for task in p['tasks']:
        means[task], comparisons[task] = {}, {}
        for kind in p['kinds']:
            values = []
            for seed in seeds:
                r = records[task, kind, seed]
                reference = records[task, primary, seed]['metrics']['anchor']
                for m in METRICS:
                    if r['metrics']['anchor'][m] != reference[m]:
                        raise ValueError('Unpaired reference metric')
                delta = {m: float(r['metrics']['anchor_composed'][m] - reference[m]) for m in METRICS}
                rows.append(dict(task=task, kind=kind, seed=seed, **delta))
                values.append(delta)
            means[task][kind] = dict(
                mean={m: float(np.mean([v[m] for v in values])) for m in METRICS},
                minimum={m: float(min(v[m] for v in values)) for m in METRICS},
                positive_count={m: sum(v[m] > 0 for v in values) for m in METRICS},
                metrics={m: float(np.mean([records[task, kind, s]['metrics']['anchor_composed'][m]
                                          for s in seeds])) for m in METRICS})
        for control in [k for k in p['kinds'] if k != primary]:
            comparisons[task][control] = {
                m: means[task][primary]['mean'][m] - means[task][control]['mean'][m]
                for m in METRICS}
        if has_offset:
            for seed in seeds:
                expected = records[task, primary, seed]['offset_metrics']['offset_composed']
                for kind in p['kinds']:
                    for m in METRICS:
                        if records[task, kind, seed]['offset_metrics']['offset_composed'][m] != expected[m]:
                            raise ValueError('Unpaired offset null')
            comparisons[task]['offset_composed'] = {
                m: float(np.mean([records[task, primary, s]['metrics']['anchor_composed'][m] -
                                  records[task, primary, s]['offset_metrics']['offset_composed'][m]
                                  for s in seeds])) for m in METRICS}
    mi, ln = means['MiDrug'][primary], means['LncDrug'][primary]
    wins = 1 if len(seeds) == 1 else 2
    tolerance = p.get('lnc_absolute_tolerance', .001)
    performance = all(mi['mean'][m] > 0 and mi['positive_count'][m] >= wins and
                      ln['minimum'][m] >= -tolerance for m in METRICS)
    mechanism = all(comparisons['MiDrug'][k][m] > 0 for k in controls for m in METRICS)
    nuisance = (all(comparisons['MiDrug']['offset_composed'][m] > 0 for m in METRICS)
                if has_offset else None)
    return dict(primary_kind=primary, primary_controls=controls, paired_seeds=list(seeds),
                performance_gate=performance, primary_mechanism_gate=mechanism,
                offset_null_gate=nuisance,
                declared_gate=performance and mechanism and (nuisance is not False),
                lnc_absolute_tolerance=tolerance, selected_fit_count=len(records),
                means=means, primary_minus_controls=comparisons, per_seed_deltas=rows,
                confirmed_final_innovations=0)


def assess(out, stage='confirm'):
    out = Path(out).resolve()
    p = common.protocol(out)
    selection = json.loads((out / 'SELECTION.json').read_text())
    ph = common.base.digest(out / 'protocol.json')
    if selection['protocol_sha256'] != ph:
        raise ValueError('Selection protocol mismatch')
    # Verify all completed fits, including the unselected screening arms.
    table = common.all_rows(out)
    seeds = [p['screen_seed']] + (p['confirm_seeds'] if stage == 'confirm' else [])
    records, files = {}, {}
    for task in p['tasks']:
        for kind in p['kinds']:
            lr = selection['chosen'][kind]['lr']
            for seed in seeds:
                dest = out / 'runs' / task / kind / ('lr' + str(lr)) / ('s' + str(seed))
                if not (dest / 'result.json').exists():
                    raise ValueError('Incomplete registered selected fit: ' + str(dest))
                r = common.verify(dest, ph)
                if 'offset_metrics' in r:
                    z = common.load(dest / 'scores.npz')
                    for name, expected in r['offset_metrics'].items():
                        actual = common.base.metrics(z['y'], z[name], z['pairs'])
                        for m in METRICS + ('macro_drug_ap',):
                            if abs(actual[m] - expected[m]) >= 1e-12:
                                raise ValueError('Offset metric mismatch')
                records[task, kind, seed] = r
                files[str(dest / 'result.json')] = common.base.digest(dest / 'result.json')
    result = evaluate(p, records, seeds)
    result.update(created_at_utc=datetime.now(timezone.utc).isoformat(), stage=stage,
                  protocol_sha256=ph, selection_sha256=common.base.digest(out / 'SELECTION.json'),
                  assessment_source_sha256=common.base.digest(__file__),
                  result_sha256=files, all_completed_metrics_recomputed=True,
                  completed_fit_records=int(len(table[['task', 'kind', 'lr', 'seed']].drop_duplicates())),
                  declaration=p['gate'], original_test_access=False,
                  scope='Historical development only. Passing does not imply fixed-partition or reserved success.')
    name = 'HISTORICAL_ASSESSMENT.json' if stage == 'confirm' else 'SCREEN_ASSESSMENT.json'
    common.base.write_json(out / name, seal(result))
    print(json.dumps({k: result[k] for k in ('primary_kind', 'performance_gate',
        'primary_mechanism_gate', 'offset_null_gate', 'declared_gate', 'means',
        'primary_minus_controls')}, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--stage', choices=('screen', 'confirm'), default='confirm')
    args = parser.parse_args()
    assess(args.out, args.stage)
