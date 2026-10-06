"""Seal two prospectively declared cores and their unchanged residual studies."""
import argparse
from datetime import datetime, timezone
import importlib
from pathlib import Path
import shutil

from improve import anchor_priority_graph_v1 as experiment
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
SPECS = {
    'polynormer': dict(module='polynormer_graph_v1',
        title='Polynormer: Polynomial-Expressive Graph Transformer in Linear Time',
        arxiv='2403.01232v3', openreview='hmv1LpNfXa', venue='ICLR 2024 Poster',
        paper='polynormer', mechanism='Multiplicative feature interactions in local and global graph updates'),
    'spline_attention': dict(module='spline_attention_graph_v1',
        title='KAA: Kolmogorov-Arnold Attention for Enhancing Attentive Graph Neural Networks',
        arxiv='2501.13456v4', openreview='atXCzVSXTJ', venue='ICLR 2025 Poster',
        paper='kaa', mechanism='Learned cubic spline scalar attention mappings with fixed feature alignment'),
}


def main(family):
    spec = SPECS[family]
    core = importlib.import_module('improve.' + spec['module'])
    smoke_path = HERE / (family.upper() + '_SMOKE_V1.json')
    smoke = base.check_seal(base.read_json(smoke_path))
    assert smoke['formal_fits'] == 0 and smoke['performance_metrics_computed'] is False
    assert smoke['core_parameters'] == core.parameters() and len(smoke['fold_audits']) == 40
    for path, digest in smoke['source_sha256'].items():
        if base.digest(path) != digest:
            raise ValueError('Smoke source changed')
    core_out = experiment.ROOT / ('results/' + family + '_core_v1_20260916')
    out = experiment.ROOT / ('results/anchor_priority_' + family + '_v1_20260916')
    queue_path = HERE / (family.upper() + '_QUEUE_PROTOCOL_V1.json')
    if core_out.exists() or out.exists() or queue_path.exists():
        raise ValueError('Use a new version rather than overwrite prepared studies')
    sources = [spec['module'] + '.py', 'test_' + spec['module'] + '.py',
        'sharpness_graph_v1.py', 'test_anchor_priority_graph_v1.py',
        'overlap_block_models.py', 'overlap_block_experiment.py',
        'non_tabular_experiment.py', 'tabular_foundation.py',
        'graph_prediction_precision.py', 'nested_anchor_inputs_v2.py',
        'residual_graph_experiment.py', 'anchor_priority_graph_v1.py',
        'assess_declared_graph.py']
    inputs = [smoke_path, HERE / (family.upper() + '_ADAPTATION_V1.md'),
        HERE / 'AUTHOR_SOURCE_V1.json', HERE / 'PDF_VERIFICATION_V1.json',
        HERE / 'feature_interaction_smoke_v1.py', Path(__file__),
        HERE / (spec['paper'] + '_author_tree_v1.json'),
        HERE / (spec['paper'] + '_openreview_v1.json')]
    paper_version = 'v3' if family == 'polynormer' else 'v4'
    inputs += [HERE / (spec['paper'] + '_arxiv_' + paper_version + suffix) for suffix in ('.pdf', '.txt')]
    author = base.read_json(HERE / 'AUTHOR_SOURCE_V1.json')
    for entry in author['records']:
        if entry['key'] == spec['paper']:
            if entry['status'] != 'downloaded' or base.digest(entry['file']) != entry['sha256']:
                raise ValueError('Missing or changed author source')
            inputs.append(Path(entry['file']))
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family=family,
        implementation_module=spec['module'], tasks=['MiDrug', 'LncDrug'],
        kinds=list(core.KINDS), primary_kind='full', primary_controls=list(core.KINDS[1:]),
        core_parameters=core.parameters(), lnc_policy='identity_strong_reference',
        identity_records_are_not_neural_fits=True,
        selection='Primary and required control are fixed before observing any new performance. Two learning rates and 400 steps match the frozen common executor; no architecture sweep.',
        source_sha256={name: base.digest(experiment.ROOT / name) for name in sources},
        inputs={str(path.resolve()): base.digest(path) for path in inputs},
        literature={k: spec[k] for k in ('title', 'arxiv', 'openreview', 'venue')},
        mechanism=spec['mechanism'],
        evaluation='Historical screen seed 666, confirm seeds 0/1, three fixed development partitions, then frozen outer recheck. All stages mandatory; failed gates stop this version.',
        no_new_performance_observed=True, smoke=False, original_test_source_access=False,
        prior_failures_preserved=True, existing_default_replaced=False, confirmed_innovations=0)
    core_out.mkdir(parents=True)
    (core_out / 'source_snapshot').mkdir()
    for name in sources:
        shutil.copy2(experiment.ROOT / name, core_out / 'source_snapshot' / name)
    base.write_json(core_out / 'protocol.json', p)
    cache = experiment.ROOT / 'results/nested_anchor_v2_20260915/historical'
    experiment.prepare(out, family, cache, core_out)
    frozen_inputs = [core_out / 'protocol.json', out / 'protocol.json', smoke_path,
                     HERE / (family.upper() + '_ADAPTATION_V1.md')]
    queue = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family=family,
        source_sha256=base.digest(HERE / 'run_feature_interaction_queue_v1.py'),
        input_sha256={str(path): base.digest(path) for path in frozen_inputs},
        device='cuda:1', stages=['screen', 'confirm'], stop_on_any_failed_gate=True,
        expected_new_neural_fits={'screen': 4, 'confirm_if_eligible': 4},
        expected_identity_records={'screen': 4, 'confirm_if_eligible': 4},
        adopted_neural_fits=0, identity_records_count_as_neural_fits=False,
        confirmed_final_innovations=0))
    base.write_json(queue_path, queue)
    print('FEATURE_STUDY_FROZEN', family, base.digest(out / 'protocol.json'), queue['sha256'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--family', required=True, choices=SPECS)
    main(parser.parse_args().family)
