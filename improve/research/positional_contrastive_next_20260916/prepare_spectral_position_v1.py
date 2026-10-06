"""Freeze full-spectrum positional kernels after training-only checks."""
from datetime import datetime, timezone
from pathlib import Path
import shutil

from improve import anchor_priority_graph_v1 as experiment
from improve import stable_spectral_position_v1 as core
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent


def main():
    family = 'stable_spectral_position'
    smoke_path = HERE / 'STABLE_SPECTRAL_POSITION_SMOKE_V1.json'
    smoke = base.check_seal(base.read_json(smoke_path))
    if (smoke['performance_metrics_computed'] or smoke['formal_fits']
            or smoke['new_short_neural_fits'] != 2 or smoke['identity_replays'] != 2
            or len(smoke['fold_audits']) != 40):
        raise ValueError('Invalid training-only smoke audit')
    if smoke['core_parameters'] != core.parameters():
        raise ValueError('Core parameters changed')
    for path, digest in smoke['source_sha256'].items():
        if base.digest(path) != digest:
            raise ValueError('Smoke source changed')
    for record in smoke['fold_audits']:
        if not record['identity']:
            if record['spectral']['truncated'] or record['spectral']['nodes'] != record['spectral']['retained_eigenvectors']:
                raise ValueError('A query fold used a truncated eigenspace')
    root = experiment.ROOT
    core_out = root / 'results/stable_spectral_position_core_v1_20260916'
    out = root / 'results/anchor_priority_stable_spectral_position_v1_20260916'
    queue_path = HERE / 'STABLE_SPECTRAL_POSITION_QUEUE_PROTOCOL_V1.json'
    if core_out.exists() or out.exists() or queue_path.exists():
        raise ValueError('Preserve existing studies')
    names = ('stable_spectral_position_v1.py', 'test_stable_spectral_position_v1.py',
        'sharpness_graph_v1.py', 'test_anchor_spectral_position_v1.py',
        'overlap_block_models.py', 'overlap_block_experiment.py',
        'non_tabular_experiment.py', 'tabular_foundation.py', 'graph_prediction_precision.py',
        'nested_anchor_inputs_v2.py', 'residual_graph_experiment.py',
        'anchor_priority_graph_v1.py', 'assess_declared_graph.py')
    inputs = [Path(__file__), smoke_path, HERE / 'spectral_position_smoke_v1.py',
        HERE / 'SPECTRAL_POSITION_ADAPTATION_V1.md', HERE / 'SPE_ID_CHECK_V1.xml',
        HERE / 'spe_v3.pdf', HERE / 'spe_v3.txt', HERE / 'SPE_SOURCE_ACQUISITION_V1.json',
        HERE / 'SPE_REPOSITORY_RESOLUTION_V1.json', HERE / 'spe_author_tree_v2.json',
        HERE / 'SPE_AUTHOR_SOURCE_V1.json', HERE / 'SPE_AUTHOR_SOURCE_V2.json']
    first = base.read_json(HERE / 'SPE_AUTHOR_SOURCE_V1.json')
    if not first['pdf_title_verified_by_letters_and_visual_text'] or base.digest(HERE / 'spe_v3.pdf') != first['pdf_sha256']:
        raise ValueError('Paper identity changed')
    sources = {}
    for manifest in (first, base.read_json(HERE / 'SPE_AUTHOR_SOURCE_V2.json')):
        if manifest['tree_sha'] != '22b10033564119f5296bf76d2c012489ef2b4aac':
            raise ValueError('Unexpected author source revision')
        for entry in manifest['records']:
            if entry.get('status') == 200:
                if base.digest(entry['file']) != entry['sha256']:
                    raise ValueError('Author source changed')
                sources[entry['source']] = Path(entry['file'])
    if len(sources) != 8:
        raise ValueError('Incomplete author implementation reading set')
    inputs.extend(sources.values())
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family=family,
        implementation_module='stable_spectral_position_v1', tasks=['MiDrug', 'LncDrug'],
        kinds=list(core.KINDS), primary_kind='full', primary_controls=['constant_spectrum'],
        core_parameters=core.parameters(), lnc_policy='identity_strong_reference',
        identity_records_are_not_neural_fits=True,
        selection='Fixed full-spectrum learned positions and a capacity-matched shared-spectrum control; only two shared learning rates, with no eigendimension, architecture or source-graph search.',
        source_sha256={name: base.digest(root / name) for name in names},
        inputs={str(path.resolve()): base.digest(path) for path in inputs},
        literature=dict(title='On the Stability of Expressive Positional Encodings for Graphs',
            arxiv='2310.02579v3', pdf_stated_venue='ICLR 2024',
            author_tree='22b10033564119f5296bf76d2c012489ef2b4aac',
            author_readme_uses_earlier_title=True, author_code_executed=False),
        mechanism_family='spectral_graph',
        mechanism='Complete-eigenspace, eigenvalue-dependent positional kernels pooled by an equivariant GIN, distinct from applying a spectral convolution to attributes.',
        same_broad_family_as_prior_polynomial_filter=True,
        no_context_feature_columns_added=True, no_universal_expression_or_generalization_theorem_claim=True,
        no_new_performance_observed=True, smoke=False, original_test_source_access=False,
        evaluation='Historical screen, three initializations, three fixed partitions, frozen outer recheck; every gate required.',
        confirmed_innovations=0, existing_default_replaced=False)
    core_out.mkdir(parents=True)
    (core_out / 'source_snapshot').mkdir()
    for name in names:
        shutil.copy2(root / name, core_out / 'source_snapshot' / name)
    base.write_json(core_out / 'protocol.json', p)
    experiment.prepare(out, family, root / 'results/nested_anchor_v2_20260915/historical', core_out)
    frozen_inputs = [core_out / 'protocol.json', out / 'protocol.json', smoke_path,
                     HERE / 'SPECTRAL_POSITION_ADAPTATION_V1.md']
    queue = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family=family,
        source_sha256=base.digest(HERE / 'run_spectral_position_queue_v1.py'),
        input_sha256={str(path): base.digest(path) for path in frozen_inputs}, device='cuda:1',
        stages=['screen', 'confirm'], stop_on_any_failed_gate=True,
        expected_new_neural_fits={'screen': 4, 'confirm_if_eligible': 4},
        expected_identity_records={'screen': 4, 'confirm_if_eligible': 4},
        adopted_neural_fits=0, identity_records_count_as_neural_fits=False, confirmed_final_innovations=0))
    base.write_json(queue_path, queue)
    print('SPECTRAL_POSITION_FROZEN', base.digest(out / 'protocol.json'), queue['sha256'], flush=True)


if __name__ == '__main__':
    main()
