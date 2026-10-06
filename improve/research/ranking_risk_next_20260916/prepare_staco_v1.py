"""Seal a STACO-inspired masked ranking study before viewing performance."""
from datetime import datetime, timezone
from pathlib import Path
import shutil

from improve import anchor_ranking_experiment_v1 as experiment
from improve import staco_ranking_v1 as core
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent


def main():
    family = 'staco_ranking'
    smoke_path = HERE / 'STACO_RANKING_SMOKE_V1.json'
    smoke = base.check_seal(base.read_json(smoke_path))
    if (smoke['performance_metrics_computed'] or smoke['formal_fits']
            or smoke['new_short_neural_fits'] != 3 or smoke['identity_replays'] != 3
            or len(smoke['fold_audits']) != 40):
        raise ValueError('Invalid training-only smoke audit')
    if smoke['core_parameters'] != core.parameters():
        raise ValueError('Core parameters changed')
    for path, digest in smoke['source_sha256'].items():
        if base.digest(path) != digest:
            raise ValueError('Smoke source changed')
    root = experiment.ROOT
    core_out = root / 'results/staco_ranking_core_v1_20260916'
    out = root / 'results/anchor_priority_staco_ranking_v1_20260916'
    queue_path = HERE / 'STACO_RANKING_QUEUE_PROTOCOL_V1.json'
    if core_out.exists() or out.exists() or queue_path.exists():
        raise ValueError('Preserve existing studies')
    names = ('staco_ranking_v1.py', 'test_staco_ranking_v1.py',
        'sharpness_graph_v1.py', 'test_anchor_ranking_experiment_v1.py',
        'overlap_block_models.py', 'overlap_block_experiment.py',
        'non_tabular_experiment.py', 'tabular_foundation.py', 'graph_prediction_precision.py',
        'nested_anchor_inputs_v2.py', 'residual_graph_experiment.py',
        'anchor_ranking_experiment_v1.py', 'assess_declared_graph.py')
    inputs = [Path(__file__), smoke_path, HERE / 'staco_smoke_v1.py',
        HERE / 'STACO_ADAPTATION_V1.md', HERE / 'DISCOVERY_ARXIV_V2.xml',
        HERE / 'double_block_pauc_v1.pdf', HERE / 'double_block_pauc_v1.txt',
        HERE / 'gap_pauc_v1.pdf', HERE / 'gap_pauc_v1.txt',
        HERE / 'DOWNLOAD_MANIFEST_V1.json', HERE / 'VERIFIED_METADATA_V1.json',
        HERE / 'STACO_OPENREVIEW_V1.json', HERE / 'STACO_CODE_DISCOVERY_V1.json',
        HERE / 'STACO_AUTHOR_REPOSITORY_SEARCH_V1.json']
    for entry in base.read_json(HERE / 'DOWNLOAD_MANIFEST_V1.json')['records']:
        if (entry['status'] != 'open_access_downloaded' or not entry['title_matched']
                or base.digest(entry['path']) != entry['sha256']
                or base.digest(Path(entry['path']).with_suffix('.txt')) != entry['text_sha256']):
            raise ValueError('Paper verification failed')
    metadata = base.read_json(HERE / 'VERIFIED_METADATA_V1.json')
    if metadata[0]['items'][0]['DOI'].lower() != '10.1109/tpami.2025.3637063':
        raise ValueError('Comparison-paper metadata changed')
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family=family,
        implementation_module='staco_ranking_v1', tasks=['MiDrug', 'LncDrug'],
        kinds=list(core.KINDS), primary_kind='full', primary_controls=['no_proximal', 'bce'],
        core_parameters=core.parameters(), lnc_policy='identity_strong_reference',
        identity_records_are_not_neural_fits=True,
        selection='Fixed full and two controls, two learning rates, eight averaged-stage checkpoints; no tail-fraction, margin, dual-step, or architecture search.',
        ranking_objective='BCE plus 0.1 times two-way CVaR saddle gradient and proximal parameter penalty. Projected dual and sparse per-positive threshold updates use independent negative blocks. Every 50-step stage averages updated weights and thresholds; only averaged stages are eligible for historical selection. BCE control keeps the same proximal/averaging machinery while disabling only ranking gradient.',
        source_sha256={name: base.digest(root / name) for name in names},
        inputs={str(path.resolve()): base.digest(path) for path in inputs},
        literature=dict(title='Stochastic Primal-Dual Double Block-Coordinate for Two-way Partial AUC Maximization',
            arxiv='2505.21944v2', pdf_stated_venue='TMLR 09/2025',
            openreview_id='M3kibBFP4q', publisher_landing_verified=False,
            author_implementation_found=False, independently_implemented_from_equations=True,
            comparison_paper_doi='10.1109/TPAMI.2025.3637063'),
        mechanism_family='ranking_risk',
        mechanism='Per-positive primal/dual blocks, independent negative estimates, and averaged proximal stages for two-way tail-risk residual learning.',
        original_sgd_solver_reproduced=False, theorem_claimed_for_adamw=False,
        prior_smooth_ap_work_observed=True, no_new_performance_observed=True,
        smoke=False, original_test_source_access=False,
        evaluation='Historical screen, three initializations, three fixed partitions, frozen outer recheck; every gate required.',
        confirmed_innovations=0, existing_default_replaced=False)
    core_out.mkdir(parents=True)
    (core_out / 'source_snapshot').mkdir()
    for name in names:
        shutil.copy2(root / name, core_out / 'source_snapshot' / name)
    base.write_json(core_out / 'protocol.json', p)
    experiment.prepare(out, family, root / 'results/nested_anchor_v2_20260915/historical', core_out)
    frozen_inputs = [core_out / 'protocol.json', out / 'protocol.json', smoke_path,
                     HERE / 'STACO_ADAPTATION_V1.md']
    queue = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), family=family,
        source_sha256=base.digest(HERE / 'run_staco_queue_v1.py'),
        input_sha256={str(path): base.digest(path) for path in frozen_inputs}, device='cuda:1',
        stages=['screen', 'confirm'], stop_on_any_failed_gate=True,
        expected_new_neural_fits={'screen': 6, 'confirm_if_eligible': 6},
        expected_identity_records={'screen': 6, 'confirm_if_eligible': 6},
        adopted_neural_fits=0, identity_records_count_as_neural_fits=False, confirmed_final_innovations=0))
    base.write_json(queue_path, queue)
    print('STACO_RANKING_FROZEN', base.digest(out / 'protocol.json'), queue['sha256'], flush=True)


if __name__ == '__main__':
    main()
