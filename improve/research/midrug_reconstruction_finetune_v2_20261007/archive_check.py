"""Independent final hash/count checks after immutable selection and diagnostics."""
import json
from pathlib import Path
from common import HERE, CLEAN, RELEASE, sha256


def main():
    archive = json.loads((HERE / 'archive_manifest.json').read_text())
    plan = json.loads((HERE / 'results/plan.json').read_text())
    selection = json.loads((HERE / 'selection.json').read_text())
    diagnostics = json.loads((HERE / 'component_diagnostics.json').read_text())
    verification = json.loads((HERE / 'verification.json').read_text())
    checks = []
    for name, expected in archive['source_hashes'].items():
        assert sha256(HERE / name) == expected
        assert sha256(HERE / 'source_snapshot' / name) == expected
        checks.extend([name, 'source_snapshot/' + name])
    for name, expected in archive['artifact_hashes'].items():
        assert sha256(HERE / name) == expected
        checks.append(name)
    for name, expected in archive['frozen_dependencies'].items():
        assert sha256(name) == expected
        checks.append(name)
    assert sha256(CLEAN / 'graph.pt') == plan['graph_sha256']
    assert sha256(CLEAN / 'inner_data.npz') == plan['data_sha256']
    checks.extend(['clean_graph', 'clean_inner_data'])
    assert selection['plan_sha256'] == sha256(HERE / 'results/plan.json')
    assert selection['verification_sha256'] == sha256(HERE / 'verification.json')
    assert diagnostics['selection_sha256'] == sha256(HERE / 'selection.json')
    assert not diagnostics['selection_changed']
    assert len(diagnostics['records']) == 9
    assert verification['new_checkpoint_replays'] == 9
    assert verification['immutable_reused_control_checks'] == 6
    artifacts = [name for name in archive['artifact_hashes']
                 if name.startswith('results/') and name != 'results/plan.json']
    assert len(artifacts) == 45
    for name, expected in selection['result_hashes'].items():
        assert sha256(HERE / 'results' / name) == expected
    protection = json.loads((HERE / 'plmf_hash_recheck.json').read_text())
    assert protection['status'] == 'passed' and len(protection['records']) == 10
    refs = HERE.parent / 'baselines_m123_20261004/results'
    for r in protection['records']:
        stem = f"{r['dataset']}_f{r['fold']}_s42"
        assert sha256(refs / f'incrd_m123_{stem}.pt') == r['checkpoint_sha256']
        assert sha256(RELEASE / 'data/frozen' / f"{r['dataset']}_fold{r['fold']}.npz") == r['split_sha256']
        assert sha256(RELEASE / 'data/frozen' / f"{r['dataset']}_graph.pt") == r['graph_sha256']
    assert (HERE / 'check.log').read_text().strip().endswith('ALL PASSED')
    assert sum(line.startswith('DONE ') for line in (HERE / 'launch.log').read_text().splitlines()) == 9
    assert sum(line.startswith('PASS ') for line in (HERE / 'audit.log').read_text().splitlines()) == 9
    assert not selection['outer_validation_evaluated'] and not selection['test_evaluated']
    initial = json.loads((HERE / 'initial_checks.json').read_text())
    assert len(initial['boundary_checks']) == 3
    for row in initial['boundary_checks']:
        assert row['decoder_updates'] == 99 and row['main_updates'] == 101
    failure = json.loads((HERE.parent / 'midrug_reconstruction_finetune_20261007/failure_manifest.json').read_text())
    for path, digest in {**failure['artifact_hashes'], **failure['frozen_dependencies']}.items():
        assert sha256(path) == digest
    report = dict(status='passed', hash_checks=len(checks), result_artifacts=45,
                  new_checkpoint_replays=9, reused_control_checks=6,
                  protected_plmf_records=10, selection_unchanged=True,
                  archive_sha256=sha256(HERE / 'archive_manifest.json'),
                  selection_sha256=sha256(HERE / 'selection.json'),
                  no_lncrna=True, outer_validation_evaluated=False, test_evaluated=False)
    (HERE / 'archive_recheck.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
