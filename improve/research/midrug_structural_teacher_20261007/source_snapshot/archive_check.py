"""Independent final hash/count checks after immutable selection and diagnostics."""
import json
import numpy as np
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
    literature=HERE.parent/'midrug_representation_literature_20261007'
    for name,expected in json.loads((literature/'manifest.json').read_text())['files'].items():
        assert sha256(literature/name)==expected
        checks.append('literature/'+name)
    geometry=HERE.parent/'midrug_representation_geometry_20261007'
    for name,expected in json.loads((geometry/'precheck.json').read_text())['source_hashes'].items():
        source=Path(name)
        assert sha256(source if source.is_absolute() else geometry/source)==expected
        checks.append('geometry/'+name)
    assert sha256(CLEAN / 'graph.pt') == plan['graph_sha256']
    assert sha256(CLEAN / 'inner_data.npz') == plan['data_sha256']
    checks.extend(['clean_graph', 'clean_inner_data'])
    assert selection['plan_sha256'] == sha256(HERE / 'results/plan.json')
    mask_manifest=json.loads((HERE/'mask_manifest.json').read_text())
    assert sha256(HERE/'masks.npz')==mask_manifest['masks_sha256']==plan['masks_sha256']
    assert mask_manifest['source_sha256']==sha256(HERE/'prepare.py')
    assert mask_manifest['graph_sha256']==plan['graph_sha256']
    with np.load(HERE/'masks.npz') as z:
        for fold in range(3):
            for node,n in [('miRNA',605),('drug',216)]:
                b=z[f'block_{fold}_{node}'];r=z[f'random_{fold}_{node}']
                assert b.shape==r.shape==(800,n) and b.dtype==r.dtype==np.bool_
                assert (b.sum(1)==round(.5*n)).all() and (r.sum(1)==round(.5*n)).all()
                np.testing.assert_array_equal(b.reshape(400,2,n).sum(1),r.reshape(400,2,n).sum(1))
    assert selection['verification_sha256'] == sha256(HERE / 'verification.json')
    assert diagnostics['selection_sha256'] == sha256(HERE / 'selection.json')
    assert not diagnostics['selection_changed']
    assert len(diagnostics['records']) == 9
    assert verification['new_checkpoint_replays'] == 9
    assert verification['immutable_reused_control_checks'] == 6
    artifacts = [name for name in archive['artifact_hashes']
                 if name.startswith('results/') and name != 'results/plan.json']
    assert len(artifacts) == 36
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
    report = dict(status='passed', hash_checks=len(checks), result_artifacts=36,
                  new_checkpoint_replays=9, reused_control_checks=6,
                  protected_plmf_records=10, selection_unchanged=True,
                  archive_sha256=sha256(HERE / 'archive_manifest.json'),
                  selection_sha256=sha256(HERE / 'selection.json'),
                  no_lncrna=True, outer_validation_evaluated=False, test_evaluated=False)
    (HERE / 'archive_recheck.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
