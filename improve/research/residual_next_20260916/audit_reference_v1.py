"""Read-only audit of completed nested references and descriptive score shift.

No model is fit or selected. Existing source, inputs, references, and candidate
recipes are only read. Results are written once to a new versioned directory.
"""
from datetime import datetime, timezone
import inspect
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

from improve import anchor_residual_validate_v2 as validation


HERE = Path(__file__).resolve().parent
base = validation.base


def describe(labels, scores):
    probability = expit(scores)
    return dict(rows=len(labels), positive_fraction=float(labels.mean()),
        mean_probability=float(probability.mean()),
        calibration_bias=float((probability - labels).mean()),
        logloss=float(np.mean(np.logaddexp(0., scores) - labels * scores)),
        brier=float(np.mean((probability - labels) ** 2)),
        auc=float(roc_auc_score(labels, scores)),
        ap=float(average_precision_score(labels, scores)),
        score_quantiles=np.quantile(scores, [.01, .1, .5, .9, .99]).tolist())


def main():
    output = HERE / 'REFERENCE_AUDIT_V1.json'
    if output.exists():
        raise ValueError('Preserve the completed reference audit')
    out, cache = validation.shared.data_source.OUT, validation.CACHE
    dp = validation.shared.inputs.protocol(out)
    cp = validation.experiment.nested.protocol(cache)
    parent = Path(cp['parent_reference_cache'])
    audit, inputs = [], {}
    for task in dp['tasks']:
        for split in dp['split_seeds']:
            offset_dir = cache / 'offsets' / task / str(split)
            manifest = base.check_seal(base.read_json(offset_dir / 'manifest.json'))
            validation.experiment.nested.verify_files(offset_dir, manifest)
            assert manifest['protocol_sha256'] == cp['sha256']
            assert manifest['every_row_predicted_once']
            offset = validation.experiment.common.load(offset_dir / 'arrays.npz')
            seen = np.zeros(manifest['rows'], dtype=int)
            for fold in range(5):
                dest = cache / 'fits' / task / str(split) / str(fold)
                r = base.check_seal(base.read_json(dest / 'result.json'))
                assert r['protocol_sha256'] == cp['sha256']
                validation.experiment.nested.verify_files(dest, r)
                if r.get('adopted_complete_fit', False):
                    old_path = Path(r['adopted_from'])
                    assert base.digest(old_path) == r['adopted_result_sha256']
                    old = base.check_seal(base.read_json(old_path))
                    validation.experiment.nested.verify_files(old_path.parent, old)
                    assert r['adopted_original_artifacts'] == old['files']
                    assert base.digest(dest / 'scores.npz') == old['files']['scores.npz']
                z = validation.experiment.common.load(dest / 'scores.npz')
                ids = z['query_indices']
                np.testing.assert_array_equal(offset['pairs_train'][:, ids], z['pairs'])
                np.testing.assert_array_equal(offset['anchor_oof'][ids], z['anchor'])
                np.testing.assert_allclose(z['anchor'], .7 * z['tabicl'] + .3 * z['classic'],
                                           rtol=0, atol=1e-12)
                np.testing.assert_array_equal(offset['folds'][ids], np.full(len(ids), fold))
                seen[ids] += 1
                audit.append(dict(task=task, split=split, fold=fold,
                    adopted_complete_fit=r.get('adopted_complete_fit', False),
                    tabicl_replay_max_abs=r['tabicl_replay_max_abs'],
                    classic_replay_max_abs=r['classic_replay_max_abs'],
                    result_sha256=base.digest(dest / 'result.json')))
            np.testing.assert_array_equal(seen, np.ones(len(seen), dtype=int))
            inputs[str(offset_dir / 'manifest.json')] = base.digest(offset_dir / 'manifest.json')
    recipes = {}
    for family in ('geometry', 'block', 'bicentric'):
        _, candidate = validation.checked(out, family)
        recipes[family] = dict(sha256=candidate['sha256'], recipes=candidate['recipes'],
            primary_kind=candidate['primary_kind'], primary_controls=candidate['primary_controls'])
    for name in ('run', 'report'):
        assert inspect.getsource(getattr(validation, name)) == inspect.getsource(
            getattr(validation.previous, name))
    descriptions = []
    for task in dp['tasks']:
        for split in dp['split_seeds']:
            a, _ = validation.shared.inputs.arrays(out, dp, task, split)
            train = {k: v for k, v in a.items() if k in validation.experiment.TRAIN_KEYS}
            scores, _ = validation.experiment.offset_arrays(cache, task, split, train)
            refdir = out / 'references' / task / str(split)
            ref_record = base.read_json(refdir / 'result.json')
            validation.shared.verify_artifacts(refdir, ref_record)
            z = validation.experiment.common.load(refdir / 'scores.npz')
            np.testing.assert_array_equal(a['y_val'], z['y'])
            np.testing.assert_array_equal(a['pairs_val'], z['pairs'])
            descriptions.append(dict(task=task, split=split,
                training_nested_oof=describe(a['y_train'], scores),
                frozen_evaluation_reference=describe(z['y'], z['anchor'])))
            inputs[str(refdir / 'result.json')] = base.digest(refdir / 'result.json')
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=base.digest(__file__), nested_protocol_sha256=cp['sha256'],
        candidate_recipes=recipes, reference_folds=audit,
        complete_folds=len(audit), exact_parent_adoptions=sum(x['adopted_complete_fit'] for x in audit),
        offset_alignment_and_formula_exact=True, duplicate_or_missing_oof_rows=False,
        training_and_report_bodies_unchanged=True,
        score_distributions=descriptions, input_sha256=inputs,
        limitations=['Descriptive comparison across different train/evaluation rows; no causal diagnosis.',
            'Evaluation labels were already exposed development labels, used only for this audit.',
            'No recipe, training loss, selection, or gate was changed from these observations.'],
        model_fits=0, confirmed_final_innovations=0))
    base.write_json(output, result)
    print('REFERENCE_AUDIT_COMPLETE', len(audit), result['exact_parent_adoptions'], flush=True)
    for cell in descriptions:
        print(cell['task'], cell['split'],
              {name: {k: round(cell[name][k], 6) for k in ('auc', 'ap', 'calibration_bias', 'logloss')}
               for name in ('training_nested_oof', 'frozen_evaluation_reference')}, flush=True)


if __name__ == '__main__':
    main()
