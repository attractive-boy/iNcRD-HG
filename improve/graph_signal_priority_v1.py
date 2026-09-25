"""Versioned graph-signal diffusion follow-up against the unchanged strong anchor.

The denoisers, reconstruction objective, cross-fitting, and training-only
calibration are inherited. The old dual-task failure remains unchanged. New
diffusion initializations, then fixed development partitions and an outer
recheck, are required before this family can be counted as a success.
"""
import argparse
from datetime import datetime, timezone
import gc
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from improve import graph_signal_calibrated as old
from improve import graph_signal_diffusion as core
from improve import non_tabular_experiment as metrics
from improve import tabular_foundation as base

ROOT = metrics.ROOT
HERE = ROOT / 'research/generative_recheck_20260916'
OUT = ROOT / 'results/graph_signal_priority_v1_20260916'
VARIANTS = ('support', 'annotation', 'shuffled')
READOUTS = ('reverse', 'one_step')
PRIMARY = 'annotation_reverse'
CONTROLS = ('support_reverse', 'shuffled_reverse', 'annotation_one_step')
KINDS = (PRIMARY,) + CONTROLS
TRAIN_KEYS = frozenset(('pairs_train', 'y_train', 'folds', 'kr', 'kd'))
OWN = ('graph_signal_priority_v1.py', 'test_graph_signal_priority_v1.py',
       'graph_signal_diffusion.py', 'graph_signal_calibrated.py',
       'graph_signal_experiment.py', 'non_tabular_experiment.py',
       'non_tabular_models.py', 'tabular_foundation.py')


def configuration(variant):
    if variant not in VARIANTS:
        raise ValueError('Unregistered diffusion variant')
    cfg = next(dict(c) for c in core.configurations() if c['id'] == variant)
    cfg['epochs'] = 200
    return cfg


def selected_load(path, keys):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key].copy() for key in keys}


def support(training, fold):
    if set(training) != TRAIN_KEYS:
        raise ValueError('Only the five declared training arrays are accepted')
    if fold not in range(5):
        raise ValueError('Unknown cross-fitting fold')
    return old.support_matrix(training['pairs_train'], training['y_train'], training['folds'],
                              fold, (len(training['kr']), len(training['kd'])))


@torch.no_grad()
def predict(model, graph):
    model.eval()
    x = torch.as_tensor(graph, device=next(model.parameters()).device)
    row = model[0].predict(x, True)
    column = model[1].predict(x.T, True).T
    one = .5 * (model[0].predict(x, False) + model[1].predict(x.T, False).T)
    return {'reverse': (.5 * row + .5 * column).cpu().numpy(),
            'one_step': one.cpu().numpy()}


def fit(training, fold, cfg, seed, device, smoke=False):
    if device != 'cuda:1' and not (smoke and device == 'cpu'):
        raise ValueError('Formal fitting is restricted to cuda:1')
    if cfg != configuration(cfg['id']) and not smoke:
        raise ValueError('Frozen denoiser recipe changed')
    graph = support(training, fold)
    torch.random.default_generator.manual_seed(seed)
    if device != 'cpu':
        torch.cuda.set_device(device)
        torch.cuda.manual_seed(seed)
    model = old.construct(graph, training['kr'], training['kd'], cfg, device)
    x = torch.as_tensor(graph, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    trace = []
    start = time.monotonic()
    for epoch in range(1, cfg['epochs'] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = .5 * ((model[0].training_prediction(x) - x).square().mean() +
                     (model[1].training_prediction(x.T) - x.T).square().mean())
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite diffusion loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if epoch % 20 == 0 or epoch == cfg['epochs']:
            trace.append(dict(epoch=epoch, training_reconstruction_loss=float(loss.detach())))
    return model, trace, dict(epoch=cfg['epochs'], training_steps=cfg['epochs'],
        seconds=time.monotonic() - start, joint_orientation_models=2,
        checkpoint_selected_on_evaluation=False, trained_neural_model=True,
        fitting_positive_edges=int(graph.sum()))


def verify_files(directory, record):
    for name, expected in record['files'].items():
        if base.digest(Path(directory) / name) != expected:
            raise ValueError('Artifact changed: ' + str(Path(directory) / name))


def protocol(out=OUT):
    p = base.check_seal(base.read_json(Path(out) / 'protocol.json'))
    for name, expected in p['source_sha256'].items():
        if base.digest(ROOT / name) != expected:
            raise ValueError('Source changed: ' + name)
    for path, expected in p['input_sha256'].items():
        if base.digest(path) != expected:
            raise ValueError('Input changed: ' + path)
    return p


def prepare(out=OUT):
    out = Path(out).resolve()
    if out.exists():
        raise ValueError('Preserve existing experiment')
    smoke_path = HERE / 'GRAPH_SIGNAL_PRIORITY_SMOKE_V1.json'
    smoke = base.check_seal(base.read_json(smoke_path))
    if (smoke['new_short_joint_fits'] != 3 or smoke['formal_fits'] or
            smoke['evaluation_labels_read'] or smoke['performance_metrics_computed'] or
            len(smoke['fold_audits']) != 40):
        raise ValueError('Invalid training-only smoke audit')
    for path, expected in smoke['source_sha256'].items():
        if base.digest(path) != expected:
            raise ValueError('Smoke source changed')
    old_p = base.read_json(old.OUT / 'protocol.json')
    if old_p['configs'] != [configuration(v) for v in VARIANTS]:
        raise ValueError('Inherited recipe mismatch')
    for name, expected in old_p['source_sha256'].items():
        if base.digest(ROOT / name) != expected:
            raise ValueError('Old frozen source changed')
    out.mkdir(parents=True)
    (out / 'source_snapshot').mkdir()
    for name in OWN:
        shutil.copy2(ROOT / name, out / 'source_snapshot' / name)
    paths = [old.OUT / 'protocol.json', smoke_path,
        HERE / 'recheck_saved_diffusion_v1.py', HERE / 'SAVED_DIFFUSION_RECHECK_V1.json',
        HERE / 'GRAPH_SIGNAL_PRIORITY_ADAPTATION_V1.md',
        HERE / 'graph_signal_priority_smoke_v1.py', HERE / 'run_graph_signal_priority_v1.py']
    literature_dir = ROOT / 'research/nontabular_frontier_20260914'
    paths += [literature_dir / name for name in ('giffcf_paper.html', 'giffcf_paper.txt',
              'giffcf_meta.json', 'giffcf_readme.md', 'giffcf_source.py', 'giffcf_denoiser.py')]
    for task in ('MiDrug', 'LncDrug'):
        inherited = old.OUT / 'inputs' / task / 'inner/arrays.npz'
        strong = ROOT / 'results/observation_offset_inputs_20260915' / task / 'arrays.npz'
        a = selected_load(inherited, TRAIN_KEYS | {'pairs_val', 'y_val'})
        b = selected_load(strong, {'pairs_train', 'y_train', 'pairs_val', 'y_val',
                                  'anchor_666', 'anchor_0', 'anchor_1'})
        for key in ('pairs_train', 'y_train', 'pairs_val', 'y_val'):
            np.testing.assert_array_equal(a[key], b[key])
        a.update({k: v for k, v in b.items() if k.startswith('anchor_')})
        if set(map(tuple, a['pairs_train'].T)) & set(map(tuple, a['pairs_val'].T)):
            raise ValueError('Train/evaluation pair overlap')
        target = out / 'inputs' / task / 'arrays.npz'
        target.parent.mkdir(parents=True)
        np.savez_compressed(target, **a)
        paths += [inherited, strong, target]
    p = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        method='graph_signal_priority_v1', tasks=['MiDrug', 'LncDrug'],
        configs=[configuration(v) for v in VARIANTS], primary=PRIMARY,
        controls=list(CONTROLS), variants=list(VARIANTS), readouts=list(READOUTS), folds=5,
        repetitions=[dict(reference_seed=666, diffusion_seed=20260914, adopt=True),
                     dict(reference_seed=0, diffusion_seed=0, adopt=False),
                     dict(reference_seed=1, diffusion_seed=1, adopt=False)],
        fold_seed_stride=1009, composition_weight=.1, lnc_policy='identity_strong_reference',
        lnc_absolute_tolerance=.001, device='cuda:1',
        source_sha256={name: base.digest(ROOT / name) for name in OWN},
        input_sha256={str(path.resolve()): base.digest(path) for path in paths},
        epoch_policy='Exactly 200 epochs, inherited from the old study; no checkpoint, learning-rate, coefficient, spectrum, or denoiser search.',
        calibration='Each variant/readout separately uses five-fold OOF training scores and training labels. Mean calibrated logits over the same five models at inference.',
        prediction='MiDrug unchanged strong anchor + 0.1 times calibrated diffusion logit. LncDrug is the exact unchanged strong anchor, with no denoiser or calibrator fit.',
        gate='Both mean MiDrug metrics increase, each in at least 2/3 diffusion initializations, and full exceeds every declared control on both mean metrics. Each LncDrug metric loss is at most 0.001. Every later fixed-partition and frozen-outer gate is mandatory.',
        old_failure_and_exploratory_diagnostic_seen=True,
        screen_is_adoption_not_independent_validation=True,
        old_shared_two_task_failure_preserved=True, previous_outer_results_seen=True,
        this_candidate_outer_results_seen=False, original_test_source_access=False,
        calibration_composition_identity_and_reuse_are_not_innovations=True,
        model_count_unit='One jointly trained two-orientation model per support fold; two readouts are not two fits. New and adopted folds are counted separately.',
        family='generative_association_reconstruction', confirmed_final_innovations=0,
        existing_default_replaced=False))
    base.write_json(out / 'protocol.json', p)
    print('GRAPH_SIGNAL_PRIORITY_FROZEN', p['sha256'], flush=True)


def run_variant(out, p, a, variant, repetition, device):
    reference_seed = repetition['reference_seed']
    cfg = configuration(variant)
    directory = out / 'runs/MiDrug' / ('s' + str(reference_seed)) / variant
    record_path = directory / 'result.json'
    if record_path.exists():
        record = base.read_json(record_path)
        if record['protocol_sha256'] != p['sha256']:
            raise ValueError('Result protocol mismatch')
        verify_files(directory, record)
        return
    training = {key: a[key] for key in TRAIN_KEYS}
    oof = {key: np.full(len(a['y_train']), np.nan) for key in READOUTS}
    values = {key: [] for key in READOUTS}
    files, fit_records = {}, []
    directory.mkdir(parents=True, exist_ok=True)
    for fold in range(5):
        dest = directory / ('fold_' + str(fold))
        dest.mkdir(exist_ok=True)
        graph = support(training, fold)
        selected = np.flatnonzero(a['folds'] == fold)
        pairs = a['pairs_train'][:, selected]
        query = a['pairs_val']
        if graph[query[0], query[1]].any():
            raise ValueError('Evaluation query entered support')
        seed = repetition['diffusion_seed'] + p['fold_seed_stride'] * fold
        if (dest / 'result.json').exists():
            fold_record = base.read_json(dest / 'result.json')
            verify_files(dest, fold_record)
            if fold_record['protocol_sha256'] != p['sha256'] or fold_record['seed'] != seed:
                raise ValueError('Fold identity mismatch')
        else:
            provenance = None
            if repetition['adopt']:
                source = old.OUT / 'runs/MiDrug/inner' / variant / ('fold_' + str(fold))
                source_record = base.read_json(source / 'result.json')
                verify_files(source, source_record)
                old_aggregate = base.read_json(source.parent / 'result.json')
                if source_record['seed'] != seed or old_aggregate['config'] != cfg:
                    raise ValueError('Adopted recipe or seed mismatch')
                verify_files(source.parent, old_aggregate)
                shutil.copy2(source / 'model.pt', dest / 'model.pt')
                shutil.copy2(source / 'trace.json', dest / 'trace.json')
                provenance = dict(source_result=str(source / 'result.json'),
                    source_result_sha256=base.digest(source / 'result.json'),
                    source_scores_sha256=base.digest(source / 'scores.npz'),
                    source_model_sha256=base.digest(source / 'model.pt'))
                detail = dict(epoch=cfg['epochs'], training_steps=0, seconds=0.,
                    joint_orientation_models=2, checkpoint_selected_on_evaluation=False,
                    trained_neural_model=False, fitting_positive_edges=int(graph.sum()))
            else:
                model, trace, detail = fit(training, fold, cfg, seed, device)
                torch.save(model.state_dict(), dest / 'model.pt')
                base.write_json(dest / 'trace.json', trace)
                before_reload = predict(model, graph)
            model = old.construct(graph, training['kr'], training['kd'], cfg, device)
            model.load_state_dict(torch.load(dest / 'model.pt', map_location=device, weights_only=True))
            predictions = predict(model, graph)
            saved = dict(oof_indices=selected, pairs_oof=pairs, pairs_val=query)
            error = 0.
            for key in READOUTS:
                saved[key + '_oof'] = predictions[key][pairs[0], pairs[1]]
                saved[key + '_val'] = predictions[key][query[0], query[1]]
                if repetition['adopt']:
                    with np.load(source / 'scores.npz', allow_pickle=False) as z:
                        for identity in ('oof_indices', 'pairs_oof', 'pairs_val'):
                            np.testing.assert_array_equal(saved[identity], z[identity])
                        for suffix in ('_oof', '_val'):
                            np.testing.assert_array_equal(saved[key + suffix], z[key + suffix])
                            error = max(error, float(np.abs(saved[key + suffix] - z[key + suffix]).max()))
                else:
                    np.testing.assert_array_equal(predictions[key], before_reload[key])
            np.savez_compressed(dest / 'scores.npz', **saved)
            fold_record = dict(protocol_sha256=p['sha256'], fold=fold, variant=variant,
                seed=seed, reference_seed=reference_seed, config=cfg, **detail,
                adopted=bool(repetition['adopt']), adoption=provenance, replay_max_abs=error,
                query_labels_used_in_denoiser_fitting=False,
                fitting_query_edges_absent=True, evaluation_query_edges_absent=True,
                files={name: base.digest(dest / name) for name in ('model.pt', 'trace.json', 'scores.npz')})
            base.write_json(dest / 'result.json', fold_record)
            del model
            gc.collect()
            torch.cuda.empty_cache()
            print('DIFFUSION_FOLD', reference_seed, variant, fold,
                  'adopted' if repetition['adopt'] else 'new', round(detail['seconds'], 2), flush=True)
        with np.load(dest / 'scores.npz', allow_pickle=False) as z:
            np.testing.assert_array_equal(z['oof_indices'], selected)
            np.testing.assert_array_equal(z['pairs_oof'], pairs)
            np.testing.assert_array_equal(z['pairs_val'], query)
            for key in READOUTS:
                oof[key][selected] = z[key + '_oof']
                values[key].append(z[key + '_val'].copy())
        fit_records.append(fold_record)
        for name in (*fold_record['files'], 'result.json'):
            files[str(Path('fold_' + str(fold)) / name)] = base.digest(dest / name)
    scores = dict(y=a['y_val'], pairs=a['pairs_val'], y_train=a['y_train'],
                  pairs_train=a['pairs_train'], folds=a['folds'],
                  anchor=a['anchor_' + str(reference_seed)])
    calibration = {}
    for key in READOUTS:
        if not np.isfinite(oof[key]).all():
            raise ValueError('Incomplete training OOF coverage')
        calibration[key] = old.fit_calibration(oof[key], a['y_train'])
        calibrated = old.apply_calibration(np.stack(values[key]), calibration[key]).mean(0)
        scores[key + '_oof'] = oof[key]
        scores[key + '_fold_val'] = np.stack(values[key])
        scores[key + '_calibrated'] = calibrated
        scores[key + '_composed'] = scores['anchor'] + p['composition_weight'] * calibrated
    np.savez_compressed(directory / 'scores.npz', **scores)
    base.write_json(directory / 'calibration.json', calibration)
    files.update({name: base.digest(directory / name) for name in ('scores.npz', 'calibration.json')})
    measured = {key: metrics.metrics(scores['y'], scores[key], scores['pairs'])
                for key in ('anchor', 'reverse_calibrated', 'one_step_calibrated',
                            'reverse_composed', 'one_step_composed')}
    record = dict(protocol_sha256=p['sha256'], reference_seed=reference_seed,
        diffusion_seed=repetition['diffusion_seed'], variant=variant, config=cfg,
        new_joint_neural_fits=sum(not r['adopted'] for r in fit_records),
        adopted_joint_neural_fits=sum(r['adopted'] for r in fit_records),
        calibration_fits=2, metrics=measured, files=files)
    base.write_json(record_path, record)
    print('DIFFUSION_VARIANT', reference_seed, variant, measured['reverse_composed'], flush=True)


def run(out=OUT, stage='screen', device='cuda:1'):
    out = Path(out).resolve()
    p = protocol(out)
    if device != 'cuda:1':
        raise ValueError('Only cuda:1 is authorized')
    if stage not in ('screen', 'confirm'):
        raise ValueError('Unknown stage')
    if stage == 'confirm':
        gate = base.check_seal(base.read_json(out / 'SCREEN_ASSESSMENT.json'))
        if gate['protocol_sha256'] != p['sha256'] or not gate['declared_gate']:
            raise ValueError('Screening failed')
    torch.set_num_threads(4)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    repetitions = p['repetitions'][:1] if stage == 'screen' else p['repetitions'][1:]
    a = selected_load(out / 'inputs/MiDrug/arrays.npz', TRAIN_KEYS | {
        'pairs_val', 'y_val', 'anchor_666', 'anchor_0', 'anchor_1'})
    for repetition in repetitions:
        for variant in VARIANTS:
            run_variant(out, p, a, variant, repetition, device)
        seed = repetition['reference_seed']
        directory = out / 'runs/LncDrug' / ('s' + str(seed))
        if (directory / 'result.json').exists():
            verify_files(directory, base.read_json(directory / 'result.json'))
            continue
        directory.mkdir(parents=True)
        identity = selected_load(out / 'inputs/LncDrug/arrays.npz', {'y_val', 'pairs_val', 'anchor_' + str(seed)})
        score = identity['anchor_' + str(seed)]
        np.savez_compressed(directory / 'scores.npz', y=identity['y_val'], pairs=identity['pairs_val'],
                            anchor=score, composed=score.copy())
        base.write_json(directory / 'result.json', dict(protocol_sha256=p['sha256'],
            reference_seed=seed, identity=True, new_joint_neural_fits=0,
            adopted_joint_neural_fits=0, calibration_fits=0, training_steps=0,
            metrics=metrics.metrics(identity['y_val'], score, identity['pairs_val']),
            files={'scores.npz': base.digest(directory / 'scores.npz')}))


def assess(out=OUT, stage='screen'):
    out = Path(out).resolve()
    p = protocol(out)
    if stage not in ('screen', 'historical'):
        raise ValueError('Unknown assessment stage')
    target = out / ('SCREEN_ASSESSMENT.json' if stage == 'screen' else 'HISTORICAL_ASSESSMENT.json')
    if target.exists():
        raise ValueError('Do not overwrite an existing assessment')
    repetitions = p['repetitions'][:1] if stage == 'screen' else p['repetitions']
    rows, ln_rows, result_hashes = [], [], {}
    new_fits = adopted_fits = 0
    for repetition in repetitions:
        seed = repetition['reference_seed']
        for variant in VARIANTS:
            directory = out / 'runs/MiDrug' / ('s' + str(seed)) / variant
            record = base.read_json(directory / 'result.json')
            if record['protocol_sha256'] != p['sha256']:
                raise ValueError('Protocol mismatch')
            verify_files(directory, record)
            result_hashes[str(directory / 'result.json')] = base.digest(directory / 'result.json')
            new_fits += record['new_joint_neural_fits']
            adopted_fits += record['adopted_joint_neural_fits']
            with np.load(directory / 'scores.npz', allow_pickle=False) as z:
                current = {key: metrics.metrics(z['y'], z[key], z['pairs']) for key in record['metrics']}
                for key, expected in record['metrics'].items():
                    for name, value in expected.items():
                        if abs(current[key][name] - value) > 1e-12:
                            raise ValueError('Stored metric mismatch')
                for key in READOUTS:
                    kind = variant + '_' + key
                    rows.append(dict(reference_seed=seed, kind=kind,
                        metrics=current[key + '_composed'],
                        delta={m: current[key + '_composed'][m] - current['anchor'][m]
                               for m in ('auc', 'ap')}))
        directory = out / 'runs/LncDrug' / ('s' + str(seed))
        record = base.read_json(directory / 'result.json')
        verify_files(directory, record)
        if record['protocol_sha256'] != p['sha256'] or not record['identity']:
            raise ValueError('Identity protocol mismatch')
        result_hashes[str(directory / 'result.json')] = base.digest(directory / 'result.json')
        with np.load(directory / 'scores.npz', allow_pickle=False) as z:
            np.testing.assert_array_equal(z['anchor'], z['composed'])
            measured = metrics.metrics(z['y'], z['composed'], z['pairs'])
            for name, value in record['metrics'].items():
                if abs(measured[name] - value) > 1e-12:
                    raise ValueError('Identity metric mismatch')
        ln_rows.append(dict(reference_seed=seed, metrics=measured, delta={'auc': 0., 'ap': 0.}))
    primary = [r for r in rows if r['kind'] == PRIMARY]
    mean_metrics = {m: float(np.mean([r['metrics'][m] for r in primary])) for m in ('auc', 'ap')}
    mean_delta = {m: float(np.mean([r['delta'][m] for r in primary])) for m in ('auc', 'ap')}
    positive = {m: sum(r['delta'][m] > 0 for r in primary) for m in ('auc', 'ap')}
    controls = {kind: {m: mean_metrics[m] - float(np.mean([r['metrics'][m] for r in rows
                                                        if r['kind'] == kind]))
                       for m in ('auc', 'ap')} for kind in CONTROLS}
    required = 1 if stage == 'screen' else 2
    performance = all(mean_delta[m] > 0 and positive[m] >= required for m in ('auc', 'ap'))
    mechanism = all(value > 0 for pair in controls.values() for value in pair.values())
    assessment = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=p['sha256'], stage=stage, primary=PRIMARY, primary_controls=list(CONTROLS),
        MiDrug=dict(mean_metrics=mean_metrics, mean_delta=mean_delta, positive_count=positive,
                    controls=controls, repetitions=rows), LncDrug=dict(repetitions=ln_rows,
                    maximum_metric_loss=0., identity=True),
        performance_gate=performance, mechanism_gate=mechanism, lnc_gate=True,
        declared_gate=bool(performance and mechanism), new_joint_neural_fits=new_fits,
        adopted_joint_neural_fits=adopted_fits, identity_scoring_records=len(ln_rows),
        all_stored_metrics_recomputed=True, result_sha256=result_hashes,
        independent_data_partitions=1, formal_confirmations_required_after_adoption=True,
        fixed_partitions_and_outer_still_required=True, confirmed_final_innovations=0,
        existing_default_replaced=False))
    base.write_json(target, assessment)
    print('DIFFUSION_ASSESSMENT', stage, mean_metrics, mean_delta, controls,
          'GATES', performance, mechanism, flush=True)
    return assessment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'assess'))
    parser.add_argument('--out', type=Path, default=OUT)
    parser.add_argument('--stage', default='screen')
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.out)
    elif args.command == 'run':
        run(args.out, args.stage, args.device)
    else:
        assess(args.out, args.stage)


if __name__ == '__main__':
    main()
