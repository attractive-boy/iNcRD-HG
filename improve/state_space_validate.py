"""Versioned fixed-step state-space validation on the registered partitions.

The original graph validator dispatches only block and geometry. This adapter
uses its frozen selection and reporting rules without changing sealed sources.
Its own source manifest is frozen before any state-space fresh predictions.
"""
import argparse
from datetime import datetime, timezone
import gc
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch.nn import functional as F

from improve import graph_mechanism_validate as validation
from improve import state_space_experiment as method

base = validation.base
LABEL = 'state_space'
EXECUTOR = 'STATE_SPACE_EXECUTOR_V1.json'
OWN = ('state_space_validate.py', 'test_state_space_validate.py')
TRAIN_KEYS = frozenset(('pairs_train', 'y_train', 'edges', 'relation',
    'target_codes', 'node_counts', 'pair_offset', 'nd', 'folds',
    'x_0', 'x_1', 'x_2', 'x_3'))


def freeze(out, parent=method.OUT):
    out, parent = Path(out).resolve(), Path(parent).resolve()
    if (out / EXECUTOR).exists() or (out / 'runs' / LABEL).exists():
        raise ValueError('State-space validation has already been registered')
    registration = base.read_json(out / 'STATE_SPACE_REGISTRATION.json')
    if registration['parent_protocol_sha256'] != base.digest(parent / 'protocol.json'):
        raise ValueError('Historical candidate differs from its registration')
    if not (out / 'candidates' / (LABEL + '.json')).exists():
        validation.freeze(out, parent, LABEL, primary='full')
    dp, cp = validation.candidate(out, LABEL)
    assert cp['parent'] == str(parent) and cp['primary_kind'] == 'full'
    assert cp['parent_protocol_sha256'] == registration['parent_protocol_sha256']
    record = base.seal(dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        candidate_sha256=cp['sha256'], data_protocol_sha256=dp['sha256'],
        registration_sha256=base.digest(out / 'STATE_SPACE_REGISTRATION.json'),
        source_sha256={n: base.digest(validation.ROOT / n) for n in OWN},
        executor='State-space setup only; fixed-step BCE training; deterministic stable inference.',
        selection='No model or checkpoint selection from fresh query predictions.',
        original_test_access=False, fresh_state_space_predictions_available=False))
    snapshot = out / 'candidate_sources' / 'state_space_executor_v1'
    snapshot.mkdir(parents=True)
    for name in OWN:
        shutil.copy2(validation.ROOT / name, snapshot / name)
    base.write_json(out / EXECUTOR, record)
    print('STATE_SPACE_EXECUTOR_FROZEN', record['sha256'], flush=True)


def checked(out):
    out = Path(out)
    dp, cp = validation.candidate(out, LABEL)
    record = base.check_seal(base.read_json(out / EXECUTOR))
    if cp['method'] != 'state_space':
        raise ValueError('This executor only accepts state-space candidates')
    assert record['candidate_sha256'] == cp['sha256']
    assert record['data_protocol_sha256'] == dp['sha256']
    assert record['registration_sha256'] == base.digest(out / 'STATE_SPACE_REGISTRATION.json')
    for name, expected in record['source_sha256'].items():
        if base.digest(validation.ROOT / name) != expected:
            raise ValueError('Sealed executor source changed: ' + name)
    return dp, cp, record


def train_fixed(training, parameters, kind, recipe, seed, device):
    unknown = set(training) - TRAIN_KEYS
    if unknown:
        raise ValueError('Only fitting data may enter state-space training: ' + ', '.join(sorted(unknown)))
    if kind not in method.KINDS:
        raise ValueError('Unknown state-space control')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, xs, graphs, folds = method.setup(training, parameters, kind, device)
    pairs = torch.as_tensor(training['pairs_train'] + training['pair_offset'][:, None], device=device)
    labels = torch.as_tensor(training['y_train'], device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=recipe['lr'],
                                 weight_decay=parameters['weight_decay'])
    trace, start = [], time.monotonic()
    for step in range(1, recipe['epochs'] + 1):
        fold = (step - 1) % len(graphs)
        ids = folds[fold]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(xs, graphs[fold], pairs[:, ids])
        loss = F.binary_cross_entropy_with_logits(prediction, labels[ids])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite fixed-step state-space loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        if step % 20 == 0 or step == recipe['epochs']:
            trace.append(dict(step=step, training_loss=float(loss.detach())))
    return model, trace, dict(epochs=recipe['epochs'], seconds=time.monotonic() - start,
                              fresh_query_outcomes_used_for_fitting=False)


def run(out, task, device='cuda:1'):
    out = Path(out)
    dp, cp, executor = checked(out)
    p = cp['training_parameters']
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(False)
    for split in dp['split_seeds']:
        a, _ = validation.inputs.arrays(out, dp, task, split)
        rd = out / 'references' / task / str(split)
        reference_record = base.read_json(rd / 'result.json')
        validation.verify_artifacts(rd, reference_record)
        assert reference_record['data_protocol_sha256'] == dp['sha256']
        ref = validation.block.load(rd / 'scores.npz')
        np.testing.assert_array_equal(a['y_val'], ref['y'])
        np.testing.assert_array_equal(a['pairs_val'], ref['pairs'])
        training = {k: v for k, v in a.items() if k in TRAIN_KEYS}
        for kind, recipe in cp['recipes'][task].items():
            dest = out / 'runs' / LABEL / task / str(split) / kind
            dest.mkdir(parents=True, exist_ok=True)
            if (dest / 'result.json').exists():
                r = base.read_json(dest / 'result.json')
                assert r['candidate_sha256'] == cp['sha256']
                assert r['executor_sha256'] == executor['sha256']
                validation.verify_artifacts(dest, r)
                continue
            model, trace, detail = train_fixed(training, p, kind, recipe, cp['model_seed'], device)
            torch.save(model.state_dict(), dest / 'model.pt')
            base.write_json(dest / 'trace.json', trace)
            reloaded, xs, graphs, _ = method.setup(training, p, kind, device)
            reloaded.load_state_dict(torch.load(dest / 'model.pt', map_location=device, weights_only=True))
            query = torch.as_tensor(a['pairs_val'] + a['pair_offset'][:, None], device=device)

            def predict(fitted, pairs):
                return validation.precision.geometry_prediction(fitted, xs, graphs, pairs).cpu().numpy()

            pred = predict(model, query)
            replay = predict(reloaded, query.flip(1))[::-1]
            chunks = np.concatenate([predict(reloaded, query[:, i:i + 800])
                                     for i in range(0, query.shape[1], 800)])
            for value in (replay, chunks):
                np.testing.assert_allclose(pred, value, atol=3e-5, rtol=3e-5)
            scores = dict(y=a['y_val'], pairs=a['pairs_val'], standalone=pred,
                          anchor=ref['anchor'], tabicl=ref['tabicl'])
            for name in ('anchor', 'tabicl'):
                scores[name + '_composed'] = (1 - cp['strong_weight']) * scores[name] + cp['strong_weight'] * pred
            np.savez_compressed(dest / 'scores.npz', **scores)
            result = dict(candidate_sha256=cp['sha256'], executor_sha256=executor['sha256'],
                data_protocol_sha256=dp['sha256'], task=task, split=split, kind=kind,
                seed=cp['model_seed'], recipe=recipe, **detail,
                replay_max_abs=float(max(abs(pred - replay).max(), abs(pred - chunks).max())),
                metrics={n: validation.block.base.metrics(scores['y'], v, scores['pairs'])
                         for n, v in scores.items() if n not in ('y', 'pairs')},
                files={n: base.digest(dest / n) for n in ('scores.npz', 'model.pt', 'trace.json')})
            base.write_json(dest / 'result.json', result)
            print('FRESH_VERIFIED', LABEL, task, split, kind, result['metrics']['anchor_composed'], flush=True)
            del model, reloaded, xs, graphs
            gc.collect()
            torch.cuda.empty_cache()


def report(out):
    dp, cp, executor = checked(out)
    for task in dp['tasks']:
        for split in dp['split_seeds']:
            for kind in cp['recipes'][task]:
                path = Path(out) / 'runs' / LABEL / task / str(split) / kind / 'result.json'
                r = base.read_json(path)
                assert r['executor_sha256'] == executor['sha256']
    validation.report(out, LABEL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('freeze', 'run', 'report'))
    parser.add_argument('--out', default=str(validation.data_source.OUT))
    parser.add_argument('--parent', default=str(method.OUT))
    parser.add_argument('--task', choices=('MiDrug', 'LncDrug'))
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if args.command == 'freeze':
        freeze(args.out, args.parent)
    elif args.command == 'run':
        if args.task is None:
            parser.error('run requires --task')
        run(args.out, args.task, args.device)
    else:
        report(args.out)
