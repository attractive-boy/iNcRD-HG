"""Separate fixed-step executor for neural neighborhood-union pair decoding.

Freeze, integrity and reporting are architecture-neutral helpers from the
residual validator. Its three-family fitting dispatch is deliberately unused.
"""
import argparse
import gc
import os
from pathlib import Path

import numpy as np
import torch

from improve import residual_graph_validate as shared
from improve import offset_graph_candidates_v4 as experiment


def run(out, family, task, device):
    out = Path(out).resolve()
    dp, cp = shared.checked(out, family)
    p, label = cp['training_parameters'], cp['label']
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for split in dp['split_seeds']:
        a, _ = shared.shared.inputs.arrays(out, dp, task, split)
        rd = out / 'references' / task / str(split)
        r = shared.base.read_json(rd / 'result.json')
        shared.shared.verify_artifacts(rd, r)
        if r['data_protocol_sha256'] != dp['sha256']:
            raise ValueError('Reference data mismatch')
        reference = experiment.common.load(rd / 'scores.npz')
        for name, key in [('y', 'y_val'), ('pairs', 'pairs_val')]:
            np.testing.assert_array_equal(reference[name], a[key])
        training = {k: v for k, v in a.items() if k in shared.BASE_TRAIN_KEYS}
        training.update(experiment.common.load(out / shared.CACHE / task / str(split) / 'offsets.npz'))
        null = experiment.residual.offset.offset_prediction(training, a['pairs_val'])
        for kind, recipe in cp['recipes'][task].items():
            dest = out / 'runs' / label / task / str(split) / kind
            dest.mkdir(parents=True, exist_ok=True)
            if (dest / 'result.json').exists():
                r = shared.base.read_json(dest / 'result.json')
                if r['candidate_sha256'] != cp['sha256']:
                    raise ValueError('Existing candidate result mismatch')
                shared.shared.verify_artifacts(dest, r)
                continue
            model, _, trace, detail = experiment.fit(training, p, kind, recipe['lr'],
                cp['model_seed'], device, fixed_epochs=recipe['epochs'])
            torch.save(model.state_dict(), dest / 'model.pt')
            shared.base.write_json(dest / 'trace.json', trace)
            loaded, xs, graphs, _ = experiment.setup(training, p, kind, device)
            loaded.load_state_dict(torch.load(dest / 'model.pt', weights_only=True, map_location=device))
            query = torch.as_tensor(a['pairs_val'] + a['pair_offset'][:, None], device=device)
            def predict(fitted, pairs):
                return experiment.precision.geometry_prediction(fitted, xs, graphs, pairs).cpu().numpy()
            prediction = predict(model, query)
            replay = predict(loaded, query.flip(1))[::-1]
            chunks = np.concatenate([predict(loaded, query[:, i:i + 800]) for i in range(0, query.shape[1], 800)])
            for values in (replay, chunks):
                np.testing.assert_allclose(prediction, values, atol=3e-5, rtol=3e-5)
            scores = dict(y=a['y_val'], pairs=a['pairs_val'], standalone=prediction,
                          offset=null, anchor=reference['anchor'], tabicl=reference['tabicl'])
            for name in ('anchor', 'tabicl'):
                scores[name + '_composed'] = .9 * scores[name] + .1 * prediction
            scores['offset_composed'] = .9 * scores['anchor'] + .1 * null
            np.savez_compressed(dest / 'scores.npz', **scores)
            result = dict(candidate_sha256=cp['sha256'], data_protocol_sha256=dp['sha256'],
                task=task, split=split, kind=kind, seed=cp['model_seed'], recipe=recipe, **detail,
                replay_max_abs=float(max(abs(prediction - replay).max(), abs(prediction - chunks).max())),
                metrics={n: experiment.common.base.metrics(scores['y'], v, scores['pairs'])
                         for n, v in scores.items() if n not in ('y', 'pairs')},
                files={n: shared.base.digest(dest / n) for n in ('model.pt', 'trace.json', 'scores.npz')})
            shared.base.write_json(dest / 'result.json', result)
            print('PARTITION_VERIFIED', label, task, split, kind, result['metrics']['anchor_composed'], flush=True)
            del model, loaded, xs, graphs
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('freeze', 'run', 'report'))
    parser.add_argument('--family', choices=tuple(experiment.LIBRARIES), required=True)
    parser.add_argument('--out', default=str(shared.shared.data_source.OUT))
    parser.add_argument('--task', choices=('MiDrug', 'LncDrug'))
    parser.add_argument('--device', default='cuda:1')
    args = parser.parse_args()
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if args.command == 'freeze':
        shared.freeze(args.out, args.family)
    elif args.command == 'run':
        run(args.out, args.family, args.task, args.device)
    else:
        shared.report(args.out, args.family)
