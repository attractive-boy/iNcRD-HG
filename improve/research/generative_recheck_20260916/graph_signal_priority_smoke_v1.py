"""Three short joint fits and support-only audits, without evaluation outcomes."""
from datetime import datetime, timezone
from pathlib import Path
import gc

import numpy as np
import torch

from improve import graph_signal_priority_v1 as study

HERE, ROOT, base = study.HERE, study.ROOT, study.base
OUT = ROOT / 'results/graph_signal_priority_smoke_v1_20260916'


def main():
    target = HERE / 'GRAPH_SIGNAL_PRIORITY_SMOKE_V1.json'
    if OUT.exists() or target.exists():
        raise ValueError('Preserve previous smoke files')
    OUT.mkdir()
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = study.old.OUT / 'inputs/MiDrug/inner/arrays.npz'
    training = study.selected_load(source, study.TRAIN_KEYS)
    records, audits, inputs = [], [], {str(source): base.digest(source)}
    for variant in study.VARIANTS:
        cfg = study.configuration(variant)
        cfg['epochs'] = 5
        torch.cuda.reset_peak_memory_stats('cuda:1')
        model, trace, detail = study.fit(training, 0, cfg, 913, 'cuda:1', smoke=True)
        directory = OUT / variant
        directory.mkdir()
        torch.save(model.state_dict(), directory / 'model.pt')
        base.write_json(directory / 'trace.json', trace)
        graph = study.support(training, 0)
        loaded = study.old.construct(graph, training['kr'], training['kd'], cfg, 'cuda:1')
        loaded.load_state_dict(torch.load(directory / 'model.pt', map_location='cuda:1', weights_only=True))
        prediction = study.predict(model, graph)
        replay = study.predict(loaded, graph)
        query = training['pairs_train'][:, training['folds'] == 0]
        for kind in study.READOUTS:
            np.testing.assert_array_equal(prediction[kind], replay[kind])
            scores = prediction[kind][query[0], query[1]]
            reversed_score = replay[kind][query[0, ::-1], query[1, ::-1]][::-1]
            batched = np.concatenate([replay[kind][query[0, start:start + 97],
                                                 query[1, start:start + 97]]
                                      for start in range(0, query.shape[1], 97)])
            np.testing.assert_array_equal(scores, reversed_score)
            np.testing.assert_array_equal(scores, batched)
            if not np.isfinite(scores).all():
                raise ValueError('Nonfinite smoke prediction')
        record = dict(variant=variant, **detail, replay_max_abs=0.,
            peak_allocated_gib=torch.cuda.max_memory_allocated('cuda:1') / 2**30,
            files={name: base.digest(directory / name) for name in ('model.pt', 'trace.json')})
        base.write_json(directory / 'result.json', record)
        records.append(record)
        print('PRIORITY_DIFFUSION_SMOKE', variant, detail['training_steps'], detail['seconds'], flush=True)
        del model, loaded
        gc.collect()
        torch.cuda.empty_cache()
    for task in ('MiDrug', 'LncDrug'):
        paths = [('historical', study.old.OUT / 'inputs' / task / 'inner/arrays.npz')]
        paths += [(split, ROOT / 'results/mechanism_validation_20260915/inputs' /
                   task / str(split) / 'arrays.npz') for split in (61453, 7919, 104729)]
        kernel = study.selected_load(study.old.OUT / 'inputs' / task / 'inner/arrays.npz', {'kr', 'kd'})
        for split, path in paths:
            inputs[str(path)] = base.digest(path)
            if task == 'LncDrug':
                audits += [dict(task=task, split=split, fold=fold, identity=True,
                                new_graph_constructed=False, new_neural_fit=False) for fold in range(5)]
                continue
            a = study.selected_load(path, {'pairs_train', 'y_train', 'folds'})
            a.update(kernel)
            for fold in range(5):
                graph = study.support(a, fold)
                query = a['pairs_train'][:, a['folds'] == fold]
                if graph[query[0], query[1]].any():
                    raise ValueError('Query edge in support')
                changed = {key: value.copy() for key, value in a.items()}
                changed['y_train'][a['folds'] == fold] = 1 - changed['y_train'][a['folds'] == fold]
                other = study.support(changed, fold)
                np.testing.assert_array_equal(graph, other)
                # Denoiser training takes only this graph and the unchanged exogenous priors.
                audits.append(dict(task=task, split=split, fold=fold, identity=False,
                    forbidden_overlap=0, heldout_query_label_invariant=True,
                    fitting_positive_edges=int(graph.sum())))
    report = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        records=records, fold_audits=audits, new_short_joint_fits=3, formal_fits=0,
        device='cuda:1', evaluation_labels_read=False, performance_metrics_computed=False,
        source_sha256={str(path): base.digest(path) for path in
            (Path(__file__), Path(study.__file__), ROOT / 'test_graph_signal_priority_v1.py')},
        input_sha256=inputs, confirmed_final_innovations=0))
    base.write_json(target, report)
    print('GRAPH_SIGNAL_PRIORITY_SMOKE_COMPLETE', report['sha256'], flush=True)


if __name__ == '__main__':
    main()
