"""Check only the new variance ablation; retain prior graph-boundary evidence."""
from datetime import datetime, timezone
from pathlib import Path
import gc

import numpy as np
import torch

from improve import anchor_eigenbasis_experiment_v1 as experiment
from improve import eigenbasis_adam_v1 as optimizer
from improve import eigenbasis_graph_v1 as core
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
OUT = experiment.ROOT / 'results/eigenbasis_adam_smoke_v1_20260916'


def main():
    target = HERE / 'EIGENBASIS_SMOKE_V1.json'
    if target.exists() or OUT.exists():
        raise ValueError('Preserve smoke artifacts')
    unit_path = HERE / 'EIGENBASIS_UNIT_V1.json'
    unit = base.check_seal(base.read_json(unit_path))
    if not unit['passed']:
        raise ValueError('Unit checks failed')
    for path, digest in unit['source_sha256'].items():
        if base.digest(path) != digest:
            raise ValueError('Unit-checked source changed')
    prior_audit_path = HERE.parent / 'preconditioning_next_20260916/PRECONDITIONING_SMOKE_V1.json'
    prior_audit = base.check_seal(base.read_json(prior_audit_path))
    for path, digest in {**prior_audit['source_sha256'], **prior_audit['input_sha256']}.items():
        if base.digest(path) != digest:
            raise ValueError('Prior graph-boundary evidence changed')
    if prior_audit['core_parameters'] != core.parameters() or len(prior_audit['fold_audits']) != 40:
        raise ValueError('Graph recipe changed')
    OUT.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    p = dict(smoke=True, family='eigenbasis_adam', implementation_module='eigenbasis_graph_v1',
        source_sha256={'eigenbasis_graph_v1.py': base.digest(core.__file__)},
        core_parameters=core.parameters(), optimizer_parameters=optimizer.parameters(),
        kinds=list(core.KINDS), epochs=25, check_interval=20, weight_decay=.001, strong_weight=.1)
    records, inputs = [], {}
    for task in ('MiDrug', 'LncDrug'):
        source = experiment.ROOT / 'results/anchor_residual_energy_v2_20260916/inputs' / task / 'arrays.npz'
        inputs[str(source)] = base.digest(source)
        a = experiment.nested.load_selected(source, experiment.TRAIN_KEYS)
        query = torch.as_tensor(a['pairs_train'][:, :97]+a['pair_offset'][:, None], device='cuda:1')
        for kind in ('separable_variance',) if task == 'MiDrug' else core.KINDS:
            torch.cuda.reset_peak_memory_stats('cuda:1')
            model, _, trace, detail = experiment.fit(a, p, kind, .001, 986, 'cuda:1', fixed_epochs=25)
            dest = OUT / task / kind
            dest.mkdir(parents=True)
            torch.save(model.state_dict(), dest/'model.pt')
            torch.save(model.optimization_final_state, dest/'optimizer_final.pt')
            base.write_json(dest/'trace.json', trace)
            loaded, xs, graphs, _ = experiment.setup(a, p, kind, 'cuda:1')
            loaded.load_state_dict(torch.load(dest/'model.pt', map_location='cuda:1', weights_only=True))
            expected = experiment.predict(model, xs, graphs, query, p).cpu().numpy()
            reverse = experiment.predict(loaded, xs, graphs, query.flip(1), p).cpu().numpy()[::-1]
            chunks = np.concatenate([experiment.predict(loaded, xs, graphs, query[:, i:i+50], p)
                .cpu().numpy() for i in range(0, query.shape[1], 50)])
            for actual in (reverse, chunks):
                np.testing.assert_array_equal(actual, expected)
            if task == 'MiDrug':
                if detail['gradient_evaluations'] != 26 or detail['training_steps'] != 25:
                    raise ValueError('Incorrect gradient budget')
                if detail['final_optimizer_diagnostics']['basis_updates_min'] != 2:
                    raise ValueError('QR updates were not exercised')
            elif trace or detail['trained_neural_model'] or detail['gradient_evaluations']:
                raise ValueError('Identity is not a neural fit')
            if any('anchor_composed' in row for row in trace):
                raise ValueError('New smoke used evaluation labels')
            record = base.seal(dict(task=task, kind=kind, **detail, replay_max_abs=0.,
                peak_allocated_gib=torch.cuda.max_memory_allocated('cuda:1')/2**30,
                files={n: base.digest(dest/n) for n in ('model.pt', 'optimizer_final.pt', 'trace.json')}))
            base.write_json(dest/'RESULT.json', record)
            records.append(record)
            print('EIGENBASIS_SMOKE_RECORD', task, kind, detail['training_steps'], flush=True)
            del model, loaded, xs, graphs
            gc.collect(); torch.cuda.empty_cache()
    sources = [Path(__file__), Path(core.__file__), Path(optimizer.__file__), Path(experiment.__file__),
               unit_path, prior_audit_path]
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), device='cuda:1',
        core_parameters=core.parameters(), optimizer_parameters=optimizer.parameters(),
        new_short_neural_fits=1, identity_replays=3, formal_fits=0,
        performance_metrics_computed=False, parent_performance_previously_seen=True,
        unchanged_graph_fold_audits_reused=prior_audit['fold_audits'], new_fold_audits=0,
        records=records, input_sha256=inputs, source_sha256={str(p): base.digest(p) for p in sources},
        confirmed_final_innovations=0))
    base.write_json(target, result)
    print('EIGENBASIS_SMOKE_COMPLETE', result['sha256'], flush=True)


if __name__ == '__main__':
    main()
