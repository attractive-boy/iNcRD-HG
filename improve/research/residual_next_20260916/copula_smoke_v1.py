"""Training-only smoke and graph/posterior boundary audit; no performance metric."""
from datetime import datetime, timezone
from pathlib import Path
import gc

import numpy as np
import torch

from improve import anchor_copula_experiment_v1 as experiment
from improve import bernoulli_copula_graph_v1 as core
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
ROOT = experiment.ROOT


def main():
    result_path = HERE / 'COPULA_SMOKE_V1.json'
    output = ROOT / 'results/bernoulli_copula_smoke_v1_20260916'
    if result_path.exists() or output.exists():
        raise ValueError('Preserve prior smoke artifacts')
    output.mkdir()
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    p = dict(smoke=True, family='bernoulli_copula', implementation_module='bernoulli_copula_graph_v1',
        source_sha256={'bernoulli_copula_graph_v1.py': base.digest(core.__file__)},
        core_parameters=core.parameters(), kinds=list(core.KINDS), epochs=5, check_interval=5,
        weight_decay=.001, private_rng_offset=29015, strong_weight=.1)
    fits, audits = [], []
    for task in ('MiDrug', 'LncDrug'):
        source = ROOT / 'results/anchor_residual_energy_v2_20260916/inputs' / task / 'arrays.npz'
        a = experiment.nested.load_selected(source, experiment.TRAIN_KEYS)
        # Query identities only. No evaluation outcome or saved prediction is read.
        queries = experiment.nested.load_selected(source, ('pairs_val',))['pairs_val']
        pairs = torch.as_tensor(queries + a['pair_offset'][:, None], device='cuda:1')
        marginal = torch.zeros(pairs.shape[1], dtype=torch.float64, device='cuda:1')
        for kind in core.KINDS:
            dest = output / task / kind; dest.mkdir(parents=True)
            model, _, trace, detail = experiment.fit(a, p, kind, .001, 932, 'cuda:1', fixed_epochs=5)
            torch.save(model.state_dict(), dest / 'model.pt')
            base.write_json(dest / 'trace.json', trace)
            loaded, xs, graphs, _ = experiment.setup(a, p, kind, 'cuda:1')
            loaded.load_state_dict(torch.load(dest / 'model.pt', map_location='cuda:1', weights_only=True))
            value = experiment.predict(model, xs, graphs, pairs, p, marginal).cpu().numpy()
            replay = experiment.predict(loaded, xs, graphs, pairs.flip(1), p, marginal.flip(0)).cpu().numpy()[::-1]
            chunks = np.concatenate([experiment.predict(loaded, xs, graphs, pairs[:, i:i+800], p,
                marginal[i:i+800]).cpu().numpy() for i in range(0, pairs.shape[1], 800)])
            for actual in (replay, chunks):
                np.testing.assert_allclose(value, actual, atol=3e-5, rtol=3e-5)
            convergence = []
            if kind == 'full':
                with torch.no_grad(), experiment.precision.deterministic_evaluation():
                    loaded.eval()
                    for graph in graphs:
                        h = loaded.encode(xs, graph)
                        residual, factor, diagonal = loaded.pair_parameters(h, graph.support_pairs)
                        _, _, audit = core.laplace_posterior(factor, diagonal,
                            core.latent_threshold(graph.support_logits + residual), graph.support_labels,
                            p['core_parameters']['posterior_steps'])
                        convergence.append(float(audit['gradient_max']))
            fits.append(dict(task=task, kind=kind, steps=5,
                replay_max_abs=float(max(abs(value-replay).max(), abs(value-chunks).max())),
                posterior_gradient_max_by_fold=convergence,
                model_sha256=base.digest(dest / 'model.pt'), trace_sha256=base.digest(dest / 'trace.json')))
            print('COPULA_SMOKE', fits[-1], flush=True)
            del model, loaded, xs, graphs
            gc.collect(); torch.cuda.empty_cache()
        paths = [('historical', source)] + [(split, ROOT / 'results/mechanism_validation_20260915/inputs' /
            task / str(split) / 'arrays.npz') for split in (61453, 7919, 104729)]
        for split, path in paths:
            training = experiment.nested.load_selected(path, experiment.nested.TRAIN_KEYS)
            for fold in range(5):
                omitted = np.flatnonzero(training['folds'] == fold)
                graph = core.context(training, omitted, p['core_parameters'], 'cpu')
                forbidden = training['pairs_train'][:, omitted] + training['pair_offset'][:, None]
                physical = set(map(tuple, graph.edges.numpy().T))
                support = set(map(tuple, graph.support_pairs.numpy().T))
                overlap = sum(tuple(q) in physical or tuple(q[::-1]) in physical or tuple(q) in support
                              for q in forbidden.T)
                if overlap:
                    raise ValueError('Query fact entered support')
                changed = {k: v.copy() for k, v in training.items()}
                changed['y_train'][omitted] = 1 - changed['y_train'][omitted]
                changed['anchor_oof'] = np.full(len(changed['y_train']), 900.)
                alternative = core.context(changed, omitted, p['core_parameters'], 'cpu')
                for name in ('edges', 'relation', 'support_pairs', 'support_labels', 'support_logits'):
                    torch.testing.assert_close(getattr(graph, name), getattr(alternative, name), atol=0, rtol=0)
                audits.append(dict(task=task, split=split, fold=fold, forbidden_overlap=int(overlap),
                    support_rows=len(graph.support_labels), query_labels_and_global_offsets_invariant=True,
                    nuisance_fit=graph.nuisance_audit))
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(path): base.digest(path) for path in (Path(core.__file__),
            Path(experiment.__file__), Path(__file__))}, fits=fits, fold_audits=audits,
        device='cuda:1', evaluation_labels_read=False, performance_metrics_computed=False,
        new_formal_fit_count=0, confirmed_final_innovations=0))
    base.write_json(result_path, result)
    print('COPULA_SMOKE_COMPLETE', len(fits), len(audits), flush=True)


if __name__ == '__main__':
    main()
