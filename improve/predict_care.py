"""Prediction API for a CARE-TabICL run that passed the frozen outer criteria.

Input pairs must be outside the saved fitting context. Supply the original 36
raw graph features and the five matching masked-graph feature views, using the
same entity indices and feature order as the saved run. No query label is read.
The saved training-only quantile transform is applied automatically when needed.
"""
import argparse
import gc
from pathlib import Path
import pickle

import numpy as np
import torch

from improve import tabular_foundation as base
from improve import tabicl_care as core
from improve import tabicl_care_modules as care
from improve import tabicl_care_refine as followup
from improve import tabicl_care_rank as rank_experiment
from improve import tabicl_care_rank_modules as rank_modules
from improve import tabicl_reli as reference


class CarePredictor:
    """One shared adapter with the original task-specific fitted TabICL contexts."""
    def __init__(self, run_dir=followup.DEFAULT_OUT, device='cuda:0', seed=666):
        self.run_dir, self.device, self.seed = Path(run_dir).resolve(), device, int(seed)
        header = base.read_json(self.run_dir / 'protocol.json')
        is_rank = header.get('architecture', '').startswith('CARE-RO:')
        self.backend = rank_experiment if is_rank else followup
        self.protocol = self.backend.protocol(self.run_dir)
        if self.seed not in self.protocol['seeds']:
            raise ValueError('No verified fitted context exists for this inference seed')
        self.selection = core.selection(self.run_dir, self.protocol)
        outcome = base.check_seal(base.read_json(self.run_dir / 'OUTCOME_outer.json'))
        if outcome['protocol_sha256'] != self.protocol['sha256'] or not outcome.get('passed_outer'):
            raise ValueError('This run has not passed the frozen outer non-degradation criteria')
        self.backend.collect_replay(self.run_dir, self.protocol, 'outer')
        self.recipe = self.selection['recipe']
        path = core.state_path(self.run_dir, self.recipe['mode'], self.recipe['trust'], self.recipe['step'])
        if is_rank:
            self.adapter = rank_modules.load_adapter(path, device)[0]
        else:
            original, _ = care.load_adapter(path, device)
            self.adapter = followup.RefinedAdapter(original, self.recipe['gate_power']).to(device)
        self.adapter.eval().requires_grad_(False)
        self._assets = {}

    def _task_assets(self, task):
        if task not in core.TASKS:
            raise ValueError('Unknown RNA task')
        if task not in self._assets:
            gp = base.read_json(Path(self.protocol['parent_dir']) / 'protocol.json')
            # The parent protocol validates the original feature cache and its
            # preprocessing state. Deserialize fitting identifiers only.
            core.base_protocol(self.protocol)
            folder = Path(gp['feature_reference_dir']) / 'features' / 'outer' / task
            manifest = base.read_json(folder / 'manifest.json')
            for name, expected in manifest['files'].items():
                if base.digest(folder / name) != expected:
                    raise ValueError('Saved feature transform/context changed')
            with np.load(folder / 'arrays.npz', allow_pickle=False) as archive:
                support, folds = archive['pairs_train'].copy(), archive['fold_ids'].copy()
            transformer = None
            if self.protocol['input_configs'][task]['input_view'] == 'quantile':
                with (folder / 'transform.pkl').open('rb') as stream:
                    transformer = pickle.load(stream)
            self._assets[task] = dict(support=support, folds=folds, transformer=transformer,
                evidence=core.evidence(self.protocol, task), feature_names=manifest['names'])
        return self._assets[task]

    def predict(self, task, pairs, raw_full, raw_views, *, return_members=False):
        from tabicl import TabICLClassifier
        assets = self._task_assets(task)
        pairs, raw_full, raw_views = np.asarray(pairs), np.asarray(raw_full), np.asarray(raw_views)
        if pairs.ndim != 2 or pairs.shape[0] != 2 or pairs.dtype.kind not in 'iu' or not pairs.shape[1]:
            raise ValueError('Expected nonempty integer pair columns')
        n = pairs.shape[1]
        if raw_full.shape != (n, 36) or raw_views.shape != (5, n, 36):
            raise ValueError('Expected N x 36 full features and 5 x N x 36 masked-graph views')
        if not np.isfinite(raw_full).all() or not np.isfinite(raw_views).all():
            raise ValueError('Graph features must be finite')
        if set(map(tuple, pairs.T)) & set(map(tuple, assets['support'].T)):
            raise ValueError('Prediction queries overlap the saved fitting context')
        # Validate endpoint ranges before loading a GPU model.
        for axis, matrix in enumerate((assets['evidence'].rna_kernel, assets['evidence'].drug_kernel)):
            if np.any(pairs[axis] < 0) or np.any(pairs[axis] >= len(matrix)):
                raise ValueError('Unknown entity index')
        full, views = raw_full.astype(np.float32), raw_views.astype(np.float32)
        if assets['transformer'] is not None:
            full = assets['transformer'].transform(full).astype(np.float32)
            views = assets['transformer'].transform(views.reshape(-1, 36)).astype(np.float32).reshape(5, n, 36)
        members, anchors = [], []
        # A benchmark reports means of per-seed metrics, which is different
        # from a new ensemble that averages probabilities across all seeds.
        # Predict with the explicitly requested, already evaluated seed.
        for seed in [self.seed]:
            by_kind = {}
            for kind in reference.kinds(task):
                if kind == 'full':
                    support = assets['support']
                    query = np.concatenate([full[None], views]) if task == 'LncDrug' else full[None]
                else:
                    fold = int(kind[-1])
                    support, query = assets['support'][:, assets['folds'] == fold], views[fold:fold + 1]
                relation, info = assets['evidence'].features(np.tile(pairs, (1, len(query))), support)
                dest = core.directory(self.run_dir, 'outer', task, seed, kind)
                core.checked_result(dest, self.protocol)
                clf = TabICLClassifier.load(dest / 'fitted.pkl', device=self.device)
                scores, _ = care.predict_variants(clf, query.reshape(-1, 36), relation, info,
                    {'selected': (self.adapter, self.recipe['strength'])})
                by_kind[kind] = {name: value.reshape(len(query), n) for name, value in scores.items()}
                del clf
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            combined = {}
            for name in ('selected', 'zero'):
                original = by_kind['full'][name][0]
                component = by_kind['full'][name][1:].mean(0) if task == 'LncDrug' else np.stack(
                    [by_kind['fold%d' % fold][name][0] for fold in range(5)]).mean(0)
                combined[name] = .5 * original + .5 * component
            members.append(combined['selected'])
            anchors.append(combined['zero'])
        members, anchors = np.stack(members), np.stack(anchors)
        if return_members:
            return dict(score=members.mean(0), per_seed=members, baseline=anchors.mean(0),
                        baseline_per_seed=anchors, seeds=np.array([self.seed]))
        return members.mean(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', default=followup.DEFAULT_OUT)
    parser.add_argument('--task', choices=core.TASKS, required=True)
    parser.add_argument('--input', required=True, help='NPZ with pairs, raw_full, raw_views')
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=666, help='Saved inference seed: 666, 0 or 1; default 666')
    args = parser.parse_args()
    path = Path(args.out)
    if path.exists():
        raise ValueError('Output already exists; choose a new path')
    with np.load(args.input, allow_pickle=False) as archive:
        inputs = {key: archive[key].copy() for key in ('pairs', 'raw_full', 'raw_views')}
    predictor = CarePredictor(args.run_dir, args.device, args.seed)
    output = predictor.predict(args.task, **inputs, return_members=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        np.savez_compressed(stream, **output, pairs=inputs['pairs'])
    print('SAVED', path)


if __name__ == '__main__':
    main()
