"""MPHGNN-style gene-mediated metapath channel for the accepted TabICLv2 reference.

Label-free (no ncRNA-drug labels are read). Builds the <RNA -> gene1 ~ gene2 -> drug>
higher-order semantic features on the frozen evidence graph, then evaluates the
frozen TabICLv2 candidate 0.5*graph + 0.5*(graph+tiered+new) against the accepted
reference on the historical inner partition for both MiDrug and LncDrug.

Run with /data1/licheng/incrd-hg/.venv-tabicl/bin/python.
"""
import argparse
from datetime import datetime, timezone
import gc
from pathlib import Path
import time

import numpy as np
from scipy import sparse

from improve import tabular_foundation as base
from improve import tabicl_context_program as accepted
from improve import tabicl_context_program_modules as evidence
from improve import tabicl_pathway as operations

ROOT = Path('/data1/licheng/incrd-hg/iNcRD-HG')
EVIDENCE = ROOT/'improve/results/tabicl_pathway_context_inputs_20260913'
ACC = ROOT/'improve/results/tabicl_pathway_context_20260913'
OUT = ROOT/'improve/results/metapath_channel_20260915'
TIERS = ('functional', 'binding', 'association')
WEIGHTS = dict(functional=1., binding=.25, association=.1)
FEATURE_NAMES = (
    'mp_gene2_dot', 'mp_gene2_cos', 'mp_gene2_shared_log', 'mp_gene2_drug_cover',
    'mp_gene2_rna_cover', 'mp_gene2_max', 'mp_func_dot', 'mp_bind_dot', 'mp_assoc_dot',
    'mp_func_cos', 'mp_gene2_program_cos', 'mp_rev_dot', 'mp_rev_cos', 'mp_gene2_count',
)
VARIANTS = ('metapath', 'metapath_shuffled', 'metapath_random')


def binary(m):
    m = sparse.csr_matrix(m, dtype=np.float64).copy()
    m.eliminate_zeros()
    m.data[:] = 1.
    return m


def l2_rows(m):
    m = sparse.csr_matrix(m, dtype=np.float64)
    norm = np.sqrt(np.asarray(m.multiply(m).sum(1)).ravel())
    inv = np.divide(1., norm, out=np.zeros_like(norm), where=norm > 0)
    return sparse.diags(inv).dot(m).tocsr()


class MetapathFeatures:
    """Same operators for both RNA types; no RNA-drug labels are used."""

    def __init__(self, seed=20260915):
        import json
        self.seed = seed
        dictionary = json.loads((EVIDENCE/'dictionary.json').read_text())
        self.tasks = tuple(dictionary['rna_names'])
        self.gp = binary(sparse.load_npz(EVIDENCE/'gene_pathway.npz'))
        self.dg = binary(sparse.load_npz(EVIDENCE/'drug_gene.npz'))
        self.gp_norm = l2_rows(self.gp)
        self.dg_norm = l2_rows(self.dg)
        self.tiers = {t: {k: binary(sparse.load_npz(EVIDENCE/(t+'_'+k+'.npz'))) for k in TIERS} for t in self.tasks}
        self.tables = {}
        active = np.asarray((self.dg > 0).sum(0)).ravel() > 0
        self.drug_genes = np.flatnonzero(active)
        self.B = self.drug_genes
        self.dgB = self.dg[:, self.B].toarray().astype(np.float32)

    def _cross_similarity(self, A):
        """Pathway-cosine between RNA genes A and drug genes B, dense |A|x|B|."""
        return np.asarray((self.gp_norm[A] @ self.gp_norm[self.B].T).todense(), dtype=np.float32)

    def _reach(self, W, A, C):
        return (sparse.csr_matrix(W[:, A]) @ sparse.csr_matrix(C)).toarray().astype(np.float32)

    def _features(self, task, W, A, C, reverse_C):
        nr = W.shape[0]
        reach = self._reach(W, A, C)                     # nr x |B|
        rn = np.linalg.norm(reach, axis=1, keepdims=True)
        dl = self.dgB                                     # nd x |B|
        dn = np.linalg.norm(dl, axis=1, keepdims=True)
        dot = reach @ dl.T
        cos = dot / np.maximum(rn * dn.T, 1e-12)
        rs = (reach > 0).astype(np.float32)
        shared = rs @ dl.T
        drug_cov = shared / np.maximum(dl.sum(1)[None, :], 1.)
        rna_cov = shared / np.maximum(rs.sum(1)[:, None], 1.)
        mx = np.broadcast_to(reach.max(1)[:, None], (nr, dl.shape[0]))
        count = np.broadcast_to(rs.sum(1)[:, None], (nr, dl.shape[0]))
        # reverse direction: drug genes -> RNA genes
        rev = (sparse.csr_matrix(dl) @ sparse.csr_matrix(reverse_C)).toarray().astype(np.float32)  # nd x |A|
        rnA = np.linalg.norm(rev, axis=1)
        wA = W[:, A].toarray().astype(np.float32)
        wnA = np.linalg.norm(wA, axis=1)
        rev_dot = rev @ wA.T
        rev_cos = rev_dot / np.maximum(wnA[None, :] * rnA[:, None], 1e-12)
        # pathway-program cosine between gene2-reach and drug gene program
        proj = self.proj[self.B]
        reach_p = l2_rows(sparse.csr_matrix(reach)) @ proj
        drug_p = l2_rows(sparse.csr_matrix(dl)) @ proj
        prog = np.asarray((reach_p @ drug_p.T).todense(), dtype=np.float32)
        return dict(mp_gene2_dot=dot, mp_gene2_cos=cos, mp_gene2_shared_log=np.log1p(shared),
                    mp_gene2_drug_cover=drug_cov, mp_gene2_rna_cover=rna_cov, mp_gene2_max=mx,
                    mp_func_dot=dot, mp_bind_dot=dot, mp_assoc_dot=dot, mp_func_cos=cos,
                    mp_gene2_program_cos=prog, mp_rev_dot=rev_dot.T, mp_rev_cos=rev_cos.T,
                    mp_gene2_count=count)

    def table(self, task, variant):
        key = (task, variant)
        if key in self.tables:
            return self.tables[key]
        layers = self.tiers[task]
        W = sparse.csr_matrix(layers['functional'].shape, dtype=np.float64)
        for k in TIERS:
            W = W.maximum(layers[k]*WEIGHTS[k])
        A = np.flatnonzero(np.asarray((W > 0).sum(0)).ravel() > 0)
        if variant == 'metapath_shuffled':
            rng = np.random.default_rng(self.seed)
            perm = rng.permutation(self.gp.shape[0])
            gp_s = self.gp_norm[perm]
            C = np.asarray((gp_s[A] @ self.gp_norm[self.B].T).todense(), dtype=np.float32)
            revC = np.asarray((self.gp_norm[self.B] @ gp_s[A].T).todense(), dtype=np.float32)
        else:
            C = self._cross_similarity(A)
            revC = np.asarray((self.gp_norm[self.B] @ self.gp_norm[A].T).todense(), dtype=np.float32)
        feats = self._features(task, W, A, C, revC)
        if variant == 'metapath_random':
            rng = np.random.default_rng(self.seed + 1)
            feats = {k: rng.standard_normal(v.shape).astype(np.float32) for k, v in feats.items()}
        # tier-specific dot/cos
        for tier, newdot, newcos in (('functional', 'mp_func_dot', 'mp_func_cos'),
                                     ('binding', 'mp_bind_dot', None), ('association', 'mp_assoc_dot', None)):
            Wt = layers[tier]
            At = np.flatnonzero(np.asarray((Wt > 0).sum(0)).ravel() > 0)
            Ct = np.asarray((self.gp_norm[At] @ self.gp_norm[self.B].T).todense(), dtype=np.float32)
            reach = self._reach(Wt, At, Ct)
            feats[newdot] = reach @ self.dgB.T
            if newcos:
                rn = np.linalg.norm(reach, axis=1, keepdims=True)
                dn = np.linalg.norm(self.dgB, axis=1, keepdims=True)
                feats[newcos] = (reach @ self.dgB.T) / np.maximum(rn * dn.T, 1e-12)
        table = np.stack([feats[n] for n in FEATURE_NAMES], axis=-1).astype(np.float32)
        if not np.isfinite(table).all():
            raise ValueError('Nonfinite metapath feature')
        self.tables[key] = table
        return table

    @property
    def proj(self):
        if not hasattr(self, '_proj'):
            gp = binary(self.gp)
            deg = np.asarray(gp.sum(1)).ravel()
            size = np.asarray(gp.sum(0)).ravel()
            inv_d = np.divide(1., np.sqrt(deg), out=np.zeros_like(deg), where=deg > 0)
            inv_s = np.divide(1., np.sqrt(size), out=np.zeros_like(size), where=size > 0)
            self._proj = (sparse.diags(inv_d) @ gp @ sparse.diags(inv_s)).tocsr()
        return self._proj

    def transform(self, pairs, table):
        pairs = np.asarray(pairs)
        return table[pairs[0], pairs[1]].copy()


def feature_transform(enc, task, pairs, variant):
    return enc.transform(np.asarray(pairs), enc.table(task, variant))


def augmented_score(out, p, a, task, variant, seed):
    """Compose the TabICL branch across kinds using the accepted graph recipe."""
    import numpy as _np
    scores = {}
    for kind in operations.kinds(p, task):
        dest = out/'inner'/task/variant/('s'+str(seed))/kind
        with _np.load(dest/'scores.npz', allow_pickle=False) as z:
            _np.testing.assert_array_equal(z['y_val'], a['y_val'])
            _np.testing.assert_array_equal(z['pairs_val'], a['pairs_val'])
            scores[kind] = z['views'].copy()
    recipe = p['graph_recipes'][task]
    full = scores['full'][0]
    if recipe['arm'] == 'frozen':
        return full
    if recipe['arm'] == 'query_bag':
        extra = scores['full'][1:].mean(0)
    else:
        extra = np.stack([scores['fold'+str(i)][0] for i in range(5)]).mean(0)
    return (1-recipe['weight'])*full + recipe['weight']*extra


def run(args):
    import torch
    from tabicl import TabICLClassifier
    out = Path(args.out_dir)
    p = accepted.protocol(ACC)
    enc = MetapathFeatures(seed=args.seed)
    tiered = evidence.ContextPrograms.load(p['inputs_dir'])
    if out.exists() and any(out.iterdir()) and not args.force:
        raise ValueError('Use a new output directory or --force')
    out.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(args.device)
    variants = args.variants.split(',')
    tasks = args.tasks.split(',')
    for task in tasks:
        a, gp = accepted.stage_data(out, p, task, 'inner')
        tp = base.read_json(Path(gp['parent_dir'])/'protocol.json')
        original_train = tiered.transform(task, a['pairs_train'], 'tiered')
        original_query = tiered.transform(task, a['pairs_val'], 'tiered')
        for variant in variants:
            table = enc.table(task, variant)
            new_train = enc.transform(a['pairs_train'], table)
            new_query = enc.transform(a['pairs_val'], table)
            train = np.concatenate([original_train, new_train], axis=1)
            query = np.concatenate([original_query, new_query], axis=1)
            for seed in p['seeds']:
                for kind in operations.kinds(p, task):
                    dest = out/'inner'/task/variant/('s'+str(seed))/kind
                    if (dest/'result.json').exists():
                        continue
                    start = time.monotonic()
                    x, y, q = operations.job_arrays(p, task, a, kind, train, query)
                    if not np.isfinite(x).all() or not np.isfinite(q).all():
                        raise ValueError('Nonfinite input')
                    clf = base.fit_context(base.estimator(dict(n_estimators=p['members']), seed, tp, args.device), x, y)
                    if base.parameter_digest(clf) != p['parameter_sha256']:
                        raise ValueError('Pretrained weights changed')
                    flat = q.reshape(-1, x.shape[1])
                    score = base.query_scores(clf, flat).reshape(q.shape[:2])
                    dest.mkdir(parents=True, exist_ok=True)
                    clf.save(dest/'fitted.pkl', save_model_weights=False, save_training_data=True, save_kv_cache=False)
                    np.savez_compressed(dest/'scores.npz', views=score, y_val=a['y_val'], pairs_val=a['pairs_val'])
                    del clf
                    gc.collect()
                    torch.cuda.empty_cache()
                    clf = TabICLClassifier.load(dest/'fitted.pkl', device=args.device)
                    replay = base.query_scores(clf, flat).reshape(q.shape[:2])
                    np.testing.assert_allclose(replay, score, atol=1e-6, rtol=1e-6)
                    base.write_json(dest/'result.json', dict(task=task, variant=variant, seed=seed, kind=kind,
                        replay_passed=True, replay_max_abs=float(np.max(np.abs(replay-score))),
                        input_columns=x.shape[1], seconds=time.monotonic()-start,
                        artifacts={n: base.digest(dest/n) for n in ('fitted.pkl', 'scores.npz')}))
                    print('VERIFIED', task, variant, seed, kind, round(time.monotonic()-start, 1), flush=True)
                    del clf
                    gc.collect()
                    torch.cuda.empty_cache()


def report(args):
    import pandas as pd
    out = Path(args.out_dir)
    p = accepted.protocol(ACC)
    rows = []
    for task in args.tasks.split(','):
        a, gp = accepted.stage_data(out, p, task, 'inner')
        for seed in p['seeds']:
            ref_path = ACC/'predictions'/'inner'/task/('s'+str(seed))/'scores.npz'
            with np.load(ref_path, allow_pickle=False) as z:
                np.testing.assert_array_equal(z['y_val'], a['y_val'])
                np.testing.assert_array_equal(z['pairs_val'], a['pairs_val'])
                reference = z['tiered_0p5000'].copy()
                baseline = z['baseline'].copy()
            rows.append(dict(task=task, seed=seed, variant='accepted_reference', **base.metrics(a['y_val'], reference, a['pairs_val'])))
            for variant in args.variants.split(','):
                aug = augmented_score(out, p, a, task, variant, seed)
                candidate = .5*baseline + .5*aug
                rows.append(dict(task=task, seed=seed, variant=variant, **base.metrics(a['y_val'], candidate, a['pairs_val'])))
    table = pd.DataFrame(rows)
    table.to_csv(out/'metrics.csv', index=False)
    summary = table.groupby(['task', 'variant'], sort=False)[list(base.METRICS)].mean().reset_index()
    summary.to_csv(out/'summary.csv', index=False)
    gates = {}
    for variant in args.variants.split(','):
        gate = {}
        for task in args.tasks.split(','):
            t = table[table.task == task].set_index(['variant', 'seed'])
            delta = {m: np.array([t.loc[(variant, s), m]-t.loc[('accepted_reference', s), m] for s in p['seeds']]) for m in ('auc', 'ap')}
            if task == 'LncDrug':
                ok = all(np.all(v >= 0) for v in delta.values())
            else:
                ok = all(v.mean() > 0 and (v > 0).sum() >= 2 for v in delta.values())
            gate[task] = dict(passed=bool(ok), delta={k: v.tolist() for k, v in delta.items()})
        gates[variant] = gate
    result = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), phase='inner',
                  reference='accepted tiered_0p5000 (TabICLv2 0.5 graph + 0.5 tiered)',
                  candidate='0.5 graph + 0.5 TabICLv2(graph + tiered + gene-metapath channel)',
                  feature_names=list(FEATURE_NAMES), gates=gates,
                  no_test_access=True, historical_inner_reused=True,
                  note='Label-free gene-gene functional metapath features; association labels never used in feature construction.')
    base.write_json(out/'ASSESSMENT.json', result)
    print(summary.to_string(index=False))
    print('GATES', gates)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('run', 'report'))
    ap.add_argument('--out-dir', default=str(OUT))
    ap.add_argument('--task', default='MiDrug,LncDrug')
    ap.add_argument('--tasks', default='MiDrug,LncDrug')
    ap.add_argument('--variants', default=','.join(VARIANTS))
    ap.add_argument('--seed', type=int, default=20260915)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    if args.command == 'run':
        run(args)
    else:
        report(args)


if __name__ == '__main__':
    main()
