"""Independent quadrature, pinned-author, derivatives, and graph-boundary checks."""
import ast
from datetime import datetime, timezone
from pathlib import Path
import unittest

import numpy as np
from scipy.integrate import quad
import torch
from torch.nn import functional as F

from improve import fourier_quantile_graph_v1 as core
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent / 'research/higher_order_next_20260916'
EVIDENCE = {}


def arrays():
    rng = np.random.default_rng(851)
    counts = np.array([2, 5, 3, 4])
    pairs = np.array([[0, 1, 2, 3, 4, 0, 1, 2], [0, 1, 2, 0, 1, 2, 0, 1]])
    labels = np.array([1., 0., 1., 0., 1., 0., 1., 0.])
    physical = pairs[:, labels == 1] + np.array([2, 7])[:, None]
    other = np.array([[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
                      [2, 5, 10, 11, 12, 13, 0, 1, 12, 11, 12, 13]])
    a = dict(node_counts=counts, pair_offset=np.array([2, 7]), nd=np.array(3),
        pairs_train=pairs, y_train=labels, folds=np.arange(len(labels)) % 2,
        anchor_oof=np.linspace(-.5, .5, len(labels)), edges=np.concatenate((physical, other), axis=1),
        relation=np.r_[np.zeros(physical.shape[1], dtype=np.int64), np.ones(other.shape[1], dtype=np.int64)],
        target_codes=np.r_[pairs[0, labels == 1] * 3 + pairs[1, labels == 1],
                           np.full(other.shape[1], -1, dtype=np.int64)])
    for index, count in enumerate(counts):
        a['x_' + str(index)] = rng.normal(size=(count, 4 + index)).astype(np.float32)
    return a


def author_dense():
    records = base.read_json(HERE / 'FSW_GNN_AUTHOR_SOURCE_V1.json')
    row = next(r for r in records if r['path'] == 'bottleneck/models/fsw/fsw_embedding.py')
    path = Path(__file__).resolve().parent.parent / row['file']
    assert base.digest(path) == row['sha256'] and row['git_blob_verified']
    tree = ast.parse(path.read_text())
    cls = next(v for v in tree.body if isinstance(v, ast.ClassDef) and v.name == 'FSW_embedding')
    method = next(v for v in cls.body if isinstance(v, ast.FunctionDef) and v.name == 'forward_helper')
    helpers = [v for v in tree.body if isinstance(v, ast.FunctionDef)
               and v.name in ('diff_zeropad', 'replace_in_tuple')]
    selected = ast.Module(body=helpers + [ast.ClassDef(name='FSW_embedding', bases=[], keywords=[],
        body=[method], decorator_list=[], type_params=[])], type_ignores=[])
    namespace = dict(torch=torch, np=np)
    exec(compile(ast.fix_missing_locations(selected), str(path), 'exec'), namespace)
    return namespace['FSW_embedding'].forward_helper, row


def small_parameters():
    return dict(core.parameters(), width=8, embedding_width=12, edge_width=3, folds=2)


class QuantileChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.random.default_generator.manual_seed(743)

    def test_author_forward_and_gradients(self):
        author, source = author_dense()
        x = torch.randn(3, 7, 3, dtype=torch.float64, requires_grad=True)
        directions = torch.randn(6, 3, dtype=torch.float64, requires_grad=True)
        frequency = torch.tensor([0., 1e-10, .17, .9, 3.7, 30.], dtype=torch.float64, requires_grad=True)
        valid = torch.tensor([[True]*7, [True, False, True, True, False, False, True], [False]*7])
        weights = valid.double() / valid.sum(1, keepdim=True).clamp_min(1)
        expected = author(x, weights, directions, frequency, False, None, False, (3,))
        actual = core.uniform_quantile_features(F.linear(x, directions), valid, frequency, 'full')
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        grad_a = torch.autograd.grad(actual.square().sum(), (x, directions, frequency), retain_graph=True)
        grad_b = torch.autograd.grad(expected.square().sum(), (x, directions, frequency))
        for left, right in zip(grad_a, grad_b):
            self.assertTrue(torch.isfinite(left).all())
            torch.testing.assert_close(left, right, atol=1e-10, rtol=1e-10)
        EVIDENCE['author'] = dict(commit=source['commit'], source_sha256=source['sha256'],
            selected_definitions=['FSW_embedding.forward_helper', 'diff_zeropad', 'replace_in_tuple'],
            selected_ast_bodies_modified=False, execution_scope='dense uniform/zero mass helper only',
            full_author_GNN_executed=False, maximum_forward_error=float((actual-expected).detach().abs().max()),
            maximum_gradient_error=max(float((a-b).abs().max()) for a,b in zip(grad_a, grad_b)),
            zero_frequency_sinc_limit_one=True)

    def test_quadrature_and_finite_difference(self):
        x = torch.randn(2, 5, 4, dtype=torch.float64, requires_grad=True)
        valid = torch.tensor([[True]*5, [True, False, True, False, True]])
        frequency = torch.tensor([.01, .7, 2.3, 8.1], dtype=torch.float64, requires_grad=True)
        actual = core.uniform_quantile_features(x, valid, frequency, 'full')
        reference = np.zeros(actual.shape)
        for group in range(2):
            for coord, freq in enumerate(frequency.detach().numpy()):
                values = np.sort(x.detach().numpy()[group, valid[group].numpy(), coord])
                reference[group, coord] = sum(quad(lambda t: 2*(1+freq)*value*np.cos(2*np.pi*freq*t),
                    i/len(values), (i+1)/len(values), epsabs=1e-12, epsrel=1e-12)[0]
                    for i, value in enumerate(values))
        np.testing.assert_allclose(actual.detach().numpy(), reference, atol=1e-12, rtol=1e-12)
        self.assertTrue(torch.autograd.gradcheck(
            lambda a,b: core.uniform_quantile_features(a,valid,b,'full'), (x, frequency),
            eps=1e-6, atol=2e-5, rtol=2e-4))
        EVIDENCE['independent_integral'] = dict(quadrature_max_error=float(abs(actual.detach().numpy()-reference).max()),
            active_feature_and_frequency_gradcheck=True, padding_has_zero_mass=True)

    def test_controls_moments_ties_empty_and_replication(self):
        frequencies = torch.tensor([0., .35, 1.2, 3.4], dtype=torch.float64)
        directions = torch.tensor([.7, 1., -1.3, 2.1], dtype=torch.float64)
        a = torch.tensor([-1.,-1.,1.,1.], dtype=torch.float64)[:,None]*directions[None]
        b = torch.tensor([-2**.5,0.,0.,2**.5], dtype=torch.float64)[:,None]*directions[None]
        values = torch.stack((a,b))
        valid = torch.ones(2,4,dtype=torch.bool)
        outputs = {kind:core.uniform_quantile_features(values,valid,frequencies,kind) for kind in core.KINDS}
        for kind in ('mean_only','mean_std'):
            torch.testing.assert_close(outputs[kind][0], outputs[kind][1],atol=1e-12,rtol=1e-12)
        self.assertGreater(float((outputs['full'][0]-outputs['full'][1]).abs().max()), .01)
        repeat = values.repeat_interleave(3, dim=1)
        replay = core.uniform_quantile_features(repeat,torch.ones(2,12,dtype=torch.bool),frequencies,'full')
        torch.testing.assert_close(replay, outputs['full'],atol=1e-12,rtol=1e-12)
        permutation = torch.tensor([3,0,2,1])
        for kind in core.KINDS:
            permuted=core.uniform_quantile_features(values[:,permutation],valid,frequencies,kind)
            torch.testing.assert_close(permuted,outputs[kind],atol=1e-12,rtol=1e-12)
            empty=core.uniform_quantile_features(values,torch.zeros_like(valid),frequencies,kind)
            torch.testing.assert_close(empty,torch.zeros_like(empty),atol=0.,rtol=0.)
        EVIDENCE['controls'] = dict(equal_mean_and_variance_distributions_separated=True,
            permutation_and_ties=True, empty_exact_zero=True, uniform_replication_invariant=True,
            cardinality_encoded_separately_in_graph=True)

    def test_ragged_graph_permutation_and_training_boundary(self):
        a,p=arrays(),small_parameters()
        with self.assertRaisesRegex(ValueError,'training arrays only'):
            core.setup(dict(a,y_val=np.array([0.])),p,'full','cpu')
        model,xs,graphs,folds=core.setup(a,p,'full','cpu')
        graph=graphs[0]
        n=len(graph.degree)
        omitted=a['pairs_train'][:,folds[0].numpy()]+a['pair_offset'][:,None]
        edge=graph.physical[0].numpy()
        code=edge[0]*n+edge[1]
        self.assertFalse(np.intersect1d(code,np.r_[omitted[0]*n+omitted[1],omitted[1]*n+omitted[0]]).size)
        actual=[]
        for bucket in graph.buckets:
            actual.extend(bucket.messages[bucket.valid].tolist())
            for row,node in enumerate(bucket.recipients.tolist()):
                self.assertEqual(int(bucket.valid[row].sum()),int(graph.degree[node]))
                ids=bucket.messages[row,bucket.valid[row]]
                self.assertTrue((edge[:,edge[0]!=edge[1]][1,ids.numpy()]==node).all())
        self.assertEqual(sorted(actual),list(range(len(graph.senders))))
        state=torch.randn(n,p['width'])
        permutation=torch.randperm(n)
        mapped=permutation[graph.physical[0]]
        counts=torch.empty_like(graph.physical[2]); counts[permutation]=graph.physical[2]
        changed=core.pack_graph((mapped,graph.physical[1],counts))
        new_state=torch.empty_like(state);new_state[permutation]=state
        model.eval()
        errors=[]
        with torch.no_grad():
            for kind in core.KINDS:
                left,_=model.layers[0](state,graph,kind)
                right,_=model.layers[0](new_state,changed,kind)
                errors.append(float((left-right[permutation]).abs().max()))
                torch.testing.assert_close(left,right[permutation],atol=1e-6,rtol=1e-6)
        EVIDENCE['graph'] = dict(all_messages_retained_once=True, forbidden_target_overlap=0,
            training_only_entry=True, node_permutation_max_error=max(errors))

    def test_matching_training_and_query_replay(self):
        a,p=arrays(),small_parameters()
        initial,random_state,records=None,None,[]
        for kind in core.KINDS:
            torch.random.default_generator.manual_seed(997)
            model,xs,graphs,folds=core.setup(a,p,kind,'cpu')
            if initial is None:
                initial={k:v.detach().clone() for k,v in model.state_dict().items()}
            else:
                for k,v in model.state_dict().items():torch.testing.assert_close(v,initial[k],atol=0.,rtol=0.)
            pairs=torch.as_tensor(a['pairs_train']+a['pair_offset'][:,None])
            model.eval()
            with torch.no_grad():
                zero=model(xs,graphs[0],pairs[:,folds[0]])
                torch.testing.assert_close(zero,torch.zeros_like(zero),atol=0.,rtol=0.)
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
            for step in range(3):
                f=step%2;idx=folds[f]
                model.train();optimizer.zero_grad(set_to_none=True)
                residual=model(xs,graphs[f],pairs[:,idx])
                loss=F.binary_cross_entropy_with_logits(residual.double()+torch.as_tensor(a['anchor_oof'])[idx],
                    torch.as_tensor(a['y_train'])[idx])
                loss.backward()
                self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters()))
                optimizer.step()
            self.assertTrue(all(int(state['step'])==3 for state in optimizer.state.values()))
            if random_state is None:random_state=torch.random.get_rng_state()
            else:torch.testing.assert_close(torch.random.get_rng_state(),random_state,atol=0.,rtol=0.)
            model.eval()
            with torch.no_grad():
                query=pairs[:,folds[0]]
                normal=model(xs,graphs[0],query)
                reverse=model(xs,graphs[0],query.flip(1)).flip(0)
                chunks=torch.cat([model(xs,graphs[0],query[:,i:i+1]) for i in range(query.shape[1])])
            torch.testing.assert_close(normal,reverse,atol=3e-5,rtol=3e-5)
            torch.testing.assert_close(normal,chunks,atol=3e-5,rtol=3e-5)
            self.assertGreater(float(normal.abs().max()),1e-8)
            records.append(dict(kind=kind,replay_max_abs=float(max((normal-reverse).abs().max(),(normal-chunks).abs().max())),
                                diagnostics=model.quantile_diagnostics))
        for record in records[1:]:
            for key in ('calls','layer_calls','sort_calls','messages','padded_slots'):
                self.assertEqual(record['diagnostics'][key],records[0]['diagnostics'][key])
        EVIDENCE['training'] = dict(initial_states_equal=True,zero_initial_residual=True,
            final_cpu_random_state_equal=True,synthetic_steps_per_arm=3,every_parameter_gradient_finite=True,
            replay_uses_omitted_fold_only=True,records=records,formal_fits=0)


if __name__ == '__main__':
    destination=HERE/'FOURIER_QUANTILE_UNIT_V1.json'
    if destination.exists():raise ValueError('Preserve previous unit evidence')
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(QuantileChecks))
    report=base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),success=result.wasSuccessful(),
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),evidence=EVIDENCE,
        source_sha256={str(Path(path).resolve()):base.digest(path) for path in (__file__,core.__file__)},
        performance_metrics_computed=False,evaluation_labels_read=False,formal_fits=0))
    base.write_json(destination,report)
    raise SystemExit(0 if result.wasSuccessful() else 1)
