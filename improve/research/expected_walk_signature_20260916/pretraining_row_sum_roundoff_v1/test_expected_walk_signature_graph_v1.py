"""Independent path enumeration, derivatives, and train/query isolation checks."""
from datetime import datetime, timezone
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

from improve import expected_walk_signature_graph_v1 as core
from improve import anchor_expected_walk_signature_experiment_v1 as experiment
from improve import tabular_foundation as base
from improve.test_nested_anchor_v2 import fixture

HERE = Path(__file__).resolve().parent/'research/expected_walk_signature_20260916'
EVIDENCE = {}


def arrays():
    a, _ = fixture()
    a['anchor_oof'] = np.linspace(-.5, .5, len(a['y_train']))
    return a


def small_protocol():
    return dict(smoke=True, family='expected_walk_signature',
        implementation_module='expected_walk_signature_graph_v1',
        source_sha256={'expected_walk_signature_graph_v1.py': base.digest(core.__file__)},
        core_parameters=dict(core.parameters(), width=8, heads=2, depth=1, signature_rank=2, walk_steps=3),
        kinds=list(core.KINDS), epochs=5, check_interval=5, weight_decay=.001, strong_weight=.1)


def small_context(steps=3):
    edges = torch.tensor([[0,0,0,1,1,2,2,2,3,3], [0,1,2,1,2,0,2,3,1,3]])
    return core.bridge_context((edges, torch.zeros(len(edges[0]), dtype=torch.long),
                                torch.zeros(4, 1)), torch.tensor([2, 3]), steps)


def enumerate_paths(context, u, v):
    """Independent NumPy Chen products, explicitly enumerate every path."""
    p = context['transition'].to_dense().numpy()
    targets = context['targets'].numpy()
    steps = len(context['masses'])-1
    mass = np.zeros((len(p), len(targets)))
    result = np.zeros((len(p), len(targets), 3*u.shape[1]))
    for origin in range(len(p)):
        states = [(origin, 1., np.zeros(u.shape[1]), np.zeros(v.shape[1]), np.zeros(u.shape[1]))]
        for _ in range(steps):
            following = []
            for node, prob, first_u, first_v, second in states:
                for neighbor in np.flatnonzero(p[node]):
                    du, dv = u[neighbor]-u[node], v[neighbor]-v[node]
                    following.append((neighbor, prob*p[node,neighbor], first_u+du, first_v+dv,
                                      second+first_u*dv+.5*du*dv))
            states = following
        for node, prob, first_u, first_v, second in states:
            matches = np.flatnonzero(targets == node)
            if len(matches):
                j = matches[0]
                mass[origin,j] += prob
                result[origin,j] += prob*np.r_[first_u, first_v, second]
    valid = mass > 0
    result[valid] /= mass[valid,None]
    return mass, result


class ExpectedWalkChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.random.default_generator.manual_seed(427)

    def test_exact_enumerated_bridge_probabilities_and_chen_signature(self):
        context = small_context()
        u, v = [torch.randn(4,2,dtype=torch.float64) for _ in range(2)]
        mass, expected = enumerate_paths(context, u.numpy(), v.numpy())
        actual = core.bridge_signature(context, u, v, 'full')
        np.testing.assert_allclose(mass, context['masses'][-1], atol=2e-15, rtol=2e-15)
        np.testing.assert_allclose(expected, actual, atol=3e-14, rtol=3e-14)
        # One step has no internal vertex: its second level is symmetric.
        one = small_context(1)
        torch.testing.assert_close(core.bridge_signature(one,u,v,'full'),
                                   core.bridge_signature(one,u,v,'symmetrized'), atol=2e-14, rtol=2e-14)
        missing = one['masses'][-1] == 0
        self.assertTrue(missing.any())
        self.assertEqual(float(core.bridge_signature(one,u,v,'full')[missing].abs().max()), 0.)
        EVIDENCE['enumeration'] = dict(all_paths_enumerated=True,
            probability_error=float(np.max(abs(mass-context['masses'][-1].numpy()))),
            chen_signature_error=float(np.max(abs(expected-actual.numpy()))),
            one_step_symmetric=True, missing_bridges_zero=True)

    def test_derivatives_translation_and_shuffle_identity(self):
        context = small_context()
        u, v = [torch.randn(4,2,dtype=torch.float64,requires_grad=True) for _ in range(2)]
        self.assertTrue(torch.autograd.gradcheck(lambda x,y:core.bridge_signature(context,x,y,'full'),
                                               (u,v), atol=1e-6, rtol=1e-4))
        result = core.bridge_signature(context,u,v,'full')
        translated = core.bridge_signature(context,u+5.31,v-2.87,'full')
        torch.testing.assert_close(result, translated, atol=1e-13, rtol=1e-13)
        reverse_coordinates = core.bridge_signature(context,v,u,'full')
        du,dv,second = result.chunk(3,-1)
        torch.testing.assert_close(second+reverse_coordinates.chunk(3,-1)[2], du*dv,
                                   atol=2e-14, rtol=2e-14)
        EVIDENCE['calculus'] = dict(gradcheck=True, translation_invariance=True,
            conditional_shuffle_identity=True)

    def test_path_order_information_and_controls(self):
        # Two different deterministic paths with identical start and end.
        u = torch.tensor([[0.],[1.],[1.],[0.]],dtype=torch.float64)
        v = torch.tensor([[0.],[0.],[1.],[1.]],dtype=torch.float64)
        first = torch.tensor([[0,1,2,3],[1,2,2,3]])
        second = torch.tensor([[0,1,2,3],[3,1,2,2]])
        contexts = [core.bridge_context((e,torch.zeros(4,dtype=torch.long),torch.zeros(4,1)),
                                       torch.tensor([2]),2) for e in (first,second)]
        values = {kind:[core.bridge_signature(c,u,v,kind)[0,0] for c in contexts] for kind in core.KINDS}
        self.assertAlmostEqual(float(values['full'][0][-1]),1.)
        self.assertAlmostEqual(float(values['full'][1][-1]),0.)
        for kind in core.KINDS[1:]:
            torch.testing.assert_close(*values[kind],atol=0,rtol=0)
        # A closed reversed traversal changes signed area.
        forward = torch.tensor([[0,1,2,3],[1,2,0,3]])
        reverse = torch.tensor([[0,1,2,3],[2,0,1,3]])
        areas=[]
        for e in (forward,reverse):
            c=core.bridge_context((e,torch.zeros(4,dtype=torch.long),torch.zeros(4,1)),torch.tensor([0]),3)
            areas.append(core.bridge_signature(c,u,v,'full')[0,0,-1])
        torch.testing.assert_close(areas[0],-areas[1],atol=0,rtol=0)
        self.assertGreater(float(abs(areas[0])),.1)
        EVIDENCE['order'] = dict(equal_endpoints_full_gap=1., symmetrized_gap=0., first_level_gap=0.,
            reversed_closed_path_area_changes_sign=True)

    def test_support_boundaries_node_permutation_and_query_batching(self):
        a,p=arrays(),small_protocol()
        model,xs,graphs,folds=core.setup(a,p['core_parameters'],'full','cpu')
        with torch.no_grad(): model.readout[-1].weight.normal_(0.,.1)
        model.eval()
        pairs=torch.as_tensor(a['pairs_train']+a['pair_offset'][:,None])
        graph=graphs[0]
        y=model(xs,graph,pairs).detach()
        reversed_y=model(xs,graph,pairs.flip(1)).detach().flip(0)
        chunks=torch.cat([model(xs,graph,pairs[:,i:i+11]).detach() for i in range(0,pairs.shape[1],11)])
        torch.testing.assert_close(y,reversed_y,atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(y,chunks,atol=1e-6,rtol=1e-6)
        n=int(a['node_counts'].sum())
        for f,g in enumerate(graphs):
            q=pairs[:,folds[f]].numpy()
            forbidden=np.r_[q[0]*n+q[1],q[1]*n+q[0]]
            edges=g['physical'][0].numpy()
            self.assertFalse(np.intersect1d(forbidden,edges[0]*n+edges[1]).size)
        altered={k:v.copy() for k,v in a.items()}
        altered['y_train']=1-altered['y_train']
        _,_,others,_=core.setup(altered,p['core_parameters'],'full','cpu')
        for g,h in zip(graphs,others):
            for x,z in zip(g['masses'],h['masses']): torch.testing.assert_close(x,z,atol=0,rtol=0)
        old_to_new=np.arange(n)
        new_to_old,perm_xs=[],[]
        starts=np.r_[0,np.cumsum(a['node_counts'])]
        for i,x in enumerate(xs):
            order=torch.randperm(len(x))
            perm_xs.append(x[order])
            old_to_new[starts[i]+order.numpy()]=starts[i]+np.arange(len(x))
            new_to_old.extend((starts[i]+order.numpy()).tolist())
        mapping=torch.as_tensor(old_to_new)
        edges,rel,counts=graph['physical']
        changed=core.bridge_context((mapping[edges],rel,counts[new_to_old]), graph['targets'],
                                    p['core_parameters']['walk_steps'])
        changed_y=model(perm_xs,changed,mapping[pairs]).detach()
        torch.testing.assert_close(y,changed_y,atol=2e-6,rtol=2e-6)
        with self.assertRaisesRegex(ValueError,'training arrays only'):
            core.setup(dict(a,y_val=np.zeros(1)),p['core_parameters'],'full','cpu')
        EVIDENCE['boundary']=dict(forbidden_query_arcs=0, query_labels_invariant=True,
            within_type_node_permutation_error=float((y-changed_y).abs().max()),
            query_batching_error=float((y-chunks).abs().max()),nonzero_decoder=True)

    def test_matched_training_and_lnc_identity(self):
        a,p=arrays(),small_protocol()
        state,rng=None,None
        final_rng=[]
        for kind in core.KINDS:
            torch.random.default_generator.manual_seed(716)
            initial,_,_,_=core.setup(a,p['core_parameters'],kind,'cpu')
            if state is None:
                state,rng=experiment.cpu_copy(initial.state_dict()),torch.random.get_rng_state()
            else:
                for name,value in initial.state_dict().items(): torch.testing.assert_close(value,state[name],atol=0,rtol=0)
                torch.testing.assert_close(torch.random.get_rng_state(),rng,atol=0,rtol=0)
            model,value,trace,detail=experiment.fit(a,p,kind,.003,716,'cpu',fixed_epochs=5)
            self.assertIsNone(value)
            self.assertFalse(any('anchor_composed' in row for row in trace))
            self.assertEqual(detail['optimizer_steps'],5)
            for name,value in model.named_parameters():
                self.assertIsNotNone(value.grad,name)
                self.assertTrue(torch.isfinite(value.grad).all(),name)
            final=model.training_final_state
            final_rng.append(final['cpu_rng'])
            for value in final['optimizer']['state'].values(): self.assertEqual(int(value['step']),5)
        for r in final_rng[1:]: torch.testing.assert_close(r,final_rng[0],atol=0,rtol=0)
        lnc=arrays()
        lnc['pair_offset']=np.array([0,14])
        for kind in core.KINDS:
            model,value,trace,detail=experiment.fit(lnc,p,kind,.003,716,'cpu',fixed_epochs=5)
            self.assertFalse(detail['trained_neural_model'])
            self.assertEqual(detail['training_steps'],0)
            self.assertFalse(list(model.parameters()))
            self.assertFalse(trace)
            result=model([],None,torch.tensor([[0,1],[14,15]]))
            torch.testing.assert_close(result,torch.zeros_like(result),atol=0,rtol=0)
        EVIDENCE['execution']=dict(paired_initialization_and_rng=True, finite_parameter_gradients=True,
            lnc_exact_zero_without_training=True, fixed_training_has_no_evaluation=True)

    def test_partial_numerical_failure_preserves_completed_steps(self):
        a,p=arrays(),small_protocol()
        original,count=core.ExpectedWalkSignatureGraph.forward,[0]
        def broken(model,*args):
            count[0]+=1
            if count[0]==3: raise FloatingPointError('injected bridge failure')
            return original(model,*args)
        with patch.object(core.ExpectedWalkSignatureGraph,'forward',broken):
            with self.assertRaises(FloatingPointError) as caught:
                experiment.fit(a,p,'full',.003,716,'cpu',fixed_epochs=5)
        error=caught.exception
        self.assertEqual(error.training_partial_state['steps'],2)
        self.assertEqual(error.training_failure_context['current_step'],3)
        for value in error.training_partial_state['optimizer']['state'].values():
            self.assertEqual(int(value['step']),2)
        with self.assertRaisesRegex(ValueError,'training arrays only'):
            experiment.fit(dict(a,y_val=np.zeros(2)),p,'full',.003,716,'cpu',fixed_epochs=5)
        EVIDENCE['failure']=dict(injected_step=3,completed_optimizer_steps=2,exact_snapshot=True)


if __name__ == '__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ExpectedWalkChecks))
    path=HERE/'EXPECTED_WALK_SIGNATURE_UNIT_V1.json'
    if path.exists(): raise ValueError('Preserve earlier unit evidence')
    base.write_json(path,base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        success=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),
        evidence=EVIDENCE,no_performance_metrics=True,
        source_sha256={str(Path(p).resolve()):base.digest(p) for p in
            [__file__,core.__file__,experiment.__file__,Path(__file__).with_name('test_nested_anchor_v2.py')]})))
    raise SystemExit(0 if result.wasSuccessful() else 1)
