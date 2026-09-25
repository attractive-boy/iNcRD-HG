"""Author projection, independent tensor calculus, support and executor checks."""
import ast
from datetime import datetime, timezone
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from improve import tensor_product_attention_graph_v1 as core
from improve import anchor_tensor_product_experiment_v1 as experiment
from improve import tabular_foundation as base
from improve.test_nested_anchor_v2 import fixture

HERE = Path(__file__).resolve().parent / 'research/tensor_attention_next_20260916'
EVIDENCE = {}
COUNT_KEYS = ('calls', 'attention_layers', 'node_rows', 'contextual_product_terms',
    'static_product_terms', 'dense_projection_outputs', 'scored_edge_heads', 'value_message_entries')


def arrays():
    a, _ = fixture()
    a['anchor_oof'] = np.linspace(-.5, .5, len(a['y_train']))
    return a


def small_protocol():
    cp = dict(core.parameters(), width=8, heads=2, q_rank=2, kv_rank=1)
    return dict(smoke=True, family='tensor_product_attention',
        implementation_module='tensor_product_attention_graph_v1',
        source_sha256={'tensor_product_attention_graph_v1.py': base.digest(core.__file__)},
        core_parameters=cp, kinds=list(core.KINDS), epochs=5, check_interval=5,
        weight_decay=.001, strong_weight=.1)


class ZeroPositionRotary(nn.Module):
    """All tokens at zero position: the declared no-positional-encoding case."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        shape = (1, x.shape[1], 1, self.dim//2)
        return x.new_ones(shape), x.new_zeros(shape)


def author_projection():
    row = next(r for r in base.read_json(HERE / 'TPA_AUTHOR_SOURCE_V1.json') if r['path'] == 'model/T6.py')
    path = Path(row['file'])
    assert base.digest(path) == row['sha256'] and row['git_blob_verified']
    assert row['commit'] == 'c276c80d5ad807881dedb4707d8d3c20b4e97ec6'
    module = ast.parse(path.read_text())
    selected = [node for node in module.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name in ('CPLinear', 'apply_rotary_emb')]
    assert len(selected) == 2
    namespace = dict(torch=torch, nn=nn, Rotary=ZeroPositionRotary)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['CPLinear'], row


class TensorProductChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.random.default_generator.manual_seed(852)

    def test_author_initialization_projection_and_gradients(self):
        author, row = author_projection()
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            torch.random.default_generator.manual_seed(37)
            reference = author(5, 3, 4, rank=2, q_rank=4)
            rng = torch.random.get_rng_state()
            torch.random.default_generator.manual_seed(37)
            local = core.TensorProductQKV(5, 3, 4, q_rank=4, kv_rank=2)
            torch.testing.assert_close(torch.random.get_rng_state(), rng, atol=0., rtol=0.)
            for name, value in reference.state_dict().items():
                torch.testing.assert_close(local.state_dict()[name], value, atol=0., rtol=0.)
            x = torch.randn(7, 5, requires_grad=True)
            y, z = local(x), tuple(v.squeeze(0) for v in reference(x.unsqueeze(0)))
            for left, right in zip(y, z):
                torch.testing.assert_close(left, right, atol=1e-14, rtol=1e-14)
            names = [name for name, value in reference.named_parameters()]
            gl = torch.autograd.grad(sum(v.square().sum() for v in y),
                [x]+[dict(local.named_parameters())[n] for n in names], retain_graph=True)
            gr = torch.autograd.grad(sum(v.square().sum() for v in z),
                [x]+[dict(reference.named_parameters())[n] for n in names])
            for left, right in zip(gl, gr):
                torch.testing.assert_close(left, right, atol=2e-13, rtol=2e-13)
        finally:
            torch.set_default_dtype(previous)
        EVIDENCE['author'] = dict(commit=row['commit'], source_sha256=row['sha256'],
            selected_bodies=['CPLinear', 'apply_rotary_emb'], selected_bodies_modified=False,
            rotary_provider='all-zero positions; identity rotation', full_T6_executed=False,
            initialization_and_random_stream_exact=True,
            maximum_forward_error=max(float((a-b).detach().abs().max()) for a,b in zip(y,z)),
            maximum_gradient_error=max(float((a-b).abs().max()) for a,b in zip(gl,gr)))

    def test_independent_quadratic_tensor_and_all_active_gradcheck(self):
        m = core.TensorProductQKV(3, 2, 2, q_rank=2, kv_rank=1).double()
        x = torch.randn(3, 3, dtype=torch.float64, requires_grad=True)
        outputs, errors = m(x), []
        for output, (letter, rank) in zip(outputs, m.ranks.items()):
            a = getattr(m, 'W_A_'+letter).weight.detach().numpy().reshape(2, rank, 3)
            b = getattr(m, 'W_B_'+letter).weight.detach().numpy().reshape(rank, 2, 3)
            tensor = np.einsum('hri,rdj->hdij', a, b)/rank
            expected = np.einsum('bi,bj,hdij->bhd', x.detach().numpy(), x.detach().numpy(), tensor)
            np.testing.assert_allclose(output.detach().numpy(), expected, atol=1e-13, rtol=1e-13)
            errors.append(float(abs(output.detach().numpy()-expected).max()))
            index = list(m.ranks).index(letter)
            hessian = torch.autograd.functional.hessian(lambda v: m(v[None])[index][0,0,0], x[0])
            np.testing.assert_allclose(hessian.numpy(), tensor[0,0]+tensor[0,0].T, atol=1e-13, rtol=1e-13)
        names = [n for n,v in m.named_parameters() if v.requires_grad]
        values = tuple(dict(m.named_parameters())[n] for n in names)
        def call(inputs, *parameters):
            return torch.cat([v.flatten() for v in torch.func.functional_call(m, dict(zip(names,parameters)), (inputs,))])
        self.assertTrue(torch.autograd.gradcheck(call, (x,)+values, atol=1e-6, rtol=1e-4))
        for y, minus, scaled in zip(outputs, m(-x), m(1.7*x)):
            torch.testing.assert_close(y, minus, atol=1e-13, rtol=1e-13)
            torch.testing.assert_close(scaled, 1.7**2*y, atol=1e-13, rtol=1e-13)
        EVIDENCE['polynomial'] = dict(independent_tensor_max_error=max(errors),
            input_and_all_active_parameter_gradcheck=True, analytic_quadratic_hessian=True,
            evenness_and_degree_two_homogeneity=True)

    def test_controls_ranks_capacity_and_initial_functions(self):
        previous = torch.get_default_dtype()
        models, rngs = {}, []
        try:
            torch.set_default_dtype(torch.float64)
            for kind in core.KINDS:
                torch.random.default_generator.manual_seed(64)
                models[kind] = core.TensorProductQKV(5,4,4,q_rank=3,kv_rank=2,kind=kind)
                rngs.append(torch.random.get_rng_state())
            initial = models['full'].state_dict()
            for model in models.values():
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, initial[name], atol=0., rtol=0.)
            for rng in rngs[1:]:
                torch.testing.assert_close(rng, rngs[0], atol=0., rtol=0.)
            x = torch.randn(6,5,requires_grad=True)
            static, dense = models['noncontextual_a'](x), models['dense_qkv'](x)
            for left, right in zip(static,dense):
                torch.testing.assert_close(left,right,atol=1e-13,rtol=1e-13)
                torch.testing.assert_close(torch.autograd.grad(left.sum(),x,retain_graph=True)[0],
                    torch.autograd.grad(right.sum(),x,retain_graph=True)[0],atol=1e-13,rtol=1e-13)
            for kind in core.KINDS[1:]:
                hessian = torch.autograd.functional.hessian(lambda v: models[kind](v[None])[0].sum(),x[0])
                torch.testing.assert_close(hessian, torch.zeros_like(hessian),atol=0.,rtol=0.)
            for value, rank in zip(models['full'](x), (3,2,2)):
                self.assertTrue(torch.all(torch.linalg.svdvals(value.detach())[:,rank:] < 1e-12))
            total = 5*7*(4+4)+7*4+3*4*4*5
            expected = {'full':5*7*(4+4),'noncontextual_a':7*4+5*7*4,'dense_qkv':3*4*4*5}
            for kind, model in models.items():
                self.assertEqual(sum(p.numel() for p in model.parameters()), total)
                self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),expected[kind])
        finally:
            torch.set_default_dtype(previous)
        EVIDENCE['controls'] = dict(initial_stored_parameters_and_rng_equal=True,
            static_dense_initial_function_and_input_gradient_equal=True,
            full_static_initial_function_equality_claimed=False, static_and_dense_hessian_zero=True,
            contextual_rank_bounds_verified=True, active_parameter_counts=expected,
            equal_effective_capacity_claimed=False)

    def test_sparse_attention_against_independent_incidence_sum(self):
        layer = core.TensorProductAttention(4,2,3,0.,2,1,'full').double()
        edges = torch.tensor([[0,1,2,0,2,1,2],[0,0,1,2,2,1,0]])
        relations = torch.tensor([0,1,2,0,1,0,2])
        q,k,v = [torch.randn(3,2,2,dtype=torch.float64,requires_grad=True) for _ in range(3)]
        result, score = layer.aggregate(q,k,v,edges,relations)
        qn,kn,vn = [w.detach().numpy() for w in (q,k,v)]
        en = layer.edge.weight.detach().numpy().reshape(3,2,2)
        expected = np.zeros((3,2,2)); expected_score = np.zeros((7,2))
        for j,(src,dst) in enumerate(edges.numpy().T):
            for h in range(2):
                expected_score[j,h] = np.dot(qn[dst,h],kn[src,h]+en[int(relations[j]),h])/np.sqrt(2.)
        for node in range(3):
            ids = np.flatnonzero(edges[1].numpy()==node)
            for h in range(2):
                weights=np.exp(expected_score[ids,h]-expected_score[ids,h].max())
                weights/=weights.sum()
                for j,weight in zip(ids,weights): expected[node,h]+=weight*vn[int(edges[0,j]),h]
        np.testing.assert_allclose(result.detach().numpy(),expected,atol=1e-13,rtol=1e-13)
        np.testing.assert_allclose(score.detach().numpy(),expected_score,atol=1e-13,rtol=1e-13)
        self.assertTrue(torch.autograd.gradcheck(lambda a,b,c:layer.aggregate(a,b,c,edges,relations)[0],
            (q,k,v),atol=1e-6,rtol=1e-4))
        EVIDENCE['attention'] = dict(independent_numpy_max_error=float(abs(result.detach().numpy()-expected).max()),
            incidence_softmax_scope_verified=True, qkv_attention_gradcheck=True)

    def test_graph_query_boundary_and_node_permutation(self):
        a,p = arrays(),small_protocol()
        with self.assertRaisesRegex(ValueError,'training arrays only'):
            core.setup(dict(a,y_val=np.array([0.])),p['core_parameters'],'full','cpu')
        model,xs,graphs,folds = core.setup(a,p['core_parameters'],'full','cpu')
        graph,n = graphs[0],int(a['node_counts'].sum())
        query=torch.as_tensor(a['pairs_train'][:,folds[0]]+a['pair_offset'][:,None])
        live=graph[0][0].numpy()*n+graph[0][1].numpy()
        forbidden=np.r_[query[0].numpy()*n+query[1].numpy(),query[1].numpy()*n+query[0].numpy()]
        self.assertEqual(np.intersect1d(live,forbidden).size,0)
        changed={k:v.copy() for k,v in a.items()}
        changed['y_train'][folds[0].numpy()]=1.-changed['y_train'][folds[0].numpy()]
        _,_,alternate,_=core.setup(changed,p['core_parameters'],'full','cpu')
        for left,right in zip(graph,alternate[0]): torch.testing.assert_close(left,right,atol=0.,rtol=0.)
        permutation,changed_xs,start=[],[],0
        for x in xs:
            order=torch.randperm(len(x)); new_x=torch.empty_like(x); new_x[order]=x
            changed_xs.append(new_x); permutation.append(order+start); start+=len(x)
        permutation=torch.cat(permutation)
        counts=torch.empty_like(graph[2]); counts[permutation]=graph[2]
        mapped=(permutation[graph[0]],graph[1],counts)
        model.eval()
        with torch.no_grad():
            model.readout[-1].weight.normal_()
            reference=model(xs,graph,query)
            reordered=model(changed_xs,mapped,permutation[query])
            reverse=model(xs,graph,query.flip(1)).flip(0)
        torch.testing.assert_close(reordered,reference,atol=3e-6,rtol=3e-6)
        torch.testing.assert_close(reverse,reference,atol=3e-6,rtol=3e-6)
        EVIDENCE['graph']=dict(query_edges_and_reverse_absent=True,omitted_labels_cannot_change_support=True,
            node_permutation_max_error=float((reference-reordered).abs().max()),
            query_order_max_error=float((reference-reverse).abs().max()))

    def test_executor_gradients_frozen_state_identity_and_rng(self):
        a,p,rngs,records=arrays(),small_protocol(),[],[]
        with self.assertRaisesRegex(ValueError,'training arrays only'):
            experiment.fit(dict(a,y_val=np.array([0.])),p,'full',.001,994,'cpu',fixed_epochs=5)
        with self.assertRaisesRegex(ValueError,'cuda:1'):
            experiment.fit(a,p,'full',.001,994,'cuda:0',fixed_epochs=5)
        for kind in core.KINDS:
            torch.random.default_generator.manual_seed(994)
            initial,_,_,_=experiment.setup(a,p,kind,'cpu')
            model,value,trace,detail=experiment.fit(a,p,kind,.001,994,'cpu',fixed_epochs=5)
            self.assertIsNone(value)
            self.assertEqual(detail['optimizer_steps'],5)
            self.assertEqual(detail['tpa_diagnostics']['calls'],5)
            self.assertTrue(all('anchor_composed' not in row for row in trace))
            final=model.training_final_state; rngs.append(final['cpu_rng'])
            for name,value in model.named_parameters():
                if value.requires_grad:
                    self.assertIsNotNone(value.grad)
                    self.assertTrue(torch.isfinite(value.grad).all())
                else:
                    self.assertIsNone(value.grad)
                    torch.testing.assert_close(value,initial.state_dict()[name],atol=0.,rtol=0.)
            for state in final['optimizer']['state'].values(): self.assertEqual(int(state['step']),5)
            records.append({key:detail['tpa_diagnostics'][key] for key in COUNT_KEYS})
        for rng in rngs[1:]: torch.testing.assert_close(rng,rngs[0],atol=0.,rtol=0.)
        for record in records[1:]: self.assertEqual(record,records[0])
        ln=dict(a,pair_offset=np.array([0,14]))
        identity,value,trace,detail=experiment.fit(ln,p,'full',.001,994,'cpu',fixed_epochs=5)
        self.assertEqual(identity.state_dict(),{})
        self.assertEqual(trace,[]); self.assertEqual(detail['optimizer_steps'],0)
        torch.testing.assert_close(identity([],None,torch.tensor([[0,1],[14,15]])),torch.zeros(2),atol=0.,rtol=0.)
        EVIDENCE['executor']=dict(training_only_inputs=True,formal_cuda0_rejected=True,
            all_active_gradients_finite=True,frozen_parameters_unchanged=True,
            random_streams_equal=True,forward_operation_counts_equal=True,counts=records[0],
            lnc_exact_identity_without_training=True,evaluation_labels_read=False,formal_neural_fits=0)

    def test_failure_records_exact_partial_state(self):
        original=core.TensorProductGraph.forward
        def fail_third(model,*args):
            if model.tpa_diagnostics['calls']==2: raise FloatingPointError('Synthetic registered failure')
            return original(model,*args)
        with patch.object(core.TensorProductGraph,'forward',fail_third):
            with self.assertRaisesRegex(FloatingPointError,'Synthetic registered failure') as caught:
                experiment.fit(arrays(),small_protocol(),'full',.001,112,'cpu',fixed_epochs=5)
        state,context=caught.exception.training_partial_state,caught.exception.training_failure_context
        self.assertEqual(state['steps'],2); self.assertEqual(context['completed_steps'],2)
        self.assertEqual(context['current_step'],3); self.assertEqual(context['operation'],'forward')
        self.assertTrue(all(int(v['step'])==2 for v in state['optimizer']['state'].values()))
        EVIDENCE['failure']=dict(exact_completed_steps=2,current_step=3,operation='forward',
            partial_model_optimizer_rng_preserved=True,formal_neural_fit=False)


if __name__=='__main__':
    destination=HERE/'TENSOR_PRODUCT_UNIT_V1.json'
    if destination.exists(): raise ValueError('Preserve existing unit evidence')
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(TensorProductChecks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    sources=[Path(__file__),Path(core.__file__),Path(experiment.__file__),
        experiment.ROOT/'test_nested_anchor_v2.py',experiment.ROOT/'overlap_block_models.py',
        experiment.ROOT/'sharpness_graph_v1.py',HERE/'TPA_AUTHOR_SOURCE_V1.json',
        HERE/'tpa_author_v1_model__T6.py',HERE/'tpa_author_v1_LICENSE']
    record=base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        tests=result.testsRun,success=result.wasSuccessful(),failures=len(result.failures),errors=len(result.errors),
        evidence=EVIDENCE,source_sha256={str(path):base.digest(path) for path in sources},
        formal_neural_fits=0,evaluation_labels_read=False))
    base.write_json(destination,record)
    raise SystemExit(0 if result.wasSuccessful() else 1)
