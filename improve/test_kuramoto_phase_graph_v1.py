"""Pinned KoPE routines, independent circular calculus and graph boundaries."""
import ast
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from improve import kuramoto_phase_graph_v1 as core
from improve import anchor_kuramoto_phase_experiment_v1 as experiment
from improve import tabular_foundation as base
from improve.test_nested_anchor_v2 import fixture

HERE=Path(__file__).resolve().parent/'research/oscillatory_next_20260916'
EVIDENCE={}
COUNT_KEYS=('calls','attention_layers','phase_updates','phase_entries',
    'feature_score_edge_heads','coupling_score_edge_heads','rotary_pairs')


def arrays():
    a,_=fixture(); a['anchor_oof']=np.linspace(-.5,.5,len(a['y_train']))
    return a


def small_protocol():
    cp=dict(core.parameters(),width=8,heads=2,depth=2)
    return dict(smoke=True,family='kuramoto_phase',implementation_module='kuramoto_phase_graph_v1',
        source_sha256={'kuramoto_phase_graph_v1.py':base.digest(core.__file__)},
        core_parameters=cp,kinds=list(core.KINDS),epochs=5,check_interval=5,
        weight_decay=.001,strong_weight=.1)


def author_routines():
    row=next(r for r in base.read_json(HERE/'KOPE_AUTHOR_SOURCE_V1.json') if r['path']=='vit_kope.py')
    path=Path(row['file'])
    assert base.digest(path)==row['sha256'] and row['git_blob_verified']
    assert row['commit']=='0eade3cebc76978d2944f3ebb9f50f57f1b7ee53'
    module=ast.parse(path.read_text())
    names=('rmsnorm','apply_rotary_pairs','KoPEAttention','KoPECoupling')
    selected=[node for node in module.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names]
    assert len(selected)==len(names)
    namespace=dict(torch=torch,nn=nn,F=F,Tuple=Tuple)
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)
    return namespace,row


class KuramotoChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.random.default_generator.manual_seed(852)

    def test_pinned_author_attention_and_coupling_gradients(self):
        author,row=author_routines()
        p=dict(core.parameters(),width=8,heads=2)
        n=4
        edges=torch.stack((torch.arange(n).repeat(n),torch.arange(n).repeat_interleave(n)))
        relation=torch.zeros(n*n,dtype=torch.long)
        h=torch.randn(n,8,dtype=torch.float64,requires_grad=True)
        theta=torch.randn(n,2,2,dtype=torch.float64)
        c=theta.cos().requires_grad_(); s=theta.sin().requires_grad_()
        def layout(x): return x.permute(1,0,2).unsqueeze(0)
        local=core.PhaseAttention(8,2,1,0.,'full').double()
        reference=author['KoPEAttention'](8,2,qkv_bias=False).double()
        with torch.no_grad():
            local.edge.weight.zero_()
            reference.qkv.weight.copy_(local.qkv.weight)
            reference.proj.weight.copy_(local.output.weight)
            reference.proj.bias.copy_(local.output.bias)
            reference.kope_mix_coef.copy_(local.mix)
        y=local.attention(h,c,s,edges,relation,p)[0]
        z=reference(h[None],layout(c),layout(s))[0]
        torch.testing.assert_close(y,z,atol=2e-13,rtol=2e-13)
        gl=torch.autograd.grad(y.square().sum(),[h,c,s,local.qkv.weight,local.output.weight,
            local.output.bias,local.mix],retain_graph=True)
        gr=torch.autograd.grad(z.square().sum(),[h,c,s,reference.qkv.weight,reference.proj.weight,
            reference.proj.bias,reference.kope_mix_coef],retain_graph=True)
        for left,right in zip(gl,gr): torch.testing.assert_close(left,right,atol=5e-12,rtol=5e-12)
        attention_error=float((y-z).detach().abs().max())
        attention_gradient_error=max(float((a-b).abs().max()) for a,b in zip(gl,gr))
        local_c=core.SharedPhaseCoupling(8,2,1).double()
        source_c=author['KoPECoupling'](8,2,qkv_bias=False).double()
        with torch.no_grad():
            local_c.edge.weight.zero_()
            source_c.q_proj.weight.copy_(local_c.q_proj.weight)
            source_c.k_proj.weight.copy_(local_c.k_proj.weight)
            source_c.qk_scale.copy_(local_c.qk_scale[None,:,None,:])
        yc,ys,_,_=local_c(h,c,s,edges,relation,p)
        zc,zs=source_c(h[None],layout(c),layout(s),h.new_tensor(p['gamma']))
        zc,zs=(v[0].permute(1,0,2) for v in (zc,zs))
        for left,right in ((yc,zc),(ys,zs)): torch.testing.assert_close(left,right,atol=2e-13,rtol=2e-13)
        weights=torch.randn_like(yc)
        gl=torch.autograd.grad((weights*(yc+2*ys)).sum(),[h,c,s,local_c.q_proj.weight,
            local_c.k_proj.weight,local_c.qk_scale],retain_graph=True)
        gr=torch.autograd.grad((weights*(zc+2*zs)).sum(),[h,c,s,source_c.q_proj.weight,
            source_c.k_proj.weight,source_c.qk_scale])
        gr=list(gr); gr[-1]=gr[-1][0,:,0,:]
        for left,right in zip(gl,gr): torch.testing.assert_close(left,right,atol=5e-12,rtol=5e-12)
        EVIDENCE['author']=dict(commit=row['commit'],source_sha256=row['sha256'],
            unchanged_selected_classes_and_functions=True,complete_support_graph=True,
            typed_edge_embeddings_zero_for_source_equivalence=True,
            full_author_ViT_or_training_executed=False,
            attention_forward_error=attention_error,attention_gradient_error=attention_gradient_error,
            coupling_forward_error=max(float((yc-zc).detach().abs().max()),float((ys-zs).detach().abs().max())),
            coupling_gradient_error=max(float((a-b).abs().max()) for a,b in zip(gl,gr)))

    def test_independent_phase_circle_rotation_and_gauge_laws(self):
        theta=torch.randn(3,2,2,dtype=torch.float64)
        c,s=theta.cos(),theta.sin()
        mc,ms=torch.randn_like(c),torch.randn_like(s)
        gamma=c.new_tensor(.05,requires_grad=True)
        actual_c,actual_s=core.phase_step(c,s,mc,ms,gamma,1e-5)
        angle=theta.detach().numpy()+np.arctan(.05*(ms.numpy()*c.numpy()-mc.numpy()*s.numpy()))
        np.testing.assert_allclose(actual_c.detach().numpy(),np.cos(angle),atol=1e-13,rtol=1e-13)
        np.testing.assert_allclose(actual_s.detach().numpy(),np.sin(angle),atol=1e-13,rtol=1e-13)
        inputs=tuple(x.detach().requires_grad_() for x in (c,s,mc,ms))+(gamma,)
        self.assertTrue(torch.autograd.gradcheck(lambda a,b,d,e,g:core.phase_step(a,b,d,e,g,1e-5),
            inputs,atol=1e-6,rtol=1e-4))
        x=torch.randn(3,2,4,dtype=torch.float64,requires_grad=True)
        rotated=core.rotate_pairs(x,c,s)
        inverse=core.rotate_pairs(rotated,c,-s)
        torch.testing.assert_close(inverse,x,atol=1e-13,rtol=1e-13)
        torch.testing.assert_close(rotated.square().sum(-1),x.square().sum(-1),atol=1e-13,rtol=1e-13)
        n=3; p=dict(core.parameters(),width=8,heads=2)
        edges=torch.tensor([[0,1,2,1,2,0],[0,0,1,1,2,2]])
        relation=torch.tensor([0,1,0,1,0,1])
        layer=core.PhaseAttention(8,2,2,0.,'full').double()
        h=torch.randn(n,8,dtype=torch.float64)
        y=layer.attention(h,c,s,edges,relation,p)[0]
        shifted=layer.attention(h,(theta+.37).cos(),(theta+.37).sin(),edges,relation,p)[0]
        torch.testing.assert_close(y,shifted,atol=1e-12,rtol=1e-12)
        cp=core.SharedPhaseCoupling(8,2,2).double()
        same_c=torch.full_like(c,np.cos(.8)); same_s=torch.full_like(s,np.sin(.8))
        yc,ys,_,_=cp(h,same_c,same_s,edges,relation,p)
        torch.testing.assert_close(yc,same_c,atol=1e-13,rtol=1e-13)
        torch.testing.assert_close(ys,same_s,atol=1e-13,rtol=1e-13)
        EVIDENCE['circular']=dict(independent_arctangent_step=True,all_phase_step_gradcheck=True,
            rotation_inverse_and_norm_preservation=True,synchronized_state_fixed_point=True,
            global_common_phase_gauge_error=float((y-shifted).detach().abs().max()))

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
            self.assertEqual(detail['phase_diagnostics']['calls'],5)
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
            records.append({key:detail['phase_diagnostics'][key] for key in COUNT_KEYS})
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

    def test_controls_initialization_freezing_and_plain_attention_equivalence(self):
        a,p=arrays(),small_protocol()
        models,states,rngs,frozen=[],[],[],{}
        for kind in core.KINDS:
            torch.random.default_generator.manual_seed(619)
            model,xs,graphs,folds=core.setup(a,p['core_parameters'],kind,'cpu')
            models.append(model)
            states.append(experiment.cpu_copy(model.state_dict()))
            rngs.append(torch.random.get_rng_state())
            expected=set()
            if kind!='full':
                expected.update(name for name,_ in model.named_parameters() if name.startswith('coupling.'))
            if kind=='no_phase_rotation':
                expected.add('phase_init.weight')
                expected.update('layers.'+str(i)+'.mix' for i in range(p['core_parameters']['depth']))
            actual={name for name,value in model.named_parameters() if not value.requires_grad}
            self.assertEqual(actual,expected)
            frozen[kind]=sorted(actual)
        for state,rng in zip(states[1:],rngs[1:]):
            self.assertEqual(set(state),set(states[0]))
            for name,value in state.items():
                torch.testing.assert_close(value,states[0][name],atol=0.,rtol=0.)
            torch.testing.assert_close(rng,rngs[0],atol=0.,rtol=0.)
        # Check the entire feature block, including its residuals and FFN.
        local=core.PhaseAttention(8,2,3,0.,'no_phase_rotation').double()
        plain=core.graph_lib.SparseTypedAttention(8,2,3,0.).double()
        plain.load_state_dict({k:v for k,v in local.state_dict().items() if k!='mix'},strict=True)
        h=torch.randn(4,8,dtype=torch.float64,requires_grad=True)
        angle=torch.randn(4,2,2,dtype=torch.float64)
        edges=torch.tensor([[0,1,2,1,3,2,0,3],[0,0,1,1,2,2,3,3]])
        relation=torch.tensor([0,1,2,0,1,2,2,0])
        actual=local(h,angle.cos(),angle.sin(),edges,relation,p['core_parameters'],core.empty_diagnostics())
        expected=plain(h,edges,relation)
        torch.testing.assert_close(actual,expected,atol=2e-13,rtol=2e-13)
        names=[name for name,_ in plain.named_parameters()]
        gl=torch.autograd.grad(actual.square().sum(),[h]+[dict(local.named_parameters())[name] for name in names])
        gr=torch.autograd.grad(expected.square().sum(),[h]+list(plain.parameters()))
        for left,right in zip(gl,gr):
            torch.testing.assert_close(left,right,atol=5e-12,rtol=5e-12)
        # A zero step size makes the full and static phase models equivalent,
        # with a nonzero readout so that equality cannot be vacuous.
        query=torch.as_tensor(a['pairs_train'][:,folds[0]]+a['pair_offset'][:,None])
        with torch.no_grad():
            models[0].readout[-1].weight.normal_()
            models[1].load_state_dict(models[0].state_dict(),strict=True)
            for model in models[:2]:
                model.eval(); model.p['gamma']=0.
            full=models[0](xs,graphs[0],query)
            static=models[1](xs,graphs[0],query)
        torch.testing.assert_close(full,static,atol=3e-6,rtol=3e-6)
        self.assertGreater(float(full.abs().max()),1e-6)
        EVIDENCE['controls']=dict(initial_stored_parameters_identical=True,initial_rng_identical=True,
            frozen_parameter_names=frozen,plain_attention_forward_error=float((actual-expected).detach().abs().max()),
            plain_attention_gradient_error=max(float((a-b).abs().max()) for a,b in zip(gl,gr)),
            zero_gamma_full_static_error=float((full-static).abs().max()),effective_capacities_equal_claimed=False)

    def test_sparse_typed_coupling_numpy_reference_and_gradients(self):
        p=dict(core.parameters(),width=8,heads=2)
        coupling=core.SharedPhaseCoupling(8,2,3).double()
        h=torch.randn(5,8,dtype=torch.float64,requires_grad=True)
        theta=torch.randn(5,2,2,dtype=torch.float64,requires_grad=True)
        edges=torch.tensor([[0,1,2,1,3,2,4,3,0,4],[0,0,1,1,2,2,3,3,4,4]])
        relation=torch.tensor([0,1,2,0,1,2,2,0,1,0])
        src,dst=edges.numpy(); rel=relation.numpy()
        names=['q_proj.weight','k_proj.weight','qk_scale','edge.weight']
        tensors=[h,theta]+[dict(coupling.named_parameters())[name] for name in names]
        original=[v.detach().numpy().copy() for v in tensors]
        def independent(values):
            features,angle,qweight,kweight,scale,embedding=values
            q=(features@qweight.T).reshape(5,2,4)
            k=(features@kweight.T).reshape(5,2,4)
            eps=np.finfo(np.float64).eps
            q=q/np.sqrt(np.mean(q*q,axis=-1,keepdims=True)+eps)*scale
            k=k/np.sqrt(np.mean(k*k,axis=-1,keepdims=True)+eps)
            scores=np.sum(q[dst]*(k[src]+embedding[rel].reshape(-1,2,4)),axis=-1)/2.
            cos,sin=np.cos(angle),np.sin(angle)
            mean_cos,mean_sin=np.zeros_like(cos),np.zeros_like(sin)
            for node in range(5):
                ids=np.flatnonzero(dst==node)
                logits=scores[ids]-np.max(scores[ids],axis=0,keepdims=True)
                weights=np.exp(logits); weights/=np.sum(weights,axis=0,keepdims=True)
                mean_cos[node]=np.sum(weights[...,None]*cos[src[ids]],axis=0)
                mean_sin[node]=np.sum(weights[...,None]*sin[src[ids]],axis=0)
            updated=angle+np.arctan(p['gamma']*(mean_sin*cos-mean_cos*sin))
            return np.cos(updated),np.sin(updated)
        yc,ys,_,_=coupling(h,theta.cos(),theta.sin(),edges,relation,p)
        nc,ns=independent(original)
        np.testing.assert_allclose(yc.detach().numpy(),nc,atol=2e-13,rtol=2e-13)
        np.testing.assert_allclose(ys.detach().numpy(),ns,atol=2e-13,rtol=2e-13)
        wc,ws=torch.randn_like(yc),torch.randn_like(ys)
        gradients=torch.autograd.grad((wc*yc+ws*ys).sum(),tensors)
        errors={}
        generator=np.random.default_rng(907)
        for index,(name,gradient) in enumerate(zip(['features','theta']+names,gradients)):
            direction=generator.normal(size=original[index].shape)
            direction/=np.linalg.norm(direction)
            plus=[v.copy() for v in original]; minus=[v.copy() for v in original]
            plus[index]+=1e-5*direction; minus[index]-=1e-5*direction
            pc,ps=independent(plus); mc,ms=independent(minus)
            finite_difference=float(np.sum(wc.numpy()*(pc-mc)+ws.numpy()*(ps-ms))/(2e-5))
            autodiff=float(np.sum(gradient.detach().numpy()*direction))
            errors[name]=abs(finite_difference-autodiff)
            np.testing.assert_allclose(autodiff,finite_difference,atol=2e-8,rtol=2e-6)
        self.assertTrue(torch.autograd.gradcheck(
            lambda x,t:coupling(x,t.cos(),t.sin(),edges,relation,p)[:2],(h,theta),atol=1e-6,rtol=1e-4))
        EVIDENCE['sparse_reference']=dict(nodes=5,edges=10,types=3,complete_graph=False,
            independent_numpy_softmax_and_arctangent=True,rms_epsilon=float(np.finfo(np.float64).eps),
            forward_max_error=max(float(np.max(abs(yc.detach().numpy()-nc))),float(np.max(abs(ys.detach().numpy()-ns)))),
            independent_directional_gradient_errors=errors,full_feature_and_angle_gradcheck=True)

    def test_failure_records_exact_partial_state(self):
        original=core.KuramotoPhaseGraph.forward
        def fail_third(model,*args):
            if model.phase_diagnostics['calls']==2: raise FloatingPointError('Synthetic registered failure')
            return original(model,*args)
        with patch.object(core.KuramotoPhaseGraph,'forward',fail_third):
            with self.assertRaisesRegex(FloatingPointError,'Synthetic registered failure') as caught:
                experiment.fit(arrays(),small_protocol(),'full',.001,112,'cpu',fixed_epochs=5)
        state,context=caught.exception.training_partial_state,caught.exception.training_failure_context
        self.assertEqual(state['steps'],2); self.assertEqual(context['completed_steps'],2)
        self.assertEqual(context['current_step'],3); self.assertEqual(context['operation'],'forward')
        self.assertTrue(all(int(v['step'])==2 for v in state['optimizer']['state'].values()))
        EVIDENCE['failure']=dict(exact_completed_steps=2,current_step=3,operation='forward',
            partial_model_optimizer_rng_preserved=True,formal_neural_fit=False)


if __name__=='__main__':
    destination=HERE/'KURAMOTO_PHASE_UNIT_V1.json'
    if destination.exists(): raise ValueError('Preserve existing unit evidence')
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(KuramotoChecks)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    sources=[Path(__file__),Path(core.__file__),Path(experiment.__file__),
        experiment.ROOT/'test_nested_anchor_v2.py',experiment.ROOT/'overlap_block_models.py',
        experiment.ROOT/'sharpness_graph_v1.py',HERE/'KOPE_AUTHOR_SOURCE_V1.json',
        HERE/'kope_author_v1_vit_kope.py',HERE/'kope_author_v1_LICENSE']
    base.write_json(destination,base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        tests=result.testsRun,success=result.wasSuccessful(),failures=len(result.failures),errors=len(result.errors),
        evidence=EVIDENCE,source_sha256={str(path):base.digest(path) for path in sources},
        formal_neural_fits=0,evaluation_labels_read=False)))
    raise SystemExit(0 if result.wasSuccessful() else 1)
