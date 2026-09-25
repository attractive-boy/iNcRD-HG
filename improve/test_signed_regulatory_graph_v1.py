"""Regulatory sign semantics, independent dynamics, derivatives and boundaries."""
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import sparse
from scipy.special import expit
import torch

from improve import signed_regulatory_graph_v1 as core
from improve import signed_regulatory_inputs_v1 as data_lib
from improve import anchor_signed_regulatory_experiment_v1 as experiment
from improve import tabular_foundation as base
from improve.test_nested_anchor_v2 import fixture

HERE=Path(__file__).resolve().parent/'research/signed_causal_flow_20260916'
EVIDENCE={}


def synthetic_evidence(directory):
    edges=np.array([[0,2,0,3,1,3,1,2],[1,1,2,2,3,0,0,3]])
    signs=np.array([1,-1,-1,1,1,-1,1,-1],dtype=np.float32)
    rna=np.zeros((8,4),dtype=np.float32)
    drug=np.zeros((6,4),dtype=np.float32)
    for row in range(7): rna[row,row%4]=1.;rna[row,(row+1)%4]=.25
    for row in range(5): drug[row,row%4]=1.
    data=dict(regulatory_edges=edges,regulatory_sign=signs,
        shuffled_sign=data_lib.shuffled_signs(edges,signs),genes=np.array(['a','b','c','d']),
        rna_names=np.array([str(i) for i in range(8)]),drug_names=np.array([str(i) for i in range(6)]),
        **data_lib.encode_sparse(sparse.csr_matrix(rna),'rna'),
        **data_lib.encode_sparse(sparse.csr_matrix(drug),'drug'))
    path,manifest=directory/'arrays.npz',directory/'DATA_AUDIT.json'
    np.savez_compressed(path,**data)
    base.write_json(manifest,base.seal(dict(association_labels_read=False,
        no_binding_action_direction_inferred=True,source_sha256={},input_sha256={})))
    return data,dict(evidence_arrays=str(path),evidence_sha256=base.digest(path),
        evidence_manifest=str(manifest),evidence_manifest_sha256=base.digest(manifest))


class SignedRegulatoryChecks(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.random.default_generator.manual_seed(427)
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.data,files=synthetic_evidence(Path(self.temporary.name))
        self.a,_=fixture()
        self.a['anchor_oof']=np.linspace(-.5,.5,len(self.a['y_train']))
        self.p=dict(smoke=True,family='signed_regulatory',implementation_module='signed_regulatory_graph_v1',
            source_sha256={'signed_regulatory_graph_v1.py':base.digest(core.__file__)},
            core_parameters=dict(core.parameters(),**files,width=8,heads=2,depth=1,gene_width=3,regulatory_steps=4),
            kinds=list(core.KINDS),epochs=5,check_interval=5,weight_decay=.001,strong_weight=.1)

    def test_sign_policy_and_stratified_control(self):
        positive=dict(is_directed=True,consensus_direction=True,is_stimulation=True,is_inhibition=False,
                      consensus_stimulation=True,consensus_inhibition=False)
        self.assertEqual(data_lib.unambiguous_sign(positive),1)
        self.assertEqual(data_lib.unambiguous_sign(dict(positive,is_stimulation=False,is_inhibition=True,
            consensus_stimulation=False,consensus_inhibition=True)),-1)
        for altered in (dict(positive,is_inhibition=True),dict(positive,consensus_direction=False),
                        dict(positive,consensus_stimulation=False),dict(positive,is_directed=False)):
            self.assertEqual(data_lib.unambiguous_sign(altered),0)
        e,s=self.data['regulatory_edges'],self.data['regulatory_sign']
        shuffled=self.data['shuffled_sign']
        self.assertGreater(int((s != shuffled).sum()),0)
        for target in np.unique(e[1]):
            ids=e[1]==target
            np.testing.assert_array_equal(np.sort(s[ids]),np.sort(shuffled[ids]))
        EVIDENCE['sign_semantics']=dict(ambiguous_or_conflicting_signs_rejected=True,
            receiver_sign_counts_preserved=True,changed_synthetic_edges=int((s != shuffled).sum()))

    def test_incidence_normalization_and_numpy_dynamics_with_gradients(self):
        pool,inject,rm,cm=core.normalized_incidence(self.data['rna_indices'],self.data['rna_values'],
            self.data['rna_shape'],'cpu')
        dense=sparse.coo_matrix((self.data['rna_values'],self.data['rna_indices']),shape=(8,4)).toarray()
        np.testing.assert_allclose(pool.to_dense(),dense/np.maximum(dense.sum(1,keepdims=True),1),atol=1e-7)
        np.testing.assert_allclose(inject.to_dense(),dense.T/np.maximum(dense.sum(0)[:,None],1e-15),atol=1e-7)
        self.assertEqual(float(pool.to_dense()[-1].abs().sum()),0.)
        operator=core.regulatory_operator(self.data['regulatory_edges'],self.data['regulatory_sign'],4,'cpu').double()
        u,g,b=[torch.randn(4,2,dtype=torch.float64,requires_grad=True) for _ in range(3)]
        actual=core.regulatory_response(operator,u,g,b,8)
        q=operator.to_dense().numpy()
        expected=np.tanh(u.detach().numpy())
        for _ in range(8): expected=np.tanh(2*expit(g.detach().numpy())*(u.detach().numpy()+.4*q@expected)+b.detach().numpy())
        np.testing.assert_allclose(actual.detach(),expected,atol=2e-14,rtol=2e-14)
        self.assertTrue(torch.autograd.gradcheck(lambda x,y,z:core.regulatory_response(operator,x,y,z,4),
                                                (u,g,b),atol=1e-6,rtol=1e-4))
        self.assertLessEqual(float(actual.detach().abs().max()),1.)
        self.assertLessEqual(float(operator.to_dense().abs().sum(1).max()),1.)
        with self.assertRaises(ValueError): core.regulatory_response(operator,u,g,b,4,coupling=.5)
        EVIDENCE['dynamics']=dict(independent_numpy_error=float(np.max(abs(expected-actual.detach().numpy()))),
            forcing_gain_bias_gradcheck=True,normalized_absolute_row_norm_at_most_one=True,
            contraction_upper_bound=.8,finite_iterate_not_fixed_point_claim=True)

    def test_single_path_inhibition_and_gene_relabeling(self):
        edges=np.array([[0,1],[1,2]])
        x=torch.tensor([[1.],[0.],[0.]],dtype=torch.float64)
        zero=torch.zeros_like(x)
        signed=core.regulatory_operator(edges,np.array([1.,-1.]),3,'cpu')
        unsigned=core.regulatory_operator(edges,np.ones(2),3,'cpu')
        a=core.regulatory_response(signed,x,zero,zero,4)
        b=core.regulatory_response(unsigned,x,zero,zero,4)
        self.assertLess(float(a[2]),0.)
        self.assertGreater(float(b[2]),0.)
        torch.testing.assert_close(a[2],-b[2],atol=1e-15,rtol=1e-15)
        order=torch.tensor([2,0,1]);inverse=torch.argsort(order)
        transformed=core.regulatory_operator(inverse.numpy()[edges],np.array([1.,-1.]),3,'cpu')
        changed=core.regulatory_response(transformed,x[order],zero[order],zero[order],4)
        torch.testing.assert_close(changed,a[order],atol=1e-15,rtol=1e-15)
        EVIDENCE['sign_effect']=dict(single_inhibitory_path_output=float(a[2]),
            unsigned_path_output=float(b[2]),gene_relabeling=True)

    def test_training_boundaries_and_query_independence(self):
        a,p=self.a,self.p
        model,xs,graphs,folds=core.setup(a,p['core_parameters'],'full','cpu')
        with torch.no_grad(): model.readout[-1].weight.normal_(0.,.1)
        model.eval()
        pairs=torch.as_tensor(a['pairs_train']+a['pair_offset'][:,None])
        y=model(xs,graphs[0],pairs).detach()
        reverse=model(xs,graphs[0],pairs.flip(1)).detach().flip(0)
        chunks=torch.cat([model(xs,graphs[0],pairs[:,i:i+11]).detach() for i in range(0,pairs.shape[1],11)])
        torch.testing.assert_close(y,reverse,atol=1e-6,rtol=1e-6)
        torch.testing.assert_close(y,chunks,atol=1e-6,rtol=1e-6)
        n=int(a['node_counts'].sum())
        for f,g in enumerate(graphs):
            q=pairs[:,folds[f]].numpy()
            forbidden=np.r_[q[0]*n+q[1],q[1]*n+q[0]]
            e=g[0].numpy()
            self.assertFalse(np.intersect1d(forbidden,e[0]*n+e[1]).size)
        altered={k:v.copy() for k,v in a.items()};altered['y_train']=1-altered['y_train']
        _,_,other,_=core.setup(altered,p['core_parameters'],'full','cpu')
        for g,h in zip(graphs,other):
            for v,w in zip(g,h):torch.testing.assert_close(v,w,atol=0,rtol=0)
        # Relabel RNA/drug entities in both the physical graph and binding maps.
        starts=np.r_[0,np.cumsum(a['node_counts'])]
        old_to_new=np.arange(n);new_to_old=[];px=[];orders=[]
        for i,x in enumerate(xs):
            order=torch.randperm(len(x));orders.append(order.numpy());px.append(x[order])
            old_to_new[starts[i]+order.numpy()]=starts[i]+np.arange(len(x))
            new_to_old.extend((starts[i]+order.numpy()).tolist())
        binding=dict(self.data)
        for prefix,index in (('rna',1),('drug',2)):
            matrix=sparse.coo_matrix((binding[prefix+'_values'],binding[prefix+'_indices']),
                                     shape=tuple(binding[prefix+'_shape'])).tocsr()
            binding.update(data_lib.encode_sparse(matrix[orders[index]],prefix))
        transformed=core.SignedRegulatoryGraph([x.shape[1] for x in px],a['node_counts'],
            int(a['relation'].max())+1,p['core_parameters'],'full',binding,'cpu')
        with torch.no_grad():
            for (_,v),(_,w) in zip(model.named_parameters(),transformed.named_parameters()):w.copy_(v)
        transformed.eval()
        mapping=torch.as_tensor(old_to_new)
        graph=(mapping[graphs[0][0]],graphs[0][1],graphs[0][2][new_to_old])
        changed=transformed(px,graph,mapping[pairs]).detach()
        torch.testing.assert_close(y,changed,atol=2e-6,rtol=2e-6)
        with self.assertRaisesRegex(ValueError,'training arrays only'):
            core.setup(dict(a,y_val=np.zeros(1)),p['core_parameters'],'full','cpu')
        EVIDENCE['boundaries']=dict(query_arcs_omitted=True,query_label_invariance=True,
            query_batching_error=float((y-chunks).abs().max()),
            physical_and_binding_entity_relabel_error=float((y-changed).abs().max()))

    def test_matched_initial_parameters_rng_and_lnc_identity(self):
        a,p=self.a,self.p
        params,rng=None,None;finals=[]
        for kind in core.KINDS:
            torch.random.default_generator.manual_seed(716)
            initial,_,_,_=core.setup(a,p['core_parameters'],kind,'cpu')
            if params is None:
                params={k:v.detach().clone() for k,v in initial.named_parameters()};rng=torch.random.get_rng_state()
            else:
                for k,v in initial.named_parameters():torch.testing.assert_close(v,params[k],atol=0,rtol=0)
                torch.testing.assert_close(torch.random.get_rng_state(),rng,atol=0,rtol=0)
            model,value,trace,detail=experiment.fit(a,p,kind,.003,716,'cpu',fixed_epochs=5)
            self.assertIsNone(value)
            self.assertFalse(any('anchor_composed' in row for row in trace))
            self.assertEqual(detail['optimizer_steps'],5)
            for name,v in model.named_parameters():
                self.assertIsNotNone(v.grad,name);self.assertTrue(torch.isfinite(v.grad).all(),name)
            state=model.training_final_state
            finals.append(state['cpu_rng'])
            for value in state['optimizer']['state'].values():self.assertEqual(int(value['step']),5)
            lnc={k:v.copy() for k,v in a.items()};lnc['pair_offset']=np.array([0,14])
            identity,_,tr,dr=experiment.fit(lnc,p,kind,.003,716,'cpu',fixed_epochs=5)
            self.assertFalse(dr['trained_neural_model']);self.assertFalse(tr)
            self.assertFalse(list(identity.parameters()))
            zero=identity([],None,torch.tensor([[0,1],[14,15]]))
            torch.testing.assert_close(zero,torch.zeros_like(zero),atol=0,rtol=0)
        for r in finals[1:]:torch.testing.assert_close(r,finals[0],atol=0,rtol=0)
        EVIDENCE['training']=dict(initial_learned_parameters_equal=True,rng_equal=True,
            finite_gradients_and_five_updates=True,lnc_exact_identity_without_training=True,
            registered_regulatory_sign_buffers_differ=True)

    def test_failure_snapshot_is_exact(self):
        original,count=core.SignedRegulatoryGraph.forward,[0]
        def broken(model,*args):
            count[0]+=1
            if count[0]==3:raise FloatingPointError('injected regulatory failure')
            return original(model,*args)
        with patch.object(core.SignedRegulatoryGraph,'forward',broken):
            with self.assertRaises(FloatingPointError) as caught:
                experiment.fit(self.a,self.p,'full',.003,716,'cpu',fixed_epochs=5)
        e=caught.exception
        self.assertEqual(e.training_partial_state['steps'],2)
        self.assertEqual(e.training_failure_context['current_step'],3)
        for value in e.training_partial_state['optimizer']['state'].values():self.assertEqual(int(value['step']),2)
        EVIDENCE['failure']=dict(injected_step=3,completed_steps=2,exact_optimizer_snapshot=True)


if __name__ == '__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SignedRegulatoryChecks))
    dest=HERE/'SIGNED_REGULATORY_UNIT_V1.json'
    if dest.exists():raise ValueError('Preserve previous test evidence')
    base.write_json(dest,base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        tests=result.testsRun,success=result.wasSuccessful(),failures=len(result.failures),errors=len(result.errors),
        no_performance_metrics=True,evidence=EVIDENCE,
        source_sha256={str(Path(x).resolve()):base.digest(x) for x in
            [__file__,core.__file__,experiment.__file__,data_lib.__file__,
             Path(__file__).with_name('test_nested_anchor_v2.py')]})))
    raise SystemExit(0 if result.wasSuccessful() else 1)
