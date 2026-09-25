"""Information-boundary and mechanism checks for the compact LinkPFN."""
import unittest

import numpy as np
import torch

from improve.link_pfn_modules import (
    ATTR_DIM, FEATURE_DIM, LinkFeatures, LinkPFN, balanced_indices,
    hide_drugs, observed_matrix, synthetic_episode, tensor_batch,
)
from improve.link_pfn_data import reserve_support, scenario, drug_groups, project_attributes


def toy_data():
    rng = np.random.RandomState(47)
    nr, no, nd = 12, 8, 15
    d = dict(rna_0=rng.normal(size=(nr,ATTR_DIM)).astype(np.float32),
             rna_1=rng.normal(size=(no,ATTR_DIM)).astype(np.float32),
             drug=rng.normal(size=(nd,ATTR_DIM)).astype(np.float32),
             side_0=(rng.rand(nr,nr)<.25).astype(np.float32),
             side_1=(rng.rand(no,no)<.25).astype(np.float32),
             side_drug=(rng.rand(nd,nd)<.25).astype(np.float32),
             cross=(rng.rand(nr,no)<.25).astype(np.float32),drug_group=np.arange(nd)%5)
    for i,n in enumerate((nr,no)):
        r,drug=np.meshgrid(np.arange(n),np.arange(nd),indexing='ij')
        pairs=np.stack([r.ravel(),drug.ravel()])
        y=((pairs[0]+pairs[1])%3==0).astype(np.float32)
        d['pairs_'+str(i)],d['labels_'+str(i)]=pairs,y
        for phase,which in (('fit',d['drug_group']<3),('dev',d['drug_group']==3),('assess',d['drug_group']==4)):
            ids=np.flatnonzero(which[pairs[1]])
            d[phase+'_idx_'+str(i)]=ids
            if phase!='fit':
                d[phase+'_reserved_'+str(i)]=reserve_support(pairs,y,ids,61+i)
    return d


class LinkPFNTests(unittest.TestCase):
    def test_attribute_projection_does_not_fit_query_drugs(self):
        rng=np.random.RandomState(21)
        raw=rng.normal(size=(40,24))
        altered=raw.copy()
        altered[30:]=100*rng.normal(size=(10,24))
        a,pa=project_attributes(raw,np.arange(30),29)
        b,pb=project_attributes(altered,np.arange(30),29)
        np.testing.assert_array_equal(a[:30],b[:30])
        for key in pa:
            np.testing.assert_array_equal(pa[key],pb[key])

    def test_checkpoint_rule_retains_initial_model_and_selects_real_step(self):
        from improve.link_pfn import choose_checkpoint
        trace=[dict(step=s,criterion=c) for s,c in [(0,.8),(20,.7),(40,.75),(60,.76)]]
        self.assertEqual(choose_checkpoint(trace,3)[0],0)
        trace.extend(dict(step=s,criterion=c) for s,c in [(80,.9),(100,.92),(120,.93)])
        self.assertEqual(choose_checkpoint(trace,3)[0],120)

    def test_known_query_rejected_and_empty_support_finite(self):
        d=toy_data()
        a=observed_matrix(d['pairs_0'],d['labels_0'],12,15)
        f=LinkFeatures(d['rna_0'],d['drug'],d['side_0'],d['side_drug'],d['cross'],a,np.zeros((8,15)),0)
        p=d['pairs_0'][:,d['labels_0']==1][:,:1]
        with self.assertRaises(ValueError):
            f.transform(p)
        x=f.transform(p,context=True)
        self.assertEqual(x.shape,(1,FEATURE_DIM))
        z=LinkFeatures(d['rna_0'],d['drug'],d['side_0'],d['side_drug'],d['cross'],a*0,np.zeros((8,15)),0)
        self.assertTrue(np.isfinite(z.transform(d['pairs_0'])).all())

    def test_cold_mask_removes_both_rna_types_and_reverse_views(self):
        d=toy_data()
        aa=[observed_matrix(d['pairs_'+str(i)],d['labels_'+str(i)],n,15) for i,n in enumerate((12,8))]
        hidden=[1,3,7]
        masked=hide_drugs(aa,hidden)
        for a,b in zip(aa,masked):
            self.assertFalse(b[:,hidden].any())
            self.assertFalse(b.T[hidden].any())
            self.assertTrue(a[:,hidden].any())
        for task in (0,1):
            s=scenario(d,task,'dev')
            self.assertFalse(s['support_counts'][s['held_drugs']].any())

    def test_assessment_labels_cannot_change_fit_or_dev_inputs(self):
        d=toy_data()
        altered={k:v.copy() for k,v in d.items()}
        for i in (0,1):
            ids=d['assess_idx_'+str(i)]
            altered['labels_'+str(i)][ids]=1-altered['labels_'+str(i)][ids]
        for phase,fold in (('fit',0),('dev',None)):
            for task in (0,1):
                a,b=scenario(d,task,phase,fold),scenario(altered,task,phase,fold)
                for k in ('cx','cy','cp','qx','qy','qp'):
                    np.testing.assert_array_equal(a[k],b[k])

    def test_fitting_query_drug_labels_cannot_change_its_inputs(self):
        d=toy_data()
        altered={k:v.copy() for k,v in d.items()}
        for i in (0,1):
            mask=d['drug_group'][d['pairs_'+str(i)][1]]==0
            altered['labels_'+str(i)][mask]=1-altered['labels_'+str(i)][mask]
        a,b=scenario(d,0,'fit',0),scenario(altered,0,'fit',0)
        for k in ('cx','cy','cp','qx','qp'):
            np.testing.assert_array_equal(a[k],b[k])

    def test_fewshot_queries_fixed_and_supports_disjoint(self):
        d=toy_data()
        a=scenario(d,0,'assess',support_budget=0,full_candidates=True)
        b=scenario(d,0,'assess',support_budget=3,full_candidates=True)
        np.testing.assert_array_equal(a['query_indices'],b['query_indices'])
        self.assertFalse(np.intersect1d(b['context_indices'],b['query_indices']).size)
        self.assertTrue(len(b['priority'])>0)
        self.assertEqual(len(a['all_p'])-len(b['all_p']),len(b['priority']))
        ids=balanced_indices(b['cy'],32,np.random.RandomState(41),b['priority'])
        self.assertTrue(set(b['priority'])<=set(ids))

    def test_group_assignment_is_reproducible_and_disjoint(self):
        d=toy_data()
        pairs=[d['pairs_0'],d['pairs_1']]
        labels=[d['labels_0'],d['labels_1']]
        a,_=drug_groups(pairs,labels,15,90)
        b,_=drug_groups(pairs,labels,15,90)
        np.testing.assert_array_equal(a,b)
        self.assertEqual(set(a),set(range(5)))

    def test_priors_share_latent_queries_but_change_observation(self):
        differences=[]
        for seed in (719,721,723):
            a,ma=synthetic_episode(seed,'hetero',32,24)
            b,mb=synthetic_episode(seed,'missing',32,24)
            for k in ('qp','qy'):
                np.testing.assert_array_equal(a[k],b[k])
            self.assertEqual(ma['truth_checksum'],mb['truth_checksum'])
            self.assertEqual(ma['query_support_overlap'],0)
            self.assertEqual(mb['query_support_overlap'],0)
            self.assertEqual(ma['observation'],mb['observation'])
            differences.append(np.max(np.abs(a['qx']-b['qx'])))
        self.assertGreater(max(differences),.01)

    def test_synthetic_false_positives_never_enter_queries(self):
        for p in ('random','hetero','missing'):
            a,m=synthetic_episode(108,p,32,24,shift=True,false_positive=.1)
            self.assertEqual(m['query_support_overlap'],0)
            self.assertTrue(np.isfinite(a['cx']).all())
            self.assertEqual(set(a['qy']),{0,1})

    def model_fixture(self):
        torch.manual_seed(91)
        ep,_=synthetic_episode(781,'missing',24,12)
        batch=tensor_batch(**{k:ep[k] for k in ('cx','cy','cp','qx','qp')},device='cpu')
        model=LinkPFN(width=16,heads=4,depth=2,dropout=0.)
        model.eval()
        return model,batch

    def test_context_and_query_permutation_equivariance(self):
        model,batch=self.model_fixture()
        expected=model(**batch)
        perm=torch.randperm(batch['cx'].shape[1])
        changed=dict(batch)
        for k in ('cx','cy','cp','raw_cx'):
            changed[k]=batch[k][:,perm]
        torch.testing.assert_close(model(**changed),expected,atol=1e-6,rtol=1e-6)
        perm=torch.randperm(batch['qx'].shape[1])
        changed=dict(batch)
        for k in ('qx','qp','raw_qx'):
            changed[k]=batch[k][:,perm]
        torch.testing.assert_close(model(**changed),expected[:,perm],atol=1e-6,rtol=1e-6)

    def test_query_chunks_and_entity_renumbering_preserve_prediction(self):
        model,batch=self.model_fixture()
        expected=model(**batch)
        changed=dict(batch)
        for k in ('qx','qp','raw_qx'):
            changed[k]=batch[k][:,:3]
        torch.testing.assert_close(model(**changed),expected[:,:3],atol=1e-6,rtol=1e-6)
        changed=dict(batch)
        for k in ('cp','qp'):
            changed[k]=batch[k]*13+29
        torch.testing.assert_close(model(**changed),expected,atol=0,rtol=0)

    def test_query_values_do_not_normalize_context_and_type_survives(self):
        ep,_=synthetic_episode(781,'missing',24,12)
        keys=('cx','cy','cp','qx','qp')
        a=tensor_batch(**{k:ep[k] for k in keys},device='cpu')
        ep['qx']=ep['qx']*100
        b=tensor_batch(**{k:ep[k] for k in keys},device='cpu')
        torch.testing.assert_close(a['cx'],b['cx'],atol=0,rtol=0)
        self.assertEqual(float(a['cx'][0,0,-1]),1.)

    def test_context_labels_and_relational_parameters_have_gradients(self):
        model,batch=self.model_fixture()
        output=model(**batch)
        other=model(**batch,intervention='shuffle_context_labels')
        self.assertGreater(float((output-other).abs().max()),1e-6)
        output.square().mean().backward()
        for name,param in model.named_parameters():
            self.assertIsNotNone(param.grad,name)
            self.assertTrue(bool(torch.isfinite(param.grad).all()),name)
        self.assertGreater(float(model.label.weight.grad.abs().sum()),0.)
        self.assertGreater(float(model.blocks[0].attention.relation.grad.abs().sum()),0.)


if __name__=='__main__':
    unittest.main()
