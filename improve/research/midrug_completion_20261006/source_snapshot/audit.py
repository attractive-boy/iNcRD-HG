"""Replay full score matrices; independently check objective and stationarity."""
import json
from pathlib import Path
import numpy as np
from scipy import linalg
from scipy.special import expit
from sklearn.metrics import roc_auc_score, average_precision_score
from common import HERE, load_inner, make_graph, sha256
from model import CONFIGS, laplacian, smooth, shrink, intercept


def main():
    plan = json.loads((HERE/'results/plan.json').read_text())
    for p,h in plan['source_hashes'].items():
        assert sha256(p) == h
    previous_path = HERE.parent/'midrug_reconstruction_ranking_20261006/verification.json'
    previous = json.loads(previous_path.read_text())
    assert previous['status'] == 'passed'
    passed_tags = {r['tag'] for r in previous['records'] if r['reused_control']}
    records = []
    for fold in range(3):
        data = load_inner(fold)
        train_y = data['train_y'].astype(float)
        assert len(np.intersect1d(data['train_indices'], data['val_indices'])) == 0
        assert len(data['train_indices'])+len(data['val_indices']) == 9712
        assert len(data['test_pairs']) == 0 and 'test_y' not in data
        graph = make_graph(data)
        assert set(graph.node_types) == {'miRNA','drug'}
        lap = [laplacian(graph[n].num_nodes, graph[n, rel, n].edge_index.numpy())
               for n,rel in [('miRNA','MiMi'),('drug','DrugDrug')]]
        # Holding out labels affects neither permitted graph nor training records.
        perturbed = dict(data, val_y=1-data['val_y'])
        g2 = make_graph(perturbed)
        for e in graph.edge_types:
            np.testing.assert_array_equal(graph[e].edge_index.numpy(), g2[e].edge_index.numpy())
        for name, byfold in plan['controls'].items():
            control = byfold[str(fold)]
            p = Path(control['path'])
            assert p.stem in passed_tags
            assert sha256(p.with_suffix('.pt')) == control['checkpoint_sha256']
            pred = p.with_name(p.stem+'_inner.npz')
            assert sha256(pred) == control['prediction_sha256']
            r = json.loads(p.read_text())
            with np.load(pred) as z:
                for key,dkey in [('pairs','val_pairs'),('y','val_y'),('indices','val_indices')]:
                    np.testing.assert_array_equal(z[key],data[dkey])
                assert abs(roc_auc_score(z['y'],z['probability'])-r['inner_auc']) < 1e-12
                assert abs(average_precision_score(z['y'],z['probability'])-r['inner_ap']) < 1e-12
        for name,cfg in CONFIGS.items():
            tag = f'{name}_f{fold}'
            path = HERE/'results'/f'{tag}.json'
            result = json.loads(path.read_text())
            assert result['settings'] == cfg and result['no_lncrna']
            assert not result['outer_validation_evaluated'] and not result['test_evaluated']
            assert sha256(path.with_suffix('.npz')) == result['checkpoint_sha256']
            assert sha256(HERE/'results'/f'{tag}_inner.npz') == result['prediction_sha256']
            info = result['fit_info']
            with np.load(path.with_suffix('.npz')) as z:
                x = z['residual']
                assert float(z['intercept']) == info['intercept']
                assert float(z['nominal_intercept']) == info['nominal_intercept']
            b,nominal = intercept(train_y,cfg['error_cost'])
            assert b == info['intercept'] and nominal == info['nominal_intercept']
            val,g,parts = smooth(x,data['train_pairs'],train_y,b,cfg['error_cost'],*lap,cfg['graph_weight'])
            s = linalg.svdvals(x)
            objective = val+info['nuclear_weight']*s.sum()
            np.testing.assert_allclose(objective,info['curves'][-1]['objective'],atol=1e-7,rtol=1e-10)
            for key in parts:
                np.testing.assert_allclose(parts[key],info['curves'][-1][key],atol=1e-7,rtol=1e-10)
            _,g0,_ = smooth(np.zeros_like(x),data['train_pairs'],train_y,b,cfg['error_cost'],*lap,0.)
            np.testing.assert_allclose(linalg.svdvals(g0)[0]*cfg['nuclear_fraction'],info['nuclear_weight'],atol=1e-10)
            step = info['step_size']
            prox,_,_ = shrink(x-step*g,step*info['nuclear_weight'])
            residual = float(linalg.norm(x-prox)/step/max(1.,linalg.norm(x)))
            assert abs(residual-info['curves'][-1]['stationarity_residual']) < 1e-10
            assert info['converged'] == (residual <= plan['stationarity_tolerance'])
            objectives = [c['objective'] for c in info['curves']]
            assert np.max(np.diff(objectives)) <= 1e-7
            if cfg['error_cost'] is not None:
                row,col = data['train_pairs'].T
                z = b+x[row,col]
                sign = 2*train_y-1
                tau = cfg['error_cost']
                correction = sign*np.maximum(np.log((1-tau)/tau)-sign*z,0.)
                explicit = np.logaddexp(0.,-sign*(z+correction)).sum()+tau*abs(correction).sum()
                np.testing.assert_allclose(explicit,parts['observed_loss'],atol=1e-8)
            row,col = data['val_pairs'].T
            score = nominal+x[row,col]
            with np.load(HERE/'results'/f'{tag}_inner.npz') as z:
                for key,dkey in [('pairs','val_pairs'),('y','val_y'),('indices','val_indices')]:
                    np.testing.assert_array_equal(z[key],data[dkey])
                np.testing.assert_array_equal(score,z['score'])
                np.testing.assert_array_equal(expit(score),z['probability'])
            assert abs(roc_auc_score(data['val_y'],score)-result['inner_auc']) < 1e-12
            assert abs(average_precision_score(data['val_y'],score)-result['inner_ap']) < 1e-12
            records.append(dict(tag=tag,status='passed',converged=info['converged'],
                stationarity_residual=residual,rank=int(np.sum(s>1e-8)),max_prediction_error=0.))
            print('PASS',tag,flush=True)
    (HERE/'verification.json').write_text(json.dumps(dict(status='passed',records=records,
        prior_control_replay_sha256=sha256(previous_path),unchanged_controls_reused=6,
        heldout_labels_excluded_from_fit=True,no_lncrna=True,
        outer_validation_evaluated=False,test_evaluated=False),indent=2))


if __name__ == '__main__':
    main()
