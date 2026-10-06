"""Execute the frozen Tensor-product screen and eligible confirmation serially."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from improve import anchor_tensor_product_experiment_v1 as experiment
from improve import tensor_product_attention_graph_v1 as core
from improve import assess_declared_graph as audit
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
OUT = experiment.ROOT / 'results/anchor_priority_tensor_product_v1_20260916'


COUNT_KEYS = ('calls', 'attention_layers', 'node_rows', 'contextual_product_terms',
    'static_product_terms', 'dense_projection_outputs', 'scored_edge_heads', 'value_message_entries')


def counts_for(a, p, step, evaluation_passes):
    """CPU-only forward budget derived from each omitted-fold physical graph."""
    cp=p['core_parameters']
    folds,depth,width,heads=cp['folds'],cp['depth'],cp['width'],cp['heads']
    nodes,relations=int(a['node_counts'].sum()),int(a['relation'].max())+1
    edge_counts=[]
    for fold in range(folds):
        edges,labels=core.graph_lib.masked_physical(a,np.flatnonzero(a['folds']==fold))
        physical=core.graph_lib.graph_inputs(edges,labels,np.empty((2,0),dtype=np.int64),nodes,relations,'cpu')
        edge_counts.append(physical[0].shape[1])
    calls=step+folds*evaluation_passes
    messages=depth*sum(count*(step//folds+int(fold<step%folds)+evaluation_passes)
                       for fold,count in enumerate(edge_counts))
    node_rows=calls*depth*nodes
    terms=node_rows*width*(cp['q_rank']+2*cp['kv_rank'])
    return dict(calls=calls,attention_layers=calls*depth,node_rows=node_rows,
        contextual_product_terms=terms,static_product_terms=terms,
        dense_projection_outputs=node_rows*3*width,scored_edge_heads=messages*heads,
        value_message_entries=messages*width)


def audit_record(path, r, a, p):
    """Read-only identity, forward budget, frozen-weight and optimizer checks."""
    trace=base.read_json(path/'trace.json')
    d=r['tpa_diagnostics']
    if not r['trained_neural_model']:
        assert r['task']=='LncDrug' and r['identity_preserves_anchor'] and not trace
        assert all(r[key]==0 for key in ('epoch','parameters','trainable_parameters','fixed_parameters',
            'training_steps','gradient_evaluations','optimizer_steps'))
        assert d==core.empty_diagnostics() and not (path/'training_final.pt').exists()
        assert not torch.load(path/'model.pt',map_location='cpu',weights_only=True)
        scores=experiment.common.load(path/'scores.npz')
        np.testing.assert_array_equal(scores['standalone'],np.zeros_like(scores['standalone']))
        np.testing.assert_array_equal(scores['anchor'],scores['anchor_composed'])
        return None
    assert r['task']=='MiDrug' and r['trainable_parameters']>0
    assert r['training_steps']==r['gradient_evaluations']==r['optimizer_steps']==400
    assert r['final_optimizer_state_epoch']==400
    assert len(trace)==20 and [row['epoch'] for row in trace]==list(range(20,401,20))
    assert {key:d[key] for key in COUNT_KEYS}==counts_for(a,p,400,21)
    assert all(np.isfinite(v) for v in d.values())
    for row in trace:
        step,diagnostic=row['epoch'],row['tpa_diagnostics']
        assert np.isfinite(row['loss']) and np.isfinite(row['gradient_norm'])
        assert {key:diagnostic[key] for key in COUNT_KEYS}==counts_for(a,p,step,step//20)
        assert all(np.isfinite(v) for v in diagnostic.values())
    final=torch.load(path/'training_final.pt',map_location='cpu',weights_only=True)
    selected=torch.load(path/'model.pt',map_location='cpu',weights_only=True)
    assert final['steps']==400
    assert all(torch.isfinite(v).all() for v in final['model'].values())
    assert all(torch.isfinite(v).all() for v in selected.values())
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(r['seed'])
        initial=core.TensorProductGraph([a['x_'+str(i)].shape[1] for i in range(len(a['node_counts']))],
            int(a['relation'].max())+1,p['core_parameters'],r['kind'])
    names=[n for n,v in initial.named_parameters() if v.requires_grad]
    frozen=[n for n,v in initial.named_parameters() if not v.requires_grad]
    assert final['optimizer_parameter_names']==names and final['frozen_parameter_names']==frozen
    assert r['parameters']==sum(v.numel() for v in initial.parameters())==174481
    assert r['trainable_parameters']==sum(v.numel() for v in initial.parameters() if v.requires_grad)
    assert r['fixed_parameters']==r['parameters']-r['trainable_parameters']
    assert r['trainable_parameters']=={'full':149825,'noncontextual_a':144785,'dense_qkv':148801}[r['kind']]
    for name in frozen:
        torch.testing.assert_close(final['model'][name],initial.state_dict()[name],atol=0.,rtol=0.)
        torch.testing.assert_close(selected[name],initial.state_dict()[name],atol=0.,rtol=0.)
    groups=final['optimizer']['param_groups']
    assert len(groups)==1 and groups[0]['lr']==r['lr'] and groups[0]['weight_decay']==p['weight_decay']
    indexes=groups[0]['params']
    assert len(names)==len(indexes)==len(final['optimizer']['state'])
    for name,index in zip(names,indexes):
        state=final['optimizer']['state'][index]
        assert int(state['step'])==400
        assert all(torch.isfinite(v).all() for v in state.values() if isinstance(v,torch.Tensor))
        assert state['exp_avg'].shape==state['exp_avg_sq'].shape==initial.state_dict()[name].shape
    return {key:final[key] for key in ('cpu_rng','cuda_rng')}


def run_stage(name):
    for task in ('MiDrug', 'LncDrug'):
        with (OUT / (name + '_' + task + '.log')).open('a', buffering=1) as stream:
            command = [sys.executable, '-u', '-m', 'improve.anchor_tensor_product_experiment_v1',
                name, '--out', str(OUT), '--task', task, '--device', 'cuda:1']
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError('Tensor-product worker failed: ' + str((name, task, completed.returncode)))


def classify(name):
    p, ph = experiment.common.protocol(OUT), base.digest(OUT / 'protocol.json')
    expected = {(task, kind, lr, p['screen_seed']) for task in p['tasks']
                for kind in p['kinds'] for lr in p['lrs']}
    if name == 'confirm':
        selection = base.read_json(OUT / 'SELECTION.json')
        expected |= {(task, kind, selection['chosen'][kind]['lr'], seed) for task in p['tasks']
                     for kind in p['kinds'] for seed in p['confirm_seeds']}
    found, neural, identity, rngs = set(), [], [], {}
    arrays = {task: experiment.common.load(OUT / 'inputs' / task / 'arrays.npz') for task in p['tasks']}
    for path in sorted((OUT / 'runs').glob('*/*/*/*/result.json')):
        r = experiment.common.verify(path.parent, ph)
        found.add((r['task'], r['kind'], r['lr'], r['seed']))
        assert not r.get('adopted_without_new_training', False)
        rng = audit_record(path.parent, r, arrays[r['task']], p)
        item = dict(path=str(path), sha256=base.digest(path), task=r['task'],
                    seed=r['seed'], kind=r['kind'], lr=r['lr'])
        if rng is not None:
            key = (r['lr'], r['seed'])
            if key in rngs:
                for direction in ('cpu_rng', 'cuda_rng'):
                    torch.testing.assert_close(rng[direction], rngs[key][direction], atol=0., rtol=0.)
            else:
                rngs[key] = rng
            neural.append(item)
        else:
            identity.append(item)
    count = 6 if name == 'screen' else 12
    if found != expected or len(neural) != count or len(identity) != count:
        raise ValueError('Missing or unexpected completed records')
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(), stage=name,
        family='tensor_product_attention', protocol_sha256=ph, cumulative_new_neural_fits=len(neural),
        cumulative_identity_scoring_records=len(identity), adopted_neural_fits=0,
        neural=neural, identity=identity, formal_neural_steps=400,
        identity_records_count_as_training=False, gradients_per_step=1,
        tensor_counts_final_optimizer_frozen_parameters_and_rng_verified=True, confirmed_final_innovations=0))
    path = HERE / ('TENSOR_PRODUCT_RECORD_CLASSIFICATION_' + name.upper() + '_V1.json')
    if path.exists():
        old = base.check_seal(base.read_json(path))
        assert old['neural'] == result['neural'] and old['identity'] == result['identity']
    else:
        base.write_json(path, result)
    return path


def main():
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG') != ':4096:8':
        raise ValueError('Declared deterministic CuBLAS environment required')
    torch.set_num_threads(4)
    frozen = base.check_seal(base.read_json(HERE / 'TENSOR_PRODUCT_QUEUE_PROTOCOL_V1.json'))
    assert frozen['family'] == 'tensor_product_attention' and frozen['source_sha256'] == base.digest(__file__)
    for path, digest in frozen['input_sha256'].items():
        assert base.digest(path) == digest, path
    experiment.common.protocol(OUT)
    destination = HERE / 'TENSOR_PRODUCT_QUEUE_RESULT_V1.json'
    if destination.exists():
        raise ValueError('Preserve completed queue result')
    for name in ('screen', 'confirm'):
        print('START_TENSOR_PRODUCT_STAGE', name, datetime.now(timezone.utc).isoformat(), flush=True)
        run_stage(name)
        classification = classify(name)
        with (OUT / (name + '_assessment.log')).open('a', buffering=1) as stream, redirect_stdout(stream):
            if name == 'screen' and not (OUT / 'SELECTION.json').exists():
                experiment.select(OUT)
            filename = 'SCREEN_ASSESSMENT.json' if name == 'screen' else 'HISTORICAL_ASSESSMENT.json'
            if (OUT / filename).exists():
                experiment.common.all_rows(OUT)
                result = base.check_seal(base.read_json(OUT / filename))
            else:
                result = audit.assess(OUT, name)
        print('TENSOR_PRODUCT_GATES', name, result['performance_gate'],
              result['primary_mechanism_gate'], result['offset_null_gate'], flush=True)
        if not result['declared_gate']:
            stop = OUT / ('SCREEN_STOP_DECISION.json' if name == 'screen' else 'HISTORICAL_STOP_DECISION.json')
            if not stop.exists():
                base.write_json(stop, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                    decision='Stop this frozen version before further confirmation, partition or outer fitting',
                    stage=name, primary_kind=result['primary_kind'], assessment_sha256=base.digest(OUT / filename),
                    performance_gate=result['performance_gate'], mechanism_gate=result['primary_mechanism_gate'],
                    offset_gate=result['offset_null_gate'], confirmed_innovations=0, default_replaced=False)))
            break
    counts = base.check_seal(base.read_json(classification))
    base.write_json(destination, base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        queue_protocol_sha256=frozen['sha256'], family='tensor_product_attention', stage=name,
        historical_gate=result['declared_gate'], assessment_sha256=base.digest(OUT / filename),
        classification_sha256=base.digest(classification), new_neural_fits=counts['cumulative_new_neural_fits'],
        identity_scoring_records=counts['cumulative_identity_scoring_records'], adopted_neural_fits=0,
        partition_verification_pending=bool(result['declared_gate']), confirmed_final_innovations=0,
        existing_default_replaced=False)))
    print('TENSOR_PRODUCT_QUEUE_COMPLETE', name, result['declared_gate'], flush=True)


if __name__ == '__main__':
    main()
