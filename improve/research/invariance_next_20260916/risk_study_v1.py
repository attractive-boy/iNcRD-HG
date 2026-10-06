"""Versioned training-only checks, prospective freeze and gated RDM execution."""
import argparse
from contextlib import redirect_stdout
from datetime import datetime, timezone
import gc
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

from improve import anchor_risk_distribution_v1 as experiment
from improve import risk_distribution_graph_v1 as core
from improve import assess_declared_graph as audit
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
ROOT = experiment.ROOT
FAMILY = 'risk_distribution'
STUDY = ROOT/'results/anchor_priority_risk_distribution_v1_20260916'
CORE = ROOT/'results/risk_distribution_core_v1_20260916'
SMOKE = ROOT/'results/risk_distribution_smoke_v1_20260916'


def now():
    return datetime.now(timezone.utc).isoformat()


def verified(path):
    obj = base.check_seal(base.read_json(path))
    for name,digest in obj.get('source_sha256',{}).items():
        if base.digest(name) != digest:
            raise ValueError('Checked source changed: '+name)
    return obj


def check_training_record(state,trace,detail,steps,warmup,kind):
    if (detail['training_steps'] != steps or detail['gradient_evaluations'] != steps
            or detail['optimizer_steps'] != steps or detail['matching_active_steps'] != max(0,steps-warmup)
            or len(trace) != steps or state['epoch'] != steps):
        raise ValueError('Incorrect training budget')
    if not state['optimizer']['state']:
        raise ValueError('Missing optimizer states')
    for item in state['optimizer']['state'].values():
        if int(item['step']) != steps or any(not torch.isfinite(v).all() for v in item.values() if isinstance(v,torch.Tensor)):
            raise ValueError('Invalid optimizer state')
    for step,row in enumerate(trace,1):
        if (row['epoch'] != step or row['omitted_fold'] != (step-1)%5
                or row['optimizer_steps'] != step or row['gradient_evaluations'] != step
                or row['matching_weight'] != (1. if step > warmup else 0.)
                or row['checkpoint_eligible'] != (step > warmup)):
            raise ValueError('Incorrect warmup/gradient trace')
        if (any(not np.isfinite(row[key]) for key in ('loss','bce','distribution_penalty','moment_penalty','effective_penalty'))
                or row['distribution_penalty'] < -1e-9
                or abs(row['loss']-row['bce']-row['effective_penalty']) > 1e-10):
            raise ValueError('Invalid empirical risk objective')
        if (kind == 'erm' or step <= warmup) and row['effective_penalty'] != 0.:
            raise ValueError('Unregistered warmup or ERM penalty')
        for group in row['domain_info']:
            if len(group['domains']) < 2 or min(group['counts']) < 2:
                raise ValueError('Insufficient actual domain counts')


def smoke():
    target = HERE/'RISK_DISTRIBUTION_SMOKE_V1.json'
    if target.exists() or SMOKE.exists():
        raise ValueError('Preserve short-run artifacts')
    unit = verified(HERE/'RISK_DISTRIBUTION_UNIT_V1.json')
    if not unit['passed'] or unit['tests_run'] != 5:
        raise ValueError('Core unit checks incomplete')
    SMOKE.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    c = core.parameters()
    p = dict(smoke=True,family=FAMILY,implementation_module='risk_distribution_graph_v1',
        source_sha256={'risk_distribution_graph_v1.py':base.digest(core.__file__)},
        core_parameters=c,kinds=list(core.KINDS),epochs=105,check_interval=20,weight_decay=.001,strong_weight=.1)
    records,fold_audits,inputs = [],[],{}
    for task in ('MiDrug','LncDrug'):
        source = ROOT/'results/anchor_residual_energy_v2_20260916/inputs'/task/'arrays.npz'
        inputs[str(source)] = base.digest(source)
        a = experiment.nested.load_selected(source,experiment.TRAIN_KEYS)
        query = torch.as_tensor(a['pairs_train'][:,:97]+a['pair_offset'][:,None],device='cuda:1')
        for kind in core.KINDS:
            torch.cuda.reset_peak_memory_stats('cuda:1')
            model,_,trace,detail = experiment.fit(a,p,kind,.001,986,'cuda:1',fixed_epochs=105)
            rng_hash = hashlib.sha256(torch.cuda.get_rng_state('cuda:1').numpy().tobytes()).hexdigest()
            dest = SMOKE/task/kind
            dest.mkdir(parents=True)
            torch.save(model.state_dict(),dest/'model.pt')
            torch.save(model.training_final_state,dest/'training_final.pt')
            base.write_json(dest/'trace.json',trace)
            loaded,xs,graphs,_ = experiment.setup(a,p,kind,'cuda:1')
            loaded.load_state_dict(torch.load(dest/'model.pt',map_location='cuda:1',weights_only=True))
            value = experiment.predict(model,xs,graphs,query,p).cpu().numpy()
            reverse = experiment.predict(loaded,xs,graphs,query.flip(1),p).cpu().numpy()[::-1]
            chunks = np.concatenate([experiment.predict(loaded,xs,graphs,query[:,i:i+50],p).cpu().numpy()
                                     for i in range(0,query.shape[1],50)])
            for actual in (reverse,chunks):
                np.testing.assert_allclose(actual,value,atol=3e-5,rtol=3e-5)
            if task == 'MiDrug':
                check_training_record(model.training_final_state,trace,detail,105,c['warmup_steps'],kind)
                if any('anchor_composed' in row for row in trace):
                    raise ValueError('Short run consulted evaluation metrics')
                domain_sha = hashlib.sha256(model.drug_domains.cpu().numpy().tobytes()).hexdigest()
            else:
                np.testing.assert_array_equal(value,np.zeros_like(value))
                if trace or detail['trained_neural_model'] or model.training_final_state is not None:
                    raise ValueError('Identity unexpectedly trained')
                domain_sha = None
            record = base.seal(dict(task=task,kind=kind,**detail,dropout_rng_sha256=rng_hash,
                training_domain_sha256=domain_sha,replay_max_abs=float(max(abs(value-reverse).max(),abs(value-chunks).max())),
                peak_allocated_gib=torch.cuda.max_memory_allocated('cuda:1')/2**30,
                files={name:base.digest(dest/name) for name in ('model.pt','training_final.pt','trace.json')}))
            base.write_json(dest/'RESULT.json',record)
            records.append(record)
            print('RDM_SHORT_RECORD',task,kind,detail['training_steps'],record['replay_max_abs'],record['peak_allocated_gib'],flush=True)
            del model,loaded,xs,graphs
            gc.collect()
            torch.cuda.empty_cache()
        sources = [('historical',source)]+[(split,ROOT/'results/mechanism_validation_20260915/inputs'/task/str(split)/'arrays.npz')
                                         for split in (61453,7919,104729)]
        for split,path in sources:
            inputs[str(path)] = base.digest(path)
            training = experiment.nested.load_selected(path,experiment.nested.TRAIN_KEYS)
            model,_,graphs,folds = core.setup(training,c,'full','cpu')
            for fold in range(5):
                ids = folds[fold].numpy()
                if task == 'LncDrug':
                    if graphs[fold] is not None or list(model.parameters()):
                        raise ValueError('Identity constructed a graph')
                    fold_audits.append(dict(task=task,split=split,fold=fold,identity=True,new_graph_constructed=False))
                    continue
                pairs = training['pairs_train'][:,ids]+training['pair_offset'][:,None]
                physical = set(map(tuple,graphs[fold][0].numpy().T))
                overlap = sum(tuple(pair) in physical or tuple(pair[::-1]) in physical for pair in pairs.T)
                if overlap:
                    raise ValueError('Query edge visible in support graph')
                changed = {key:value.copy() for key,value in training.items()}
                changed['y_train'][ids] = 1-changed['y_train'][ids]
                other,_,other_graphs,_ = core.setup(changed,c,'full','cpu')
                torch.testing.assert_close(model.drug_domains,other.drug_domains,atol=0,rtol=0)
                for first,second in zip(graphs[fold],other_graphs[fold]):
                    torch.testing.assert_close(first,second,atol=0,rtol=0)
                fold_audits.append(dict(task=task,split=split,fold=fold,identity=False,forbidden_overlap=0,
                                       query_labels_invariant=True,domain_assignments_outcome_invariant=True))
    neural = [row for row in records if row['task']=='MiDrug']
    if len({r['dropout_rng_sha256'] for r in neural}) != 1 or len({r['training_domain_sha256'] for r in neural}) != 1:
        raise ValueError('Arms have different random streams or domains')
    sources = [Path(__file__),Path(core.__file__),Path(experiment.__file__),ROOT/'sharpness_graph_v1.py',
               ROOT/'overlap_block_models.py',HERE/'RISK_DISTRIBUTION_UNIT_V1.json']
    result = base.seal(dict(created_at_utc=now(),device='cuda:1',core_parameters=c,
        new_short_neural_fits=3,identity_replays=3,formal_fits=0,performance_metrics_computed=False,
        records=records,fold_audits=fold_audits,input_sha256=inputs,
        source_sha256={str(path):base.digest(path) for path in sources},
        original_test_source_access=False,confirmed_final_innovations=0))
    base.write_json(target,result)
    print('RDM_SMOKE_COMPLETE',result['sha256'],flush=True)


def freeze():
    target = HERE/'RISK_DISTRIBUTION_QUEUE_PROTOCOL_V1.json'
    if target.exists() or CORE.exists() or STUDY.exists():
        raise ValueError('Preserve frozen study')
    unit = verified(HERE/'RISK_DISTRIBUTION_UNIT_V1.json')
    smoke_result = verified(HERE/'RISK_DISTRIBUTION_SMOKE_V1.json')
    if (not unit['passed'] or unit['tests_run'] != 5 or smoke_result['new_short_neural_fits'] != 3
            or smoke_result['identity_replays'] != 3 or smoke_result['formal_fits']
            or smoke_result['performance_metrics_computed'] or len(smoke_result['fold_audits']) != 40
            or smoke_result['core_parameters'] != core.parameters()):
        raise ValueError('Implementation evidence incomplete')
    for path,digest in smoke_result['input_sha256'].items():
        if base.digest(path) != digest:
            raise ValueError('Audited input changed')
    method = base.read_json(HERE/'RDM_METHOD_SOURCE_V1.json')
    for row in method['records']:
        if not row['git_blob_verified'] or base.digest(row['file']) != row['sha256']:
            raise ValueError('Author source verification incomplete')
    files = [Path(__file__),HERE/'RISK_DISTRIBUTION_UNIT_V1.json',HERE/'RISK_DISTRIBUTION_SMOKE_V1.json',
             HERE/'RDM_ADAPTATION_V1.md',HERE/'RISK_EXECUTOR_DERIVATION_V1.json',
             HERE/'RDM_SOURCE_FETCH_V2.json',HERE/'RDM_SOURCE_FETCH_V3.json',
             HERE/'RDM_AUTHOR_SOURCE_V1.json',HERE/'RDM_METHOD_SOURCE_V1.json',
             HERE/'rdm_arxiv_2310.18598v1_retry.pdf',HERE/'rdm_arxiv_2310.18598v1_retry.txt',
             HERE/'rdm_crossref_exact_v1.json',HERE/'rdm_cvf_landing_v1.html',HERE/'rdm_author_README.md']
    files += [Path(row['file']) for row in method['records']]
    names = tuple(dict.fromkeys(experiment.OWN+('tabular_foundation.py','consolidate_declared_history_v1.py')))
    p = dict(created_at_utc=now(),family=FAMILY,implementation_module='risk_distribution_graph_v1',
        tasks=['MiDrug','LncDrug'],kinds=list(core.KINDS),primary_kind='full',
        primary_controls=['moment_matching','erm'],core_parameters=core.parameters(),
        lnc_policy='identity_strong_reference',identity_records_are_not_neural_fits=True,
        selection='Select only checkpoints after 100 warmup updates: steps 120 through 400 every 20 updates, by mean global ROC AUC and sklearn AP of reference plus 0.1 residual. Then one shared learning rate per arm among 0.001 and 0.003 subject to primary MiDrug improvement and LncDrug tolerance, then mean MiDrug gain, then smaller learning rate.',
        literature=dict(title='Domain Generalisation via Risk Distribution Matching',doi='10.1109/WACV57701.2024.00277',
            venue='WACV 2024',publisher_verified=True,fulltext='arXiv:2310.18598v1',
            author_commit=method['commit'],author_code_read=True,author_code_executed=False),
        mechanism_family='class_conditional_scalar_risk_distribution_alignment',
        local_adaptation='Global nested-reference BCE; training-drug attribute domains; label-conditional matching to equal-domain empirical mixtures; exact weighted V-statistic; summed eight RBFs; 100 warmup steps within a 400-step constant-lr AdamW budget without reset; no extra variance penalty; prediction domains unused. Not an exact author reproduction or a causal guarantee.',
        source_sha256={name:base.digest(ROOT/name) for name in names},
        inputs={str(path):base.digest(path) for path in files},no_new_performance_observed=True,
        smoke=False,original_test_source_access=False,confirmed_innovations=0,existing_default_replaced=False)
    CORE.mkdir(parents=True)
    (CORE/'source_snapshot').mkdir()
    for name in names:
        shutil.copy2(ROOT/name,CORE/'source_snapshot'/name)
    base.write_json(CORE/'protocol.json',p)
    experiment.prepare(STUDY,FAMILY,ROOT/'results/nested_anchor_v2_20260915/historical',CORE)
    frozen = [CORE/'protocol.json',STUDY/'protocol.json',HERE/'RDM_ADAPTATION_V1.md',HERE/'RISK_DISTRIBUTION_SMOKE_V1.json']
    queue = base.seal(dict(created_at_utc=now(),family=FAMILY,source_sha256=base.digest(__file__),
        input_sha256={str(path):base.digest(path) for path in frozen},device='cuda:1',
        stages=['screen','confirm'],stop_on_any_failed_gate=True,adopted_neural_fits=0,
        expected_new_neural_fits={'screen':6,'confirm_if_eligible':6},
        expected_identity_records={'screen':6,'confirm_if_eligible':6},confirmed_final_innovations=0))
    base.write_json(target,queue)
    print('RDM_FROZEN',base.digest(STUDY/'protocol.json'),queue['sha256'],flush=True)


def classify(stage):
    p = experiment.common.protocol(STUDY)
    ph = base.digest(STUDY/'protocol.json')
    expected = {(task,kind,lr,p['screen_seed']) for task in p['tasks'] for kind in p['kinds'] for lr in p['lrs']}
    if stage == 'confirm':
        selection = base.read_json(STUDY/'SELECTION.json')
        expected |= {(task,kind,selection['chosen'][kind]['lr'],seed) for task in p['tasks']
                     for kind in p['kinds'] for seed in p['confirm_seeds']}
    found,neural,identity = set(),[],[]
    for path in sorted((STUDY/'runs').glob('*/*/*/*/result.json')):
        r = experiment.common.verify(path.parent,ph)
        found.add((r['task'],r['kind'],r['lr'],r['seed']))
        if r.get('adopted_without_new_training',False):
            raise ValueError('No reuse registered')
        trace = base.read_json(path.parent/'trace.json')
        state = torch.load(path.parent/'training_final.pt',map_location='cpu',weights_only=True)
        item = dict(path=str(path),sha256=base.digest(path),task=r['task'],kind=r['kind'],lr=r['lr'],seed=r['seed'])
        if r['trained_neural_model']:
            if r['task'] != 'MiDrug' or r['epoch'] not in range(120,401,20):
                raise ValueError('Invalid neural task or selected checkpoint')
            check_training_record(state,trace,r,400,100,r['kind'])
            if [row['epoch'] for row in trace if 'anchor_composed' in row] != list(range(120,401,20)):
                raise ValueError('Changed checkpoint-selection budget')
            neural.append(item)
        else:
            if r['task'] != 'LncDrug' or state is not None or trace or any(r[key] for key in
                    ('epoch','parameters','trainable_parameters','training_steps','gradient_evaluations','optimizer_steps','matching_active_steps')):
                raise ValueError('Invalid identity record')
            z = experiment.common.load(path.parent/'scores.npz')
            np.testing.assert_array_equal(z['standalone'],np.zeros_like(z['standalone']))
            np.testing.assert_array_equal(z['anchor_composed'],z['anchor'])
            identity.append(item)
    count = len(p['kinds'])*(len(p['lrs'])+(len(p['confirm_seeds']) if stage=='confirm' else 0))
    if found != expected or len(neural) != count or len(identity) != count:
        raise ValueError('Incorrect record set')
    result = base.seal(dict(created_at_utc=now(),family=FAMILY,stage=stage,protocol_sha256=ph,
        cumulative_new_neural_fits=len(neural),cumulative_identity_scoring_records=len(identity),
        neural=neural,identity=identity,adopted_neural_fits=0,identity_records_count_as_training=False,
        optimizer_updates=400,gradient_evaluations=400,warmup_updates=100,confirmed_final_innovations=0))
    target = HERE/('RISK_DISTRIBUTION_RECORD_CLASSIFICATION_'+stage.upper()+'_V1.json')
    if target.exists():
        old = base.check_seal(base.read_json(target))
        if old['neural'] != neural or old['identity'] != identity:
            raise ValueError('Existing classification disagrees')
    else:
        base.write_json(target,result)
    return target


def run():
    frozen = base.check_seal(base.read_json(HERE/'RISK_DISTRIBUTION_QUEUE_PROTOCOL_V1.json'))
    target = HERE/'RISK_DISTRIBUTION_QUEUE_RESULT_V1.json'
    if target.exists() or frozen['source_sha256'] != base.digest(__file__):
        raise ValueError('Completed queue or changed source')
    for path,digest in frozen['input_sha256'].items():
        if base.digest(path) != digest:
            raise ValueError('Frozen input changed')
    for stage in ('screen','confirm'):
        print('START_RDM_STAGE',stage,now(),flush=True)
        for task in ('LncDrug','MiDrug'):
            with (STUDY/(stage+'_'+task+'.log')).open('a',buffering=1) as stream:
                command = [sys.executable,'-u','-m','improve.anchor_risk_distribution_v1',stage,
                           '--family',FAMILY,'--out',str(STUDY),'--task',task,'--device','cuda:1']
                subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
        classification = classify(stage)
        with (STUDY/(stage+'_assessment.log')).open('a',buffering=1) as stream,redirect_stdout(stream):
            if stage == 'screen' and not (STUDY/'SELECTION.json').exists():
                experiment.select(STUDY)
            filename = 'SCREEN_ASSESSMENT.json' if stage=='screen' else 'HISTORICAL_ASSESSMENT.json'
            if (STUDY/filename).exists():
                experiment.common.all_rows(STUDY)
                assessment = base.check_seal(base.read_json(STUDY/filename))
            else:
                assessment = audit.assess(STUDY,stage)
        print('RDM_GATES',stage,assessment['performance_gate'],assessment['primary_mechanism_gate'],assessment['offset_null_gate'],flush=True)
        if not assessment['declared_gate']:
            stop = STUDY/(stage.upper()+'_STOP_DECISION.json')
            if not stop.exists():
                base.write_json(stop,base.seal(dict(created_at_utc=now(),stage=stage,
                    decision='Stop this frozen version before further fitting',
                    assessment_sha256=base.digest(STUDY/filename),confirmed_innovations=0,default_replaced=False)))
            break
    counts = base.check_seal(base.read_json(classification))
    base.write_json(target,base.seal(dict(created_at_utc=now(),family=FAMILY,stage=stage,
        queue_protocol_sha256=frozen['sha256'],historical_gate=assessment['declared_gate'],
        assessment_sha256=base.digest(STUDY/filename),classification_sha256=base.digest(classification),
        new_neural_fits=counts['cumulative_new_neural_fits'],identity_scoring_records=counts['cumulative_identity_scoring_records'],
        adopted_neural_fits=0,partition_verification_pending=bool(assessment['declared_gate']),
        confirmed_final_innovations=0,existing_default_replaced=False)))
    print('RDM_QUEUE_COMPLETE',stage,assessment['declared_gate'],flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('smoke','freeze','run'))
    args = parser.parse_args()
    globals()[args.command]()
