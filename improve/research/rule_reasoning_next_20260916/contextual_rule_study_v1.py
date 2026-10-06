"""Training-only audits, prospective freeze and gated contextual-rule execution."""
import argparse
from contextlib import redirect_stdout
from datetime import datetime,timezone
import gc
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

from improve import anchor_contextual_rule_v1 as experiment
from improve import contextual_rule_graph_v1 as core
from improve import assess_declared_graph as audit
from improve import tabular_foundation as base

HERE = Path(__file__).resolve().parent
ROOT = experiment.ROOT
FAMILY = 'contextual_rule'
STUDY = ROOT/'results/anchor_priority_contextual_rule_v1_20260916'
CORE = ROOT/'results/contextual_rule_core_v1_20260916'
SMOKE = ROOT/'results/contextual_rule_smoke_v1_20260916'


def now():
    return datetime.now(timezone.utc).isoformat()


def verified(path):
    value = base.check_seal(base.read_json(path))
    for source,digest in value['source_sha256'].items():
        if base.digest(source)!=digest:
            raise ValueError('Checked source changed')
    return value


def check_record(state,trace,detail,steps,triplets):
    if (len(trace)!=steps or state['epoch']!=steps or len(state['support_audits'])!=5
            or any(detail[key]!=steps for key in ('training_steps','gradient_evaluations','optimizer_steps'))
            or detail['rule_triplets_per_update']!=triplets):
        raise ValueError('Incorrect rule-model training budget')
    if not state['optimizer']['state']:
        raise ValueError('Missing optimizer state')
    for item in state['optimizer']['state'].values():
        if int(item['step'])!=steps or any(not torch.isfinite(value).all() for value in item.values() if isinstance(value,torch.Tensor)):
            raise ValueError('Invalid optimizer state')
    for step,row in enumerate(trace,1):
        if (row['epoch']!=step or row['omitted_fold']!=(step-1)%5 or row['triplet_count']!=triplets
                or row['optimizer_steps']!=step or row['gradient_evaluations']!=step or not row['rule_supervision_verified']
                or row['eligible_positive_queries']<1 or row['maximum_absolute_residual']>1.000001
                or any(not np.isfinite(row[key]) for key in ('loss','bce','ranking_loss','gradient_norm'))
                or abs(row['loss']-row['bce']-.1*row['ranking_loss'])>1e-7):
            raise ValueError('Invalid rule-supervision trace')


def smoke():
    target = HERE/'CONTEXTUAL_RULE_SMOKE_V1.json'
    if target.exists() or SMOKE.exists():
        raise ValueError('Preserve short-run artifacts')
    unit = verified(HERE/'CONTEXTUAL_RULE_UNIT_V2.json')
    if not unit['passed'] or unit['tests_run']!=4:
        raise ValueError('Unit checks incomplete')
    SMOKE.mkdir(parents=True)
    torch.set_num_threads(4)
    torch.cuda.set_device('cuda:1')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    c = core.parameters()
    p = dict(smoke=True,family=FAMILY,implementation_module='contextual_rule_graph_v1',
        source_sha256={'contextual_rule_graph_v1.py':base.digest(core.__file__)},
        core_parameters=c,kinds=list(core.KINDS),epochs=10,check_interval=20,
        weight_decay=.001,strong_weight=.1,private_rng_offset=29015)
    records,fold_audits,inputs = [],[],{}
    for task in ('MiDrug','LncDrug'):
        source = ROOT/'results/anchor_residual_energy_v2_20260916/inputs'/task/'arrays.npz'
        inputs[str(source)] = base.digest(source)
        a = experiment.nested.load_selected(source,experiment.TRAIN_KEYS)
        query = torch.as_tensor(a['pairs_train'][:,:97]+a['pair_offset'][:,None],device='cuda:1')
        for kind in core.KINDS:
            torch.cuda.reset_peak_memory_stats('cuda:1')
            model,_,trace,detail = experiment.fit(a,p,kind,.001,986,'cuda:1',fixed_epochs=10)
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
                np.testing.assert_allclose(value,actual,atol=3e-5,rtol=3e-5)
            if task=='MiDrug':
                check_record(model.training_final_state,trace,detail,10,c['triplet_batch'])
                if any('anchor_composed' in row for row in trace):
                    raise ValueError('Short fit consulted validation performance')
                private_hash = hashlib.sha256(model.training_final_state['private_rng'].numpy().tobytes()).hexdigest()
                if model.training_final_state['support_audits']!=[graph['audit'] for graph in graphs]:
                    raise ValueError('Reconstructed symbolic supports differ')
            else:
                np.testing.assert_array_equal(value,np.zeros_like(value))
                if trace or detail['trained_neural_model'] or model.training_final_state is not None:
                    raise ValueError('Identity unexpectedly trained')
                private_hash = None
            record = base.seal(dict(task=task,kind=kind,**detail,dropout_rng_sha256=rng_hash,
                private_rng_sha256=private_hash,replay_max_abs=float(max(abs(value-reverse).max(),abs(value-chunks).max())),
                peak_allocated_gib=torch.cuda.max_memory_allocated('cuda:1')/2**30,
                files={name:base.digest(dest/name) for name in ('model.pt','training_final.pt','trace.json')}))
            base.write_json(dest/'RESULT.json',record)
            records.append(record)
            print('CONTEXTUAL_RULE_SHORT',task,kind,record['replay_max_abs'],record['peak_allocated_gib'],flush=True)
            del model,loaded,xs,graphs
            gc.collect();torch.cuda.empty_cache()
        paths = [('historical',source)]+[(split,ROOT/'results/mechanism_validation_20260915/inputs'/task/str(split)/'arrays.npz')
                                       for split in (61453,7919,104729)]
        for split,path in paths:
            inputs[str(path)] = base.digest(path)
            training = experiment.nested.load_selected(path,experiment.nested.TRAIN_KEYS)
            for fold in range(5):
                if core.identity_task(training,c):
                    model,_,graphs,_ = core.setup(training,c,'full','cpu')
                    if list(model.parameters()) or graphs[fold] is not None:
                        raise ValueError('Identity constructed a graph')
                    fold_audits.append(dict(task=task,split=split,fold=fold,identity=True,new_graph_constructed=False))
                    continue
                graph = core.build_support(training,c,fold,'cpu')
                selected = training['folds']==fold
                changed = {key:value.copy() for key,value in training.items()}
                changed['y_train'][selected] = 1-changed['y_train'][selected]
                other = core.build_support(changed,c,fold,'cpu')
                if graph['audit']!=other['audit']:
                    raise ValueError('Query outcomes affected symbolic/context inputs')
                matrices,_,_ = core.relation_matrices(training,c,fold)
                pairs = training['pairs_train'][:,selected]
                overlap = int(np.count_nonzero(np.asarray(matrices[3][pairs[0],pairs[1]])))
                if overlap:
                    raise ValueError('Query is visible in its support')
                pool = core.rule_pool(training,graph,fold,'cpu')
                q = pool['queries'].numpy()
                if not np.all(training['folds'][q]==fold) or not np.all(training['y_train'][q]==1):
                    raise ValueError('Rule-supervision pool crosses training folds')
                fold_audits.append(dict(task=task,split=split,fold=fold,identity=False,forbidden_overlap=0,
                    paths_statistics_context_outcome_invariant=True,eligible_positive_queries=len(q),support_audit=graph['audit']))
    neural = [row for row in records if row['task']=='MiDrug']
    if len({r['dropout_rng_sha256'] for r in neural})!=1 or len({r['private_rng_sha256'] for r in neural})!=1:
        raise ValueError('Arms consume different random streams')
    sources = [Path(__file__),Path(core.__file__),Path(experiment.__file__),ROOT/'sharpness_graph_v1.py',
               ROOT/'overlap_block_models.py',HERE/'CONTEXTUAL_RULE_UNIT_V2.json']
    result = base.seal(dict(created_at_utc=now(),device='cuda:1',core_parameters=c,new_short_neural_fits=2,
        identity_replays=2,formal_fits=0,performance_metrics_computed=False,records=records,fold_audits=fold_audits,
        input_sha256=inputs,source_sha256={str(path):base.digest(path) for path in sources},
        original_test_source_access=False,confirmed_final_innovations=0))
    base.write_json(target,result)
    print('CONTEXTUAL_RULE_SMOKE_COMPLETE',result['sha256'],flush=True)


def freeze():
    target = HERE/'CONTEXTUAL_RULE_QUEUE_PROTOCOL_V1.json'
    if target.exists() or CORE.exists() or STUDY.exists():
        raise ValueError('Preserve study artifacts')
    unit,short = verified(HERE/'CONTEXTUAL_RULE_UNIT_V2.json'),verified(HERE/'CONTEXTUAL_RULE_SMOKE_V1.json')
    if (not unit['passed'] or unit['tests_run']!=4 or short['new_short_neural_fits']!=2 or short['identity_replays']!=2
            or short['formal_fits'] or short['performance_metrics_computed'] or len(short['fold_audits'])!=40
            or short['core_parameters']!=core.parameters()):
        raise ValueError('Implementation evidence incomplete')
    for path,digest in short['input_sha256'].items():
        if base.digest(path)!=digest:
            raise ValueError('Audited input changed')
    files = [Path(__file__),HERE/'CONTEXTUAL_RULE_UNIT_V1.json',HERE/'CONTEXTUAL_RULE_UNIT_V2.json',
        HERE/'CONTEXTUAL_RULE_SMOKE_V1.json',HERE/'CONTEXTUAL_RULE_SUPPORT_PREFLIGHT_V1.json',
        HERE/'SLOGIC_ADAPTATION_V1.md',HERE/'CONTEXTUAL_RULE_EXECUTOR_DERIVATION_V1.json',
        HERE/'SLOGIC_SOURCE_FETCH_V1.json',HERE/'SLOGIC_SOURCE_FETCH_V2.json',
        HERE/'slogic_arxiv_2510.00279v3_retry.pdf',HERE/'slogic_arxiv_2510.00279v3_retry.txt',
        HERE/'slogic_crossref_title_v1.json',HERE/'slogic_arxiv_metadata_v1.xml']
    for name,key in [('SLOGIC_AUTHOR_SOURCE_V1.json','files'),('SLOGIC_METHOD_SOURCE_V1.json','records'),('SLOGIC_AGGREGATION_SOURCE_V1.json','records')]:
        manifest = base.read_json(HERE/name)
        files.append(HERE/name)
        for row in manifest[key]:
            if (not row['git_blob_verified'] or row['commit']!='c19e05d9368c109611d92932b922fa5447ec0a52'
                    or base.digest(row['file'])!=row['sha256']):
                raise ValueError('Author source changed')
            files.append(Path(row['file']))
    names = tuple(dict.fromkeys(experiment.OWN+('test_contextual_rule_v2.py','tabular_foundation.py','consolidate_declared_history_v1.py')))
    p = dict(created_at_utc=now(),family=FAMILY,implementation_module='contextual_rule_graph_v1',
        tasks=['MiDrug','LncDrug'],kinds=list(core.KINDS),primary_kind='full',primary_controls=['global_rules'],
        core_parameters=core.parameters(),lnc_policy='identity_strong_reference',identity_records_are_not_neural_fits=True,
        selection='Every 20 updates select the checkpoint by mean global ROC AUC and sklearn AP of reference plus 0.1 residual; choose shared learning rate per arm from 0.001/0.003 using MiDrug improvement, LncDrug tolerance, then mean MiDrug gain and smaller learning rate.',
        literature=dict(title='SLogic: Subgraph-Informed Logical Rule Learning for Knowledge Graph Completion',
            arxiv='2510.00279v3',author_claimed_venue='KDIR 2026',publisher_status_independently_verified=False,
            author_commit='c19e05d9368c109611d92932b922fa5447ec0a52',author_code_read=True,author_code_executed=False),
        mechanism_family='head_context_conditioned_logical_rule_scoring',
        local_adaptation='Whole-fold support deletion; exact length-two/three simple-path counts; all-pair body denominators; width-64 structural RGCN and packed GRU; centered bounded rule potential; nested-reference BCE plus 0.1 hard-rule margin loss; no length penalty; fixed LncDrug identity. Global control uses one shared mean graph context.',
        source_sha256={name:base.digest(ROOT/name) for name in names},inputs={str(path):base.digest(path) for path in files},
        no_new_performance_observed=True,smoke=False,original_test_source_access=False,confirmed_innovations=0,existing_default_replaced=False)
    CORE.mkdir(parents=True);(CORE/'source_snapshot').mkdir()
    for name in names:
        shutil.copy2(ROOT/name,CORE/'source_snapshot'/name)
    base.write_json(CORE/'protocol.json',p)
    experiment.prepare(STUDY,FAMILY,ROOT/'results/nested_anchor_v2_20260915/historical',CORE)
    frozen = [CORE/'protocol.json',STUDY/'protocol.json',HERE/'SLOGIC_ADAPTATION_V1.md',HERE/'CONTEXTUAL_RULE_SMOKE_V1.json']
    result = base.seal(dict(created_at_utc=now(),family=FAMILY,source_sha256=base.digest(__file__),
        input_sha256={str(path):base.digest(path) for path in frozen},device='cuda:1',stages=['screen','confirm'],
        stop_on_any_failed_gate=True,adopted_neural_fits=0,expected_new_neural_fits={'screen':4,'confirm_if_eligible':4},
        expected_identity_records={'screen':4,'confirm_if_eligible':4},confirmed_final_innovations=0))
    base.write_json(target,result)
    print('CONTEXTUAL_RULE_FROZEN',base.digest(STUDY/'protocol.json'),result['sha256'],flush=True)


def classify(stage):
    p = experiment.common.protocol(STUDY)
    ph = base.digest(STUDY/'protocol.json')
    expected = {(task,kind,lr,p['screen_seed']) for task in p['tasks'] for kind in p['kinds'] for lr in p['lrs']}
    if stage=='confirm':
        selection = base.read_json(STUDY/'SELECTION.json')
        expected |= {(task,kind,selection['chosen'][kind]['lr'],seed) for task in p['tasks'] for kind in p['kinds'] for seed in p['confirm_seeds']}
    found,neural,identity,supports = set(),[],[],None
    for path in sorted((STUDY/'runs').glob('*/*/*/*/result.json')):
        r = experiment.common.verify(path.parent,ph)
        found.add((r['task'],r['kind'],r['lr'],r['seed']))
        if r.get('adopted_without_new_training',False):
            raise ValueError('No reuse registered')
        trace = base.read_json(path.parent/'trace.json')
        state = torch.load(path.parent/'training_final.pt',map_location='cpu',weights_only=True)
        item = dict(path=str(path),sha256=base.digest(path),task=r['task'],kind=r['kind'],lr=r['lr'],seed=r['seed'])
        if r['trained_neural_model']:
            if r['task']!='MiDrug' or r['epoch'] not in range(20,401,20):
                raise ValueError('Incorrect neural task or selected checkpoint')
            check_record(state,trace,r,400,p['core_parameters']['triplet_batch'])
            if [row['epoch'] for row in trace if 'anchor_composed' in row]!=list(range(20,401,20)):
                raise ValueError('Changed checkpoint budget')
            if supports is not None and supports!=state['support_audits']:
                raise ValueError('Arms saw different symbolic supports')
            supports = state['support_audits']
            neural.append(item)
        else:
            if r['task']!='LncDrug' or state is not None or trace or any(r[key] for key in
                    ('epoch','parameters','trainable_parameters','training_steps','gradient_evaluations','optimizer_steps','rule_triplets_per_update')):
                raise ValueError('Invalid identity record')
            z = experiment.common.load(path.parent/'scores.npz')
            np.testing.assert_array_equal(z['standalone'],np.zeros_like(z['standalone']))
            np.testing.assert_array_equal(z['anchor_composed'],z['anchor'])
            identity.append(item)
    count = len(p['kinds'])*(len(p['lrs'])+(len(p['confirm_seeds']) if stage=='confirm' else 0))
    if found!=expected or len(neural)!=count or len(identity)!=count:
        raise ValueError('Incorrect record set')
    result = base.seal(dict(created_at_utc=now(),family=FAMILY,stage=stage,protocol_sha256=ph,
        cumulative_new_neural_fits=len(neural),cumulative_identity_scoring_records=len(identity),
        neural=neural,identity=identity,adopted_neural_fits=0,identity_records_count_as_training=False,
        optimizer_updates=400,gradient_evaluations=400,confirmed_final_innovations=0))
    target = HERE/('CONTEXTUAL_RULE_RECORD_CLASSIFICATION_'+stage.upper()+'_V1.json')
    if target.exists():
        old = base.check_seal(base.read_json(target))
        if old['neural']!=neural or old['identity']!=identity:
            raise ValueError('Existing classification disagrees')
    else:
        base.write_json(target,result)
    return target


def run():
    frozen = base.check_seal(base.read_json(HERE/'CONTEXTUAL_RULE_QUEUE_PROTOCOL_V1.json'))
    target = HERE/'CONTEXTUAL_RULE_QUEUE_RESULT_V1.json'
    if target.exists() or frozen['source_sha256']!=base.digest(__file__):
        raise ValueError('Completed queue or changed source')
    for path,digest in frozen['input_sha256'].items():
        if base.digest(path)!=digest:
            raise ValueError('Frozen input changed')
    for stage in ('screen','confirm'):
        print('START_CONTEXTUAL_RULE_STAGE',stage,now(),flush=True)
        for task in ('LncDrug','MiDrug'):
            with (STUDY/(stage+'_'+task+'.log')).open('a',buffering=1) as stream:
                command = [sys.executable,'-u','-m','improve.anchor_contextual_rule_v1',stage,'--family',FAMILY,
                           '--out',str(STUDY),'--task',task,'--device','cuda:1']
                subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
        classification = classify(stage)
        with (STUDY/(stage+'_assessment.log')).open('a',buffering=1) as stream,redirect_stdout(stream):
            if stage=='screen' and not (STUDY/'SELECTION.json').exists():
                experiment.select(STUDY)
            filename = 'SCREEN_ASSESSMENT.json' if stage=='screen' else 'HISTORICAL_ASSESSMENT.json'
            if (STUDY/filename).exists():
                experiment.common.all_rows(STUDY)
                assessment = base.check_seal(base.read_json(STUDY/filename))
            else:
                assessment = audit.assess(STUDY,stage)
        print('CONTEXTUAL_RULE_GATES',stage,assessment['performance_gate'],assessment['primary_mechanism_gate'],assessment['offset_null_gate'],flush=True)
        if not assessment['declared_gate']:
            stop = STUDY/(stage.upper()+'_STOP_DECISION.json')
            if not stop.exists():
                base.write_json(stop,base.seal(dict(created_at_utc=now(),stage=stage,
                    decision='Stop this frozen version before further fitting',assessment_sha256=base.digest(STUDY/filename),
                    confirmed_innovations=0,default_replaced=False)))
            break
    counts = base.check_seal(base.read_json(classification))
    base.write_json(target,base.seal(dict(created_at_utc=now(),family=FAMILY,stage=stage,queue_protocol_sha256=frozen['sha256'],
        historical_gate=assessment['declared_gate'],assessment_sha256=base.digest(STUDY/filename),
        classification_sha256=base.digest(classification),new_neural_fits=counts['cumulative_new_neural_fits'],
        identity_scoring_records=counts['cumulative_identity_scoring_records'],adopted_neural_fits=0,
        partition_verification_pending=bool(assessment['declared_gate']),confirmed_final_innovations=0,existing_default_replaced=False)))
    print('CONTEXTUAL_RULE_QUEUE_COMPLETE',stage,assessment['declared_gate'],flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('smoke','freeze','run'))
    args = parser.parse_args()
    globals()[args.command]()
