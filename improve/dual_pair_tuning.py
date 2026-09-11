"""Bounded hyperparameter rescue of the existing dual-pair architecture.

Screen 12 settings for the baseline and candidate, confirm the top three with
two more initialization seeds, then freeze one setting/epoch using a smoothed
three-seed inner-validation curve. Outer labels never enter selection.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from improve import dual_pair as previous
from improve import pair_exchange as parent
from improve.dual_pair_modules import DualPairModel
from improve.expression_consensus import TASKS, make_model
from improve.expression_scale_mechanism import state_digest
from improve.first_round_features import json_digest, sha256, write_json
from improve.network_feature_ablation import score
from improve.pair_exchange_modules import remove_target_pairs, sample_queries


SEARCH_ARMS = ('pca_query', 'dual_pair')
REFIT_ARMS = ('pca_query', 'dual_previous', 'matched_local', 'matched_transport', 'dual_pair')
DEFAULT_STEP = 1. / (1. + math.exp(1.))
OWN_FILES = ('dual_pair_tuning.py', 'test_dual_pair_tuning.py')


def setting(identifier, lr, wd=0., branch_lr=1., scale=1., step=DEFAULT_STEP):
    return dict(id=identifier, lr_multiplier=lr, weight_decay=wd,
                branch_lr_multiplier=branch_lr, correction_scale=scale, step_init=step)


def configuration_grid(smoke=False):
    if smoke:
        return {arm: [setting('smoke', .3, 1e-4, 3., .3, .5)] for arm in SEARCH_ARMS}
    baseline = [setting('b%02d' % i, lr, wd) for i, (lr, wd) in enumerate(
        (lr, wd) for lr in (.1, .3, 1., 3.) for wd in (0., 1e-4, 1e-3))]
    recipes = [(1., 0., 1., DEFAULT_STEP), (.3, 1e-4, .3, .1),
               (3., 1e-4, 1., .5), (10., 1e-3, .3, DEFAULT_STEP)]
    dual = [setting('d%02d' % i, lr, wd, branch, scale, step)
            for i, (lr, (branch, wd, scale, step)) in enumerate(
                (lr, recipe) for lr in (.1, .3, 1.) for recipe in recipes)]
    return dict(pca_query=baseline, dual_pair=dual)


class TunedDualPairModel(DualPairModel):
    """Same topology; configurable residual-output scale and step initialization."""
    def __init__(self, base, graph, hidden, rna_type, arm, config, width=16, depth=2):
        mode = {'dual_previous': 'dual_pair', 'matched_local': 'dual_local',
                'matched_transport': 'dual_transport'}.get(arm, arm)
        super().__init__(base, graph, hidden, rna_type, mode, width, depth)
        if not 0. < config['step_init'] < 1. or config['correction_scale'] <= 0.:
            raise ValueError('Invalid step initialization or output scale')
        self.correction_scale = float(config['correction_scale'])
        if self.branch is not None:
            value = math.log(config['step_init'] / (1. - config['step_init']))
            with torch.no_grad():
                for block in list(self.branch.blocks) + list(self.branch.local_blocks):
                    for parameter in block.step.values():
                        parameter.fill_(value)

    def forward(self, x_dict, edge_index_dict, pairs, weights,
                return_components=False, intervention=None):
        z = self.base.encoder(x_dict, edge_index_dict, weights)
        base = self.base.decoder(z, pairs)
        correction = (torch.zeros_like(base) if self.branch is None else
                      self.correction_scale * self.branch(z, edge_index_dict, pairs,
                                                          intervention=intervention))
        if return_components:
            return base + correction, base, correction
        return base + correction


def make_optimizer(model, base_lr, config):
    lr = base_lr * config['lr_multiplier']
    groups = [dict(params=list(model.base.parameters()), lr=lr, name='backbone')]
    if model.branch is not None:
        groups.append(dict(params=list(model.branch.parameters()),
                           lr=lr * config['branch_lr_multiplier'], name='pair_branch'))
    return torch.optim.RMSprop(groups, lr=lr, alpha=.99, weight_decay=config['weight_decay'])


def choose_epoch(traces, window):
    """First maximum of a trailing-window mean, equally averaged across seeds."""
    if not traces or window < 1:
        raise ValueError('Need traces and a positive smoothing window')
    curves = []
    for trace in traces:
        if not {'epoch', 'inner_auc', 'inner_aupr', 'criterion'}.issubset(trace.columns):
            raise ValueError('Only inner-validation traces can select a checkpoint')
        np.testing.assert_array_equal(trace.epoch.to_numpy(), np.arange(1, len(trace) + 1))
        curve = trace.criterion.to_numpy(dtype=float)
        if not np.isfinite(curve).all():
            raise ValueError('Nonfinite inner metric')
        curves.append(curve)
    if len(set(map(len, curves))) != 1 or len(curves[0]) < window:
        raise ValueError('Mismatched trace lengths or insufficient epochs')
    mean = np.mean(np.stack(curves), axis=0)
    smooth = np.asarray([np.mean(mean[i-window+1:i+1]) for i in range(window-1, len(mean))])
    position = int(np.argmax(smooth)) + window - 1
    return dict(epochs=position+1, selection_score=float(smooth[position-window+1]),
                inner_auc=float(np.mean([t.inner_auc.iloc[position] for t in traces])),
                inner_aupr=float(np.mean([t.inner_aupr.iloc[position] for t in traces])))


def sources():
    return dict(previous.sources(), **{name: sha256(Path(__file__).with_name(name)) for name in OWN_FILES})


def prepare(args):
    out, reference = Path(args.out_dir).resolve(), Path(args.previous_dir).resolve()
    if (out / 'protocol.json').exists():
        raise ValueError('Protocol exists; resume or use a new directory')
    prior = previous.load_protocol(reference)
    prior_selection = previous.load_selection(reference, prior)
    validation = json.loads((reference / 'VALIDATION.json').read_text())
    if not validation['passed'] or validation['smoke']:
        raise ValueError('Verified full previous experiment required')
    training = {task: dict(cfg, train_epochs=3 if args.smoke else int(1.5*cfg['train_epochs']))
                for task, cfg in prior['training'].items()}
    old_settings = {}
    for task in TASKS:
        old = prior_selection['chosen'][task]['dual_pair']
        old_settings[task] = dict(config=setting('previous', old['config']['lr_multiplier'],
                                                old['config']['weight_decay']),
                                  epochs=3 if args.smoke else old['epochs'])
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        previous_dir=str(reference), previous_protocol_sha256=prior['sha256'],
        previous_selection_sha256=prior_selection['sha256'], previous_validation_sha256=sha256(reference/'VALIDATION.json'),
        source_sha256=sources(), tasks=list(TASKS), search_arms=list(SEARCH_ARMS), refit_arms=list(REFIT_ARMS),
        primary='dual_pair', smoke=bool(args.smoke), configurations=configuration_grid(args.smoke),
        screening_seed=666, confirmation_seeds=[] if args.smoke else [0, 1],
        refit_seeds=[666] if args.smoke else [666, 0, 1, 2, 3],
        finalists_per_arm=1 if args.smoke else 3, smoothing_window=2 if args.smoke else 5,
        training=training, previous_settings=old_settings, query_fraction=.5, pair_width=16, pair_depth=2,
        inner_split_seed=prior['inner_split_seed'], inner_fraction=.2,
        selection='Each search arm has 12 fixed configurations. Rank screening seed 666 by the maximum trailing-5-epoch inner mean(AUC,AP). Confirm the top 3 with seeds 0 and 1. Select a single config and epoch from the smoothed curve averaged equally across all 3 seeds, then freeze before outer refitting.',
        optimization='RMSprop alpha=0.99, ordinary unweighted BCE, constant group learning rates; only LR, L2 decay, branch output scale and learnable pair-step initialization vary. Maximum epochs 150/450 for all search settings.',
        controls='Retuned original structure gets the same trial/seed/epoch budget. matched_local and matched_transport use the candidate-selected configuration and epoch without separate tuning, testing the propagation operator at fixed optimization settings. dual_previous reproduces the old selected hyperparameters.',
        graph_policy='Same frozen inputs and graph schema; current-stage target positives only, queried positives removed in both directions at every update. No explicit association boundary token in pair branch.',
        performance_screen='Mean outer AUC and AP both higher than retuned pca_query on BOTH tasks and >=4/5 positive paired mean-metric gains per task.',
        rescue_screen='Mean outer AUC and AP both higher than dual_previous on BOTH tasks and >=4/5 positive paired mean-metric gains per task.',
        mechanism_screen='Mean outer AUC and AP both higher than BOTH fixed-configuration matched controls, with >=4/5 positive paired mean-metric gains per task.',
        interventions=['branch_off','no_transport','zero_local'], no_test_evaluation=True,
        design_uses_previous_outer_results=True,
        limitations=['Both inner and outer validation have been reused; exploratory hyperparameter search.',
                     'Initialization seeds are not independent data splits; GPU training need not be bitwise deterministic.',
                     'Matched structural controls are not independently optimized in this round.',
                     'Equal allocated parameters do not imply equal effective capacity when a relation is disabled.',
                     'Existing graph includes circRNA extensions; not an exact reproduction of the original paper.'],
        files={}, audit=prior['audit'])
    out.mkdir(parents=True, exist_ok=True)
    for task in TASKS:
        dest = out / 'inputs' / task
        dest.mkdir(parents=True, exist_ok=True)
        for name in ('inner.npz','outer.npz','inner_indices.npz','train_graph.pth'):
            shutil.copy2(reference/'inputs'/task/name, dest/name)
            p['files'][str((dest/name).relative_to(out))] = sha256(dest/name)
    snapshot = out / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for name in p['source_sha256']:
        shutil.copy2(Path(__file__).with_name(name), snapshot/name)
    p['sha256'] = json_digest(p)
    write_json(out/'protocol.json', p)
    print('FROZEN', out, 'screen 48 / confirm 24 / refit 50' if not args.smoke else 'smoke', flush=True)


def load_protocol(out):
    out = Path(out)
    p = json.loads((out/'protocol.json').read_text())
    if p['sha256'] != json_digest({k:v for k,v in p.items() if k != 'sha256'}):
        raise ValueError('Protocol changed')
    if sources() != p['source_sha256']:
        raise ValueError('Frozen training source changed')
    for name, digest in p['files'].items():
        if sha256(out/name) != digest:
            raise ValueError('Input changed: '+name)
    return p


def build(arrays, graph, task, arm, seed, config, p, device):
    if arm not in REFIT_ARMS:
        raise ValueError(arm)
    base, data, pairs, weights = make_model(arrays, graph, task, 'pca', seed, device)
    torch.manual_seed(seed + 473299)
    cfg = p['training'][task]
    model = TunedDualPairModel(base, data, cfg['channels'], cfg['rna'], arm, config,
                               p['pair_width'], p['pair_depth']).to(device)
    return model, data, pairs, weights


predict = previous.predict
checked_result = previous.checked_result


def directory(out, task, arm, stage, config_id, seed):
    return Path(out) / ('refits' if stage == 'outer' else 'tuning') / task / arm / config_id / ('s'+str(seed))


def train(arrays, graph, task, arm, seed, config, stage, epochs, p, device):
    started = time.monotonic()
    model, data, pairs, weights = build(arrays, graph, task, arm, seed, config, p, device)
    initial, backbone = state_digest(model), state_digest(model.base)
    optimizer = make_optimizer(model, p['training'][task]['lr'], config)
    groups = [dict(name=g['name'], lr=g['lr'], weight_decay=g['weight_decay'],
                   params=sum(t.numel() for t in g['params'])) for g in optimizer.param_groups]
    labels = torch.from_numpy(arrays['y_train']).to(device)
    best, selected, state, best_prediction = -np.inf, None, None, None
    trace = []
    for epoch in range(1, epochs+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        query = sample_queries(arrays['y_train'], p['query_fraction'], (seed+1)*1000003+epoch)
        held = arrays['pairs_train'][:, query[arrays['y_train'][query] == 1]]
        edges, step_weights = remove_target_pairs(data.edge_index_dict, weights, task,
            p['training'][task]['rna'], data['drug'].num_nodes, held)
        indices = torch.from_numpy(arrays['pairs_train'][:,query]).to(device)
        logits = model(data.x_dict, edges, indices, step_weights)
        loss = F.binary_cross_entropy_with_logits(logits, labels[query])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training loss')
        loss.backward()
        optimizer.step()
        row = dict(epoch=epoch, loss=float(loss.detach()), queries=len(query))
        if stage == 'inner':
            output = predict(model, data, pairs['val'], weights)
            auc, ap = score(arrays['y_val'], output['val'])
            row.update(inner_auc=auc, inner_aupr=ap, criterion=(auc+ap)/2)
        trace.append(row)
        if stage == 'inner' and epoch >= p['smoothing_window']:
            criterion = float(np.mean([v['criterion'] for v in trace[-p['smoothing_window']:]]))
            if criterion > best:
                best, selected, best_prediction = criterion, epoch, output['val'].copy()
                state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        if epoch % max(50, epochs//2) == 0:
            print('EPOCH', task, arm, config['id'], seed, epoch, 'loss',row['loss'],
                  'inner' if stage=='inner' else 'refit',row.get('criterion',''),flush=True)
    if stage == 'outer':
        selected = epochs
        state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(state)
    output = predict(model, data, pairs['val'], weights)
    restore_error = 0.
    if best_prediction is not None:
        np.testing.assert_allclose(output['val'], best_prediction, atol=2e-4, rtol=2e-5)
        restore_error = float(np.max(np.abs(output['val']-best_prediction)))
    auc, ap = score(arrays['y_val'], output['val'])
    result = dict(task=task, arm=arm, seed=seed, config_id=config['id'], config=config,
        stage=stage, auc=auc, aupr=ap, criterion=(auc+ap)/2,
        selection_score=best if stage=='inner' else None,
        selected_epoch=selected, trained_epochs=epochs,
        initial_state_sha256=initial, initial_backbone_sha256=backbone,
        params=sum(t.numel() for t in model.parameters()), optimizer_groups=groups,
        restore_max_abs=restore_error, pair_abs_correction_mean=float(np.abs(output['correction']).mean()),
        elapsed_seconds=time.monotonic()-started, protocol_sha256=p['sha256'])
    return result, output, trace, state


def load_decision(out, name, p):
    out = Path(out)
    result = json.loads((out/name).read_text())
    if result['sha256'] != json_digest({k:v for k,v in result.items() if k != 'sha256'}):
        raise ValueError('Decision changed: '+name)
    if result['protocol_sha256'] != p['sha256']:
        raise ValueError('Wrong decision protocol')
    for path, digest in result['source_results_sha256'].items():
        if sha256(out/path) != digest:
            raise ValueError('Tuning result changed after selection')
    return result


def save_decision(out, name, decision):
    path = Path(out)/name
    if path.exists():
        raise ValueError('Decision already frozen: '+name)
    decision['sha256'] = json_digest(decision)
    write_json(path, decision)


def shortlist(args):
    out, p = Path(args.out_dir), load_protocol(args.out_dir)
    decision = dict(protocol_sha256=p['sha256'], chosen={}, ranking={}, source_results_sha256={})
    for task in TASKS:
        decision['chosen'][task], decision['ranking'][task] = {}, {}
        for arm in SEARCH_ARMS:
            records=[]
            for config in p['configurations'][arm]:
                path=directory(out,task,arm,'inner',config['id'],p['screening_seed'])
                result=checked_result(path,p)
                records.append(result)
                decision['source_results_sha256'][str((path/'result.json').relative_to(out))]=sha256(path/'result.json')
            ranked=sorted(records,key=lambda r:(-r['selection_score'],r['config_id']))
            decision['chosen'][task][arm]=[r['config'] for r in ranked[:p['finalists_per_arm']]]
            decision['ranking'][task][arm]=[dict(config_id=r['config_id'],selection_score=r['selection_score'],
                                                  epoch=r['selected_epoch']) for r in ranked]
    save_decision(out,'SHORTLIST.json',decision)
    print(json.dumps(decision['chosen'],indent=2),flush=True)


def select(args):
    out,p=Path(args.out_dir),load_protocol(args.out_dir)
    short=load_decision(out,'SHORTLIST.json',p)
    decision=dict(protocol_sha256=p['sha256'],shortlist_sha256=short['sha256'],chosen={},
                  candidates={},source_results_sha256={})
    for task in TASKS:
        decision['chosen'][task],decision['candidates'][task]={},{}
        for arm in SEARCH_ARMS:
            options=[]
            for config in short['chosen'][task][arm]:
                traces=[]
                for seed in [p['screening_seed']]+p['confirmation_seeds']:
                    path=directory(out,task,arm,'inner',config['id'],seed)
                    checked_result(path,p)
                    traces.append(pd.read_csv(path/'trace.csv',float_precision='round_trip'))
                    decision['source_results_sha256'][str((path/'result.json').relative_to(out))]=sha256(path/'result.json')
                options.append(dict(config=config,**choose_epoch(traces,p['smoothing_window'])))
            chosen=sorted(options,key=lambda r:(-r['selection_score'],r['config']['id']))[0]
            decision['chosen'][task][arm]=chosen
            decision['candidates'][task][arm]=options
        for arm in ('matched_local','matched_transport'):
            decision['chosen'][task][arm]=dict(decision['chosen'][task]['dual_pair'])
        decision['chosen'][task]['dual_previous']=p['previous_settings'][task]
    save_decision(out,'SELECTION.json',decision)
    print(json.dumps(decision['chosen'],indent=2),flush=True)


def jobs(out,p,stage):
    decision = load_decision(out,'SELECTION.json' if stage=='outer' else 'SHORTLIST.json',p) if stage!='screen' else None
    result=[]
    for task in TASKS:
        for arm in REFIT_ARMS if stage=='outer' else SEARCH_ARMS:
            if stage=='outer':
                selected=decision['chosen'][task][arm]
                configs=[selected['config']]
                seeds=p['refit_seeds']
                epochs=selected['epochs']
            else:
                configs=p['configurations'][arm] if stage=='screen' else decision['chosen'][task][arm]
                seeds=[p['screening_seed']] if stage=='screen' else p['confirmation_seeds']
                epochs=p['training'][task]['train_epochs']
            result.extend((task,arm,config,seed,epochs) for config in configs for seed in seeds)
    return result


def run(args):
    out,p=Path(args.out_dir),load_protocol(args.out_dir)
    if not 0 <= args.shard < args.shards:
        raise ValueError('Invalid shard')
    stage='outer' if args.stage=='outer' else 'inner'
    cache={}
    for task,arm,config,seed,epochs in jobs(out,p,args.stage)[args.shard::args.shards]:
        path=directory(out,task,arm,stage,config['id'],seed)
        path.mkdir(parents=True,exist_ok=True)
        with (path/'run.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if (path/'result.json').exists():
                checked_result(path,p)
                print('SKIP',task,arm,config['id'],seed,flush=True)
                continue
            if task not in cache:
                cache[task]=parent.load_stage(out,p,task,stage)
            arrays,graph=cache[task]
            print('START',args.stage,task,arm,config['id'],seed,flush=True)
            result,output,trace,state=train(arrays,graph,task,arm,seed,config,stage,epochs,p,args.device)
            np.savez_compressed(path/'scores.npz',**output,y_val=arrays['y_val'],pairs_val=arrays['pairs_val'])
            pd.DataFrame(trace).to_csv(path/'trace.csv',index=False)
            torch.save(state,path/'checkpoint.pt')
            result['artifacts_sha256']={name:sha256(path/name) for name in ('scores.npz','trace.csv','checkpoint.pt')}
            write_json(path/'result.json',result)
            print('DONE',task,arm,config['id'],seed,'%.6f/%.6f epoch %d %.1fs' % (
                result['auc'],result['aupr'],result['selected_epoch'],result['elapsed_seconds']),flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','shortlist','select'))
    parser.add_argument('--out-dir',required=True)
    parser.add_argument('--previous-dir',default='improve/results/dual_pair_20260909')
    parser.add_argument('--stage',choices=('screen','confirm','outer'),default='screen')
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--shards',type=int,default=1)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--cpu-threads',type=int,default=4)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    {'prepare':prepare,'run':run,'shortlist':shortlist,'select':select}[args.command](args)


if __name__=='__main__':
    main()
