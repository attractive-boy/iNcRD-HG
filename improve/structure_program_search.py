"""Run a frozen four-arm structural-program trial with the existing TabM.

prepare -> run screen -> select screen -> run confirm -> select confirm ->
run outer -> report. Feature/rule selection never uses held-out labels.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import pickle
import shutil
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score

from improve import backbone_replacement as parent
from improve import graph_tabm as table
from improve.graph_tabm_modules import fit_transform
from improve.structure_program_modules import ProgramEncoder, grammar, select_programs


ARMS = ('base', 'random', 'search', 'excess')
OWN = ('structure_program_search.py', 'structure_program_modules.py', 'test_structure_programs.py')


def sources():
    return dict(table.sources(), **{name: parent.digest(Path(__file__).with_name(name)) for name in OWN})


def prepare(args):
    out, reference = Path(args.out_dir).resolve(), Path(args.reference_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new output directory')
    prior = table.protocol(reference)
    if prior['smoke'] or not json.loads((reference/'VALIDATION.json').read_text())['passed']:
        raise ValueError('A verified full TabM reference is required')
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), smoke=args.smoke,
        reference_dir=str(reference), reference_protocol_sha256=prior['sha256'],
        input_reference_dir=prior['reference_dir'], tasks=list(parent.TASKS), arms=list(ARMS),
        screening_seed=666, confirmation_seeds=[0] if args.smoke else [0, 1],
        refit_seeds=[666] if args.smoke else [666, 0, 1],
        configurations=prior['configurations'][:1] if args.smoke else prior['configurations'],
        epochs={t: 3 if args.smoke else prior['epochs'][t] for t in parent.TASKS},
        batch_size=prior['batch_size'], smoothing_window=2 if args.smoke else 5,
        folds=prior['folds'], fold_seed=prior['fold_seed'], k=12, grammar_max_length=4,
        grammar=[x.record() for x in grammar(4)], null_samples=2 if args.smoke else 8,
        swaps_per_edge=2 if args.smoke else 5, null_seed=20260911, selector_seed=20260911,
        source_sha256=sources(),
        selection='Enumerate the typed grammar. Select 12 programs using inner-fitting labels only, after cross-fitted feature generation: correlation with degree-logistic residual, minus fixed length/base redundancy/selected redundancy penalties. Freeze program IDs before any network screening. Random control uses the same eligible pool and count. Screen two learning rates per arm, earliest maximum trailing-five mean(AUC,AP), then confirm seeds 0/1 and fix median epoch for outer refits.',
        arm_policy='base: original 36 features; random/search: append 12 raw programs; excess: append the conditional degree-null standardized values of the identical search programs. Same TabM and optimizer. Extended arms have identical input dimension and parameter count.',
        graph_policy='Target support excludes the entire query fold. Side graphs and other-task labels follow the historical protocol. Target bipartite double-edge swaps preserve both degree sequences and prohibit every query pair without reading query labels. Null variance floor, clipping and raw fallback are fixed. Programs shared as a grammar, task-specific selections; no transfer experiment claimed.',
        outer_policy='Outer inputs cannot be opened before REFIT is frozen. Program IDs are inherited from inner fitting, not reselected on outer data. No outer metric is computed within training epochs.',
        performance_gate='For a task, both mean AUC and AP exceed base, each with at least 2/3 positive seed deltas, and macro AP does not decrease. Dual-task success requires both tasks. Search-mechanism gate additionally requires positive mean AUC/AP versus random with >=2/3 positive seed deltas; null-mechanism gate requires the same versus raw search.',
        no_test_evaluation=True,
        limitations=['Historical inner and outer validation have already been used in development.',
            'This bounded grammar search is not an AntEvolve or SLogic reproduction.',
            'Eight conditional null samples are a pilot estimate, not a mixing proof.',
            'The null graph conditions on the complete query identity set, so its finite-sample features can change with a different query set.',
            'Unknown sampled pairs are not experimentally confirmed negatives.',
            'Three seeds measure optimization variation; no new split, external test or biological mechanism is established.'])
    out.mkdir(parents=True)
    (out/'source_snapshot').mkdir()
    for name in p['source_sha256']:
        shutil.copy2(Path(__file__).with_name(name), out/'source_snapshot'/name)
    p['sha256'] = parent.json_digest(p)
    parent.write_json(out/'protocol.json', p)
    print('FROZEN', out, 'programs', len(p['grammar']), flush=True)


def protocol(out):
    p = json.loads((Path(out)/'protocol.json').read_text())
    if p['sha256'] != parent.json_digest({k:v for k,v in p.items() if k != 'sha256'}) or p['source_sha256'] != sources():
        raise ValueError('Frozen protocol or source changed')
    if table.protocol(p['reference_dir'])['sha256'] != p['reference_protocol_sha256']:
        raise ValueError('TabM reference changed')
    return p


def checked_manifest(path, p):
    m = json.loads((path/'manifest.json').read_text())
    if m['protocol_sha256'] != p['sha256']:
        raise ValueError('Feature protocol changed')
    for name, sha in m['files'].items():
        if parent.digest(path/name) != sha:
            raise ValueError('Feature artifact changed: '+name)
    for filename, sha in m['dependencies'].items():
        if parent.digest(filename) != sha:
            raise ValueError('Feature dependency changed: '+filename)
    return m


def build_features(out, p, task, phase):
    out = Path(out)
    path = out/'features'/phase/task
    if phase == 'outer':
        parent.decision(out, 'REFIT.json', p)
    if (path/'manifest.json').exists():
        checked_manifest(path, p)
        return path
    started = time.monotonic()
    arrays, graph, _ = parent.stage_inputs(p['input_reference_dir'], task, phase)
    refp = table.protocol(p['reference_dir'])
    baseline = table.features(p['reference_dir'], refp, task, 'outer' if phase == 'outer' else 'screen')
    for name in ('pairs_train', 'pairs_val', 'y_train', 'y_val'):
        np.testing.assert_array_equal(arrays[name], baseline[name])
    programs = grammar(p['grammar_max_length'])
    encoder = ProgramEncoder(graph, task)
    pairs, labels, folds = arrays['pairs_train'], arrays['y_train'], baseline['fold_ids']
    raw = np.zeros((len(labels), len(programs)), dtype=np.float32)
    for fold in range(p['folds']):
        query, fitting = folds == fold, (folds != fold) & (labels == 1)
        raw[query] = encoder.evaluate(pairs[:,query], pairs[:,fitting], programs)
        print('PROGRAMS', phase, task, 'fold', fold, 'seconds', round(time.monotonic()-started, 1), flush=True)
    val = encoder.evaluate(arrays['pairs_val'], pairs[:,labels == 1], programs)
    path.mkdir(parents=True, exist_ok=True)
    dependencies = {str(Path(p['reference_dir'])/'features'/phase/task/'manifest.json'):
                    baseline['feature_manifest_sha256']}
    if phase == 'inner':
        rules = select_programs(baseline['raw_train'], raw, labels, programs, p['k'], p['selector_seed'])
        rules['protocol_sha256'] = p['sha256']
        rules['selected_names'] = [programs[i].name for i in rules['selected']]
        rules['random_names'] = [programs[i].name for i in rules['random']]
        parent.write_json(path/'rules.json', rules)
    else:
        inner = out/'features'/'inner'/task
        checked_manifest(inner, p)
        rules = json.loads((inner/'rules.json').read_text())
        shutil.copy2(inner/'rules.json', path/'rules.json')
        dependencies[str((inner/'rules.json').resolve())] = parent.digest(inner/'rules.json')
    selected = [programs[i] for i in rules['selected']]
    excess = np.zeros((len(labels), p['k']), dtype=np.float32)
    audits = []
    for fold in range(p['folds']):
        query, fitting = folds == fold, (folds != fold) & (labels == 1)
        original, excess[query], audit = encoder.with_null(pairs[:,query], pairs[:,fitting], selected,
            p['null_samples'], p['null_seed'] + 1000*fold, p['swaps_per_edge'])
        np.testing.assert_allclose(original, raw[query][:,rules['selected']], rtol=1e-6, atol=1e-6)
        audits.append(dict(fold=fold, **audit))
        print('NULL', phase, task, 'fold', fold, 'seconds', round(time.monotonic()-started, 1), flush=True)
    original, excess_val, audit = encoder.with_null(arrays['pairs_val'], pairs[:,labels == 1], selected,
        p['null_samples'], p['null_seed'] + 9000, p['swaps_per_edge'])
    np.testing.assert_allclose(original, val[:,rules['selected']], rtol=1e-6, atol=1e-6)
    audits.append(dict(fold='evaluation', **audit))
    data = dict(y_train=labels, y_val=arrays['y_val'], pairs_train=pairs,
        pairs_val=arrays['pairs_val'], fold_ids=folds, raw_program_train=raw, raw_program_val=val,
        excess_program_train=excess, excess_program_val=excess_val)
    for arm in ARMS:
        tr, ev = baseline['raw_train'], baseline['raw_val']
        if arm != 'base':
            ids = rules['random'] if arm == 'random' else rules['selected']
            tr = np.concatenate([tr, excess if arm == 'excess' else raw[:,ids]], 1)
            ev = np.concatenate([ev, excess_val if arm == 'excess' else val[:,ids]], 1)
        x, xv, transform = fit_transform(tr, ev)
        if arm == 'base':
            np.testing.assert_array_equal(x, baseline['x_train'])
            np.testing.assert_array_equal(xv, baseline['x_val'])
        data[arm+'_x_train'], data[arm+'_x_val'] = x, xv
        with (path/('transform_'+arm+'.pkl')).open('wb') as stream:
            pickle.dump(transform, stream, protocol=4)
    np.savez_compressed(path/'arrays.npz', **data)
    parent.write_json(path/'null_audit.json', dict(protocol_sha256=p['sha256'], phases=audits))
    filenames = ['arrays.npz', 'rules.json', 'null_audit.json'] + ['transform_'+a+'.pkl' for a in ARMS]
    parent.write_json(path/'manifest.json', dict(protocol_sha256=p['sha256'], task=task, phase=phase,
        dependencies=dependencies, files={name:parent.digest(path/name) for name in filenames},
        seconds=time.monotonic()-started, program_names=[v.name for v in programs],
        selected_names=rules['selected_names'], random_names=rules['random_names']))
    print('FEATURES', phase, task, raw.shape, 'selected', rules['selected_names'], flush=True)
    return path


def features(out, p, task, stage, arm):
    path = build_features(out, p, task, 'outer' if stage == 'outer' else 'inner')
    checked_manifest(path, p)
    with np.load(path/'arrays.npz') as archive:
        a = {key:archive[key].copy() for key in ('y_train','y_val','pairs_train','pairs_val')}
        a['x_train'], a['x_val'] = archive[arm+'_x_train'].copy(), archive[arm+'_x_val'].copy()
    a['feature_manifest_sha256'] = parent.digest(path/'manifest.json')
    return a


def run(args):
    out, p = Path(args.out_dir), protocol(args.out_dir)
    planned = [job for job in table.jobs(out,p,args.stage) if args.arm is None or job[1] == args.arm]
    for task in ([args.task] if args.task else p['tasks']):
        # Feature creation is shared and serialized. Training jobs use separate
        # locks so disjoint arms can run concurrently without duplicate writes.
        with (out/(task+'_features.lock')).open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            build_features(out, p, task, 'outer' if args.stage == 'outer' else 'inner')
        for t, arm, cfg, seed, epochs in planned:
            if t != task:
                continue
            dest = parent.directory(out, args.stage, task, arm, cfg['id'], seed)
            dest.mkdir(parents=True, exist_ok=True)
            with (dest/'job.lock').open('a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if (dest/'result.json').exists():
                    parent.checked_result(dest, p)
                    continue
                a = features(out, p, task, args.stage, arm)
                r, scores, trace, state = table.train(a, task, 'tabm', cfg, seed, args.stage, epochs, p, args.device)
                r['arm'] = arm
                r['input_dim'] = a['x_train'].shape[1]
                np.savez_compressed(dest/'scores.npz', val=scores, y_val=a['y_val'], pairs_val=a['pairs_val'])
                pd.DataFrame(trace).to_csv(dest/'trace.csv', index=False)
                torch.save(state, dest/'checkpoint.pt')
                r['artifacts_sha256'] = {name:parent.digest(dest/name) for name in ('scores.npz','trace.csv','checkpoint.pt')}
                parent.write_json(dest/'result.json', r)
                print('DONE', task, arm, cfg['id'], seed, r['auc'], r['ap'], r['selected_epoch'], flush=True)


def select(args):
    out, p = Path(args.out_dir), protocol(args.out_dir)
    if args.stage == 'outer':
        raise ValueError('Outer results must not select a model')
    filename = 'PARAMETERS.json' if args.stage == 'screen' else 'REFIT.json'
    if (out/filename).exists():
        print(json.dumps(parent.decision(out, filename, p)['chosen'], indent=2), flush=True)
        return
    previous = parent.decision(out, 'PARAMETERS.json', p)['chosen'] if args.stage == 'confirm' else None
    decision = dict(protocol_sha256=p['sha256'], chosen={}, results_sha256={})
    for task in p['tasks']:
        decision['chosen'][task] = {}
        for arm in p['arms']:
            if args.stage == 'screen':
                paths = [parent.directory(out,'screen',task,arm,cfg['id'],p['screening_seed']) for cfg in p['configurations']]
            else:
                cfg = previous[task][arm]
                paths = [parent.directory(out,'screen',task,arm,cfg['id'],p['screening_seed'])]
                paths += [parent.directory(out,'confirm',task,arm,cfg['id'],seed) for seed in p['confirmation_seeds']]
            rows = [parent.checked_result(path,p) for path in paths]
            for path in paths:
                decision['results_sha256'][str((path/'result.json').relative_to(out))] = parent.digest(path/'result.json')
            if args.stage == 'screen':
                decision['chosen'][task][arm] = min(rows,key=lambda r:(-r['selection_score'],r['config_id']))['config']
            else:
                decision['chosen'][task][arm] = dict(config=previous[task][arm],
                    epochs=int(np.rint(np.median([r['selected_epoch'] for r in rows]))),
                    inner_auc=float(np.mean([r['auc'] for r in rows])),
                    inner_ap=float(np.mean([r['ap'] for r in rows])))
    decision['sha256'] = parent.json_digest(decision)
    parent.write_json(out/filename,decision)
    print(json.dumps(decision['chosen'],indent=2),flush=True)


def extra_metrics(y, scores, pairs):
    top10, per_drug = [], []
    for drug in np.unique(pairs[1]):
        mask = pairs[1] == drug
        if len(np.unique(y[mask])) < 2:
            continue
        ids = np.flatnonzero(mask)
        order = ids[np.argsort(-scores[ids], kind='stable')]
        top10.append(float(y[order[:10]].sum() / y[ids].sum()))
        per_drug.append(dict(drug=int(drug), ap=float(average_precision_score(y[ids], scores[ids])),
                             n=len(ids), positives=int(y[ids].sum())))
    return dict(macro_recall10_sampled=float(np.mean(top10))), per_drug


def gate(frame, task, arm, control, macro=True):
    left = frame[(frame.task == task) & (frame.arm == arm)].sort_values('seed')
    right = frame[(frame.task == task) & (frame.arm == control)].sort_values('seed')
    if len(left) != 3 or len(right) != 3:
        return False
    delta = left[['auc','ap','macro_drug_ap']].to_numpy() - right[['auc','ap','macro_drug_ap']].to_numpy()
    return bool(np.all(delta[:,:2].mean(0) > 0) and np.all((delta[:,:2] > 0).sum(0) >= 2)
                and (not macro or delta[:,2].mean() >= 0))


def report(args):
    out, p = Path(args.out_dir), protocol(args.out_dir)
    rows, checks, drug_rows = [], [], []
    for stage in ('screen','confirm','outer'):
        for task, arm, cfg, seed, epochs in table.jobs(out, p, stage):
            dest = parent.directory(out, stage, task, arm, cfg['id'], seed)
            r = parent.checked_result(dest, p)
            a = features(out, p, task, stage, arm)
            if r['feature_manifest_sha256'] != a['feature_manifest_sha256']:
                raise ValueError('Training feature hash changed')
            with np.load(dest/'scores.npz') as archive:
                saved = archive['val'].copy()
                np.testing.assert_array_equal(archive['y_val'], a['y_val'])
                np.testing.assert_array_equal(archive['pairs_val'], a['pairs_val'])
            for key, value in parent.metrics(a['y_val'], saved, a['pairs_val']).items():
                if value is not None:
                    np.testing.assert_allclose(r[key], value, rtol=0, atol=1e-12)
            trace = pd.read_csv(dest/'trace.csv')
            np.testing.assert_array_equal(trace.epoch, np.arange(1, epochs+1))
            if stage != 'outer':
                epoch, score = parent.choose_checkpoint(trace.to_dict('records'), p['smoothing_window'])
                if epoch != r['selected_epoch']:
                    raise AssertionError('Checkpoint epoch mismatch')
                np.testing.assert_allclose(score, r['selection_score'], rtol=0, atol=1e-12)
            elif any(k in trace for k in ('auc','ap','criterion')):
                raise ValueError('Outer metrics were used during training')
            model = table.model_for('tabm', a['x_train'].shape[1], cfg, seed, args.device)
            model.load_state_dict(torch.load(dest/'checkpoint.pt', map_location='cpu', weights_only=False))
            replay = table.predict(model, torch.from_numpy(a['x_val']).to(args.device), p['batch_size'])
            np.testing.assert_allclose(replay, saved, rtol=2e-6, atol=2e-6)
            checks.append(dict(path=str(dest.relative_to(out)), max_abs=float(np.max(np.abs(replay-saved)))))
            if stage == 'outer':
                extras, drug = extra_metrics(a['y_val'], saved, a['pairs_val'])
                r.update(extras)
                drug_rows += [dict(task=task, arm=arm, seed=seed, **v) for v in drug]
            rows.append(r)
    data = pd.DataFrame(rows)
    data.to_csv(out/'results.csv', index=False)
    outer = data[data.stage == 'outer']
    metrics = ['auc','ap','macro_drug_ap','macro_recall10_sampled']
    outer.groupby(['task','arm'])[metrics].agg(['mean','std']).to_csv(out/'outer_summary.csv')
    pd.DataFrame(drug_rows).to_csv(out/'per_drug.csv', index=False)
    deltas = []
    for task in p['tasks']:
        for arm, control in [('random','base'),('search','base'),('excess','base'),('search','random'),('excess','search')]:
            x = outer[(outer.task == task) & (outer.arm == arm)].set_index('seed')
            y = outer[(outer.task == task) & (outer.arm == control)].set_index('seed')
            for seed in x.index:
                deltas.append(dict(task=task, arm=arm, control=control, seed=int(seed),
                                   **{m:float(x.loc[seed,m]-y.loc[seed,m]) for m in metrics}))
    pd.DataFrame(deltas).to_csv(out/'paired_deltas.csv', index=False)
    gates = {task: dict(search_vs_base=gate(outer,task,'search','base'),
        excess_vs_base=gate(outer,task,'excess','base'),
        search_vs_random=gate(outer,task,'search','random',False),
        excess_vs_search=gate(outer,task,'excess','search',False)) for task in p['tasks']}
    verification = dict(passed=True, smoke=p['smoke'], completed_runs=len(rows),
        protocol_sha256=p['sha256'], no_test_evaluation=True, all_checkpoints_replayed=True,
        gates=gates, replays=checks)
    parent.write_json(out/'VALIDATION.json', verification)
    means = outer.groupby(['task','arm'])[metrics].mean()
    lines = ['# 类型约束的结构程序搜索：冻结实验结果', '',
        '历史外层验证，三个初始化；AP 为 average precision。不是独立测试。', '',
        '| 任务 | 组别 | AUC | AP | 按药物宏 AP |', '|---|---|---:|---:|---:|']
    for task in p['tasks']:
        for arm in p['arms']:
            v = means.loc[(task,arm)]
            lines.append('| {} | {} | {:.6f} | {:.6f} | {:.6f} |'.format(task,arm,v.auc,v.ap,v.macro_drug_ap))
    lines += ['', '四组使用相同 TabM：base 为原 36 维；其余三组均为 48 维。random 使用随机程序，search 使用训练部分选择的程序，excess 使用同一批搜索程序的条件度数空模型校正值。', '',
        '## 预设筛选结果', '', '性能筛选要求 AUC/AP 均值高于 base，两个指标各至少 2/3 种子提高，且宏 AP 不下降。机制筛选比较 search 与 random、excess 与 search；不以工程核验通过代替性能通过。', '']
    for task, values in gates.items():
        lines.append('- '+task+'：'+json.dumps(values, ensure_ascii=False))
    lines += ['', '## 冻结规则', '']
    for task in p['tasks']:
        rule = json.loads((out/'features'/'inner'/task/'rules.json').read_text())
        lines.append('- '+task+'：'+', '.join('`'+s+'`' for s in rule['selected_names']))
    lines += ['', '符号：r 为目标 RNA，o 为另一类 RNA，d 为药物；A 为 r→d，B 为 o→d，C 为 r→o，小写为逆向；S/T/D 为对应类型相似图。count 为计数，walk 为逐关系行归一化，symmetric 为端点度数对称归一化。nonbacktracking 为目标关联图上长度五、禁止立即回到上一个节点的路径。', '',
        '## 核验与限制', '',
        '- 完成 {} 次训练，保存预测/轨迹/检查点的哈希，重放所有检查点并重算指标。'.format(len(rows)),
        '- 训练特征按五折遮蔽目标正边生成；规则只用内层拟合标签选择，外层沿用固定规则；外层重训轮数由三个内层轮数的中位数确定。',
        '- 空模型只随机交换目标关联，保留两端度数，排除全部查询位置。8 次随机图是小原型估计；每折混合程度及回退比例见 features/*/*/null_audit.json。',
        '- Recall@10 只在给定验证候选集合内计算，不能解释为全 RNA 候选筛选的召回率。',
        '- 原图含另一任务和 circRNA 信息，沿用历史单任务协议；本轮没有证明严格冷启动或跨任务迁移。',
        '- 历史验证被多次用于开发；初始化种子不是新划分。未读取测试集。', '',
        '逐种子及标准差：results.csv、outer_summary.csv；配对差：paired_deltas.csv；逐药物：per_drug.csv；核验：VALIDATION.json。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(means.to_string(), flush=True)
    print('VERIFIED', len(rows), 'checkpoints', json.dumps(gates), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare','run','select','report'))
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--reference-dir', default='improve/results/graph_tabm_20260910')
    parser.add_argument('--stage', choices=('screen','confirm','outer'), default='screen')
    parser.add_argument('--task', choices=parent.TASKS)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    {'prepare':prepare, 'run':run, 'select':select, 'report':report}[args.command](args)


if __name__ == '__main__':
    main()
