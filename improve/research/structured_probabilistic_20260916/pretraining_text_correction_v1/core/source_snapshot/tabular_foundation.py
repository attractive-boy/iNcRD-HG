"""Frozen official TabICLv2 inference on the verified graph-feature inputs.

prepare -> inner (four recipes, three inference seeds) -> select -> outer -> report.
The default recipe is always reported; outer scores never select a recipe.
This module deliberately does not import the historical graph/PyTorch stack.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


OWN = ('tabular_foundation.py', 'test_tabular_foundation.py', 'analyze_tabular_foundation.py')
DEFAULT_CHECKPOINT = '/data1/licheng/incrd-hg/.external/models/tabicl/tabicl-classifier-v2-20260212.ckpt'
DEFAULT_REFERENCE = 'improve/results/numeric_embedding_20260912'
DEFAULT_OUT = 'improve/results/tabicl_20260912'
METRICS = ('auc', 'ap', 'macro_drug_ap')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            value.update(block)
    return value.hexdigest()


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def check_seal(value):
    if value.get('sha256') != json_digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('Frozen document changed')
    return value


def seal(value):
    return dict(value, sha256=json_digest(value))


def sources():
    return {name: digest(Path(__file__).with_name(name)) for name in OWN}


def configurations():
    return [dict(id='c%d' % i, input_view=view, n_estimators=members)
            for i, (view, members) in enumerate((('raw', 8), ('raw', 32), ('quantile', 8), ('quantile', 32)))]


def metrics(y, scores, pairs):
    y, scores, pairs = np.asarray(y), np.asarray(scores), np.asarray(pairs)
    if y.ndim != 1 or scores.shape != y.shape or pairs.shape != (2, len(y)):
        raise ValueError('Invalid metric input shapes')
    if set(np.unique(y)) != {0, 1} or not np.isfinite(scores).all():
        raise ValueError('Metrics require both classes and finite scores')
    auc, ap = float(roc_auc_score(y, scores)), float(average_precision_score(y, scores))
    per_drug = []
    for drug in np.unique(pairs[1]):
        mask = pairs[1] == drug
        if len(np.unique(y[mask])) == 2:
            per_drug.append(average_precision_score(y[mask], scores[mask]))
    return dict(auc=auc, ap=ap, criterion=(auc+ap)/2,
                macro_drug_ap=float(np.mean(per_drug)) if per_drug else None, macro_drugs=len(per_drug))


def validate_arrays(a):
    for part in ('train', 'val'):
        y, pairs = a['y_'+part], a['pairs_'+part]
        if y.ndim != 1 or pairs.shape != (2, len(y)) or set(np.unique(y)) != {0, 1}:
            raise ValueError('Invalid labeled pairs')
        if len(set(map(tuple, pairs.T))) != len(y):
            raise ValueError('Duplicate labeled pairs')
        for prefix in ('raw', 'x'):
            x = a[prefix+'_'+part]
            if x.shape != (len(y), 36) or not np.isfinite(x).all():
                raise ValueError('Expected 36 finite graph features')
    if set(map(tuple, a['pairs_train'].T)) & set(map(tuple, a['pairs_val'].T)):
        raise ValueError('Fitting and evaluation pairs overlap')


def balanced_subset(y, budget, seed):
    if budget < 2:
        raise ValueError('Subset needs both classes')
    if len(y) <= budget:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    chosen = [int(rng.choice(np.flatnonzero(y == c))) for c in (0, 1)]
    available = np.setdiff1d(np.arange(len(y)), chosen)
    return np.sort(np.r_[chosen, rng.choice(available, budget-2, replace=False)]).astype(int)


def selection(out, p):
    value = check_seal(read_json(Path(out)/'SELECTION.json'))
    if value['protocol_sha256'] != p['sha256']:
        raise ValueError('Selection belongs to another protocol')
    for relative, expected in value['results_sha256'].items():
        if digest(Path(out)/relative) != expected:
            raise ValueError('Selected inner result changed')
    return value


def load_features(out, p, task, phase):
    if phase not in ('inner', 'outer') or task not in p['tasks']:
        raise ValueError('Unknown task or phase')
    if phase == 'outer':
        selection(out, p)  # This check must precede np.load of outer arrays.
    ref = Path(p['reference_dir'])
    relative = 'features/%s/%s' % (phase, task)
    manifest_path = ref/relative/'manifest.json'
    if digest(manifest_path) != p['reference_files'][relative+'/manifest.json']:
        raise ValueError('Feature manifest changed')
    manifest = read_json(manifest_path)
    if manifest['protocol_sha256'] != p['reference_protocol_sha256']:
        raise ValueError('Feature protocol mismatch')
    for name, sha in manifest['files'].items():
        if digest(ref/relative/name) != sha:
            raise ValueError('Feature cache changed')
    with np.load(ref/relative/'arrays.npz', allow_pickle=False) as archive:
        if any('test' in key.lower() for key in archive.files):
            raise ValueError('Test arrays are forbidden')
        keys = ('raw_train', 'raw_val', 'x_train', 'x_val', 'y_train', 'y_val', 'pairs_train', 'pairs_val')
        a = {key: archive[key].copy() for key in keys}
    validate_arrays(a)
    if p['smoke']:
        for part, budget in (('train', 128), ('val', 40)):
            keep = balanced_subset(a['y_'+part], budget, 20260912)
            for prefix in ('raw', 'x', 'y'):
                a[prefix+'_'+part] = a[prefix+'_'+part][keep]
            a['pairs_'+part] = a['pairs_'+part][:, keep]
    a['feature_manifest_sha256'] = digest(manifest_path)
    return a


def audit_reference(ref):
    p = check_seal(read_json(ref/'protocol.json'))
    v = read_json(ref/'VALIDATION.json')
    if p['smoke'] or not v['passed'] or not v['all_checkpoints_replayed'] or v['protocol_sha256'] != p['sha256']:
        raise ValueError('Need a verified formal reference')
    if not read_json(ref/'ANCHOR_VALIDATION.json')['passed']:
        raise ValueError('Reference TabM anchor was not verified')
    for name, expected in p['source_sha256'].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError('Historical source changed: '+name)
    parent_root = Path(p['reference_dir'])
    parent = check_seal(read_json(parent_root/'protocol.json'))
    if parent['sha256'] != p['reference_protocol_sha256']:
        raise ValueError('Reference input protocol changed')
    for name, expected in parent['files'].items():
        if digest(parent_root/name) != expected:
            raise ValueError('Historical graph input changed')
    return p


def package_state():
    import tabicl
    packages = ('tabicl', 'torch', 'numpy', 'scipy', 'scikit-learn', 'pandas', 'einops', 'huggingface-hub')
    root = Path(tabicl.__file__).parent
    return dict(versions={name: importlib.metadata.version(name) for name in packages},
                source_root=str(root), source_sha256={str(f.relative_to(root)): digest(f)
                                                     for f in sorted(root.rglob('*.py'))})


def prepare(args):
    out, ref, checkpoint = Path(args.out_dir).resolve(), Path(args.reference_dir).resolve(), Path(args.checkpoint).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new experiment directory')
    before = audit_reference(ref)
    weight_manifest = read_json(checkpoint.parent/'download_manifest.json')
    if checkpoint.name != weight_manifest['filename'] or digest(checkpoint) != weight_manifest['sha256']:
        raise ValueError('Official checkpoint hash mismatch')
    files = ['protocol.json', 'VALIDATION.json', 'ANCHOR_VALIDATION.json', 'REFIT.json']
    baseline = {}
    chosen = read_json(ref/'REFIT.json')['chosen']
    for task in before['tasks']:
        baseline[task] = {}
        for seed in (666, 0, 1):
            relative = 'outer/%s/tabm/%s/s%d' % (task, chosen[task]['tabm']['config']['id'], seed)
            baseline[task][str(seed)] = relative
            files += [relative+'/'+name for name in ('result.json', 'scores.npz', 'checkpoint.pt')]
        for phase in ('inner', 'outer'):
            relative = 'features/%s/%s' % (phase, task)
            manifest = read_json(ref/relative/'manifest.json')
            if manifest['protocol_sha256'] != before['sha256']:
                raise ValueError('Cached feature protocol changed')
            for name, expected in manifest['files'].items():
                if digest(ref/relative/name) != expected:
                    raise ValueError('Cached feature artifact changed')
            files.append(relative+'/manifest.json')
            files += [relative+'/'+name for name in manifest['files']]
    p = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), smoke=args.smoke,
        tasks=before['tasks'], seeds=[666] if args.smoke else [666, 0, 1],
        configurations=[configurations()[0], configurations()[2]] if args.smoke else configurations(),
        default_id='c0', reference_dir=str(ref), reference_protocol_sha256=before['sha256'],
        reference_files={name: digest(ref/name) for name in files}, baseline=baseline,
        checkpoint=str(checkpoint), checkpoint_metadata=weight_manifest,
        checkpoint_manifest_sha256=digest(checkpoint.parent/'download_manifest.json'),
        package=package_state(), source_sha256=sources(), no_test_evaluation=True,
        implementation=dict(upstream_repository='https://github.com/soda-inria/tabicl',
            upstream_commit='0dbff3ec8fc68c123c87af77b0ea8b25cd2d23f3',
            checkpoint_version='tabicl-classifier-v2-20260212.ckpt'),
        inference=dict(batch_size=4, use_amp=False, use_fa3=False, kv_cache=False,
            offload_mode='auto', n_jobs=4, softmax_temperature=.9, average_logits=True,
            feat_shuffle_method='latin', class_shuffle_method='shift', outlier_threshold=4.),
        selection='Four recipes (raw/old quantile features x 8/32 inference members) all evaluated on inner seeds 666/0/1. Select the largest seed-mean (AUC+AP)/2, stable ties by config ID. Freeze before any outer arrays are opened. Always report the raw/8 default as well as the inner-selected recipe.',
        scope='Official frozen pretrained weights, all fitting rows as context, no parameter updates, no extra biological inputs. The reference caches contain verified five-fold graph features with query positives removed. Official none/power ensemble preprocessing is fit only on context rows.',
        randomization='TabICL seeds control preprocessing/feature/class permutation ensembles; TabM seeds control initialization. Matching seed labels is bookkeeping, not a matched source of statistical randomness or independent pretrained models.',
        gate='For each predeclared panel (default or inner-selected), both tasks must improve mean AUC and AP versus the frozen verified TabM anchor, with positive differences in at least 2/3 corresponding seed labels for each metric. Macro drug AP is reported separately. No outer recipe selection.',
        replay_atol=1e-6, replay_rtol=1e-6,
        limitations=before['limitations']+[
            'External synthetic pretraining compute is not matched to TabM; this compares available predictors.',
            'The scalar TabM anchor also remained selected in the previous 12-recipe tuning study.',
            'Official inference ensembles share one pretrained checkpoint; repeated seeds are not training replications.',
            'Historical validation is reused development evidence, not independent generalization evidence.',
            'Float32 and ensemble batch size are fixed resource settings; A40 runtime is measured locally.'])
    p = seal(p)
    out.mkdir(parents=True, exist_ok=True)
    snapshot = out/'source_snapshot'
    snapshot.mkdir()
    for name in OWN:
        shutil.copy2(Path(__file__).with_name(name), snapshot/name)
    write_json(out/'protocol.json', p)
    import torch
    write_json(out/'ENVIRONMENT.json', dict(executable=sys.executable, python=sys.version,
        torch_cuda=torch.version.cuda, devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        packages=p['package']['versions'], pip_freeze=subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)))
    print('FROZEN', out, flush=True)


def protocol(out):
    p = check_seal(read_json(Path(out)/'protocol.json'))
    if sources() != p['source_sha256'] or package_state() != p['package']:
        raise ValueError('Frozen experiment implementation/environment changed')
    if digest(p['checkpoint']) != p['checkpoint_metadata']['sha256']:
        raise ValueError('Frozen pretrained weights changed')
    if digest(Path(p['checkpoint']).parent/'download_manifest.json') != p['checkpoint_manifest_sha256']:
        raise ValueError('Checkpoint provenance changed')
    for name, expected in p['reference_files'].items():
        if digest(Path(p['reference_dir'])/name) != expected:
            raise ValueError('Reference artifact changed: '+name)
    return p


def estimator(cfg, seed, p, device):
    import torch
    from tabicl import TabICLClassifier
    torch.set_num_threads(4)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return TabICLClassifier(n_estimators=cfg['n_estimators'], random_state=seed,
        model_path=p['checkpoint'], checkpoint_version=p['implementation']['checkpoint_version'],
        allow_auto_download=False, device=device, **p['inference'])


def fit_context(model, x, y):
    if x.ndim != 2 or y.shape != (len(x),) or set(np.unique(y)) != {0, 1}:
        raise ValueError('Invalid context data')
    model.fit(x, y.astype(np.int64))
    return model


def query_scores(model, x):
    probabilities = np.asarray(model.predict_proba(x))
    classes = np.asarray(model.classes_)
    if probabilities.shape != (len(x), 2) or set(classes) != {0, 1}:
        raise ValueError('Expected binary probability columns')
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0) or np.any(probabilities > 1):
        raise ValueError('Invalid class probabilities')
    np.testing.assert_allclose(probabilities.sum(1), 1., atol=2e-6, rtol=0)
    return probabilities[:, int(np.flatnonzero(classes == 1)[0])].astype(np.float64)


def parameter_digest(model):
    h = hashlib.sha256()
    for name, value in model.model_.named_parameters():
        array = value.detach().cpu().contiguous().numpy()
        h.update(name.encode())
        h.update(str(array.shape).encode())
        h.update(str(array.dtype).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def directory(out, phase, task, cfg_id, seed):
    return Path(out)/phase/task/cfg_id/('s%d' % seed)


def jobs(out, p, phase):
    frozen = selection(out, p) if phase == 'outer' else None
    if phase not in ('inner', 'outer'):
        raise ValueError(phase)
    for task in p['tasks']:
        configs = p['configurations']
        if phase == 'outer':
            needed = {p['default_id'], frozen['chosen'][task]['config']['id']}
            configs = [cfg for cfg in configs if cfg['id'] in needed]
        for cfg in configs:
            for seed in p['seeds']:
                yield task, cfg, seed


def checked_result(dest, p):
    r = read_json(Path(dest)/'result.json')
    if r['protocol_sha256'] != p['sha256']:
        raise ValueError('Result protocol mismatch')
    for name, expected in r['artifacts_sha256'].items():
        if digest(Path(dest)/name) != expected:
            raise ValueError('Inference artifact changed')
    return r


def run_job(out, p, phase, task, cfg, seed, device):
    import torch
    dest = directory(out, phase, task, cfg['id'], seed)
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/'.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (dest/'result.json').exists():
            checked_result(dest, p)
            print('VERIFIED_SKIP', phase, task, cfg['id'], seed, flush=True)
            return
        a = load_features(out, p, task, phase)
        prefix = 'raw' if cfg['input_view'] == 'raw' else 'x'
        start = time.monotonic()
        if str(device).startswith('cuda'):
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
        model = fit_context(estimator(cfg, seed, p, device), a[prefix+'_train'], a['y_train'])
        before = parameter_digest(model)
        scores = query_scores(model, a[prefix+'_val'])
        if parameter_digest(model) != before:
            raise ValueError('Pretrained parameters changed during inference')
        model.save(dest/'fitted.pkl', save_model_weights=False, save_training_data=True, save_kv_cache=False)
        np.savez_compressed(dest/'scores.npz', val=scores, y_val=a['y_val'], pairs_val=a['pairs_val'])
        r = dict(task=task, phase=phase, config=cfg, seed=seed, **metrics(a['y_val'], scores, a['pairs_val']),
            fitting_rows=len(a['y_train']), evaluation_rows=len(a['y_val']),
            trained_epochs=0, pretrained_parameters=sum(v.numel() for v in model.model_.parameters()),
            parameter_sha256=before, elapsed_seconds=time.monotonic()-start, device=device,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(device) if str(device).startswith('cuda') else None,
            feature_manifest_sha256=a['feature_manifest_sha256'], protocol_sha256=p['sha256'],
            artifacts_sha256={name: digest(dest/name) for name in ('scores.npz', 'fitted.pkl')})
        write_json(dest/'result.json', r)
        print('INFERENCE', phase, task, cfg['id'], seed, 'AUC', round(r['auc'], 6),
              'AP', round(r['ap'], 6), 'seconds', round(r['elapsed_seconds'], 1), flush=True)
        del model
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()


def select_recipe(configs, rows, seeds):
    if len(rows) != len(configs)*len(seeds):
        raise ValueError('Incomplete configuration/seed grid')
    summaries = []
    for cfg in configs:
        group = [r for r in rows if r['config'] == cfg]
        if len(group) != len(seeds) or sorted(r['seed'] for r in group) != sorted(seeds):
            raise ValueError('Missing or repeated inference seeds')
        if any(not np.isfinite(r['criterion']) for r in group):
            raise ValueError('Non-finite inner selection score')
        measured = {}
        for metric in METRICS:
            available = [r[metric] for r in group if r[metric] is not None]
            measured['inner_'+metric] = float(np.mean(available)) if available else None
        summaries.append(dict(config=cfg, mean_selection_score=float(np.mean([r['criterion'] for r in group])), **measured))
    return sorted(summaries, key=lambda r: (-r['mean_selection_score'], r['config']['id']))[0]


def select(out, p):
    if (Path(out)/'SELECTION.json').exists():
        selection(out, p)
        print('SELECTION_ALREADY_FROZEN', flush=True)
        return
    records, hashes = {}, {}
    for task, cfg, seed in jobs(out, p, 'inner'):
        dest = directory(out, 'inner', task, cfg['id'], seed)
        r = checked_result(dest, p)
        if (r['task'], r['config'], r['seed'], r['phase']) != (task, cfg, seed, 'inner'):
            raise ValueError('Inner result identity mismatch')
        records.setdefault(task, []).append(r)
        hashes[str((dest/'result.json').relative_to(out))] = digest(dest/'result.json')
    chosen = {task: select_recipe(p['configurations'], records[task], p['seeds']) for task in p['tasks']}
    write_json(Path(out)/'SELECTION.json', seal(dict(protocol_sha256=p['sha256'], chosen=chosen, results_sha256=hashes)))
    print('SELECTED', json.dumps(chosen, ensure_ascii=False), flush=True)


def pipeline(out, devices):
    p = protocol(out)
    logs = Path(out)/'logs'
    logs.mkdir(exist_ok=True)
    for phase in ('inner', 'outer'):
        if phase == 'outer':
            select(out, p)
        workers = []
        for index, task in enumerate(p['tasks']):
            log = (logs/(phase+'_'+task+'.log')).open('a')
            command = [sys.executable, '-u', '-m', 'improve.tabular_foundation', 'run',
                '--out-dir', str(out), '--phase', phase, '--task', task, '--device', devices[index % len(devices)]]
            workers.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), log, task))
        while any(process.poll() is None for process, _, _ in workers):
            time.sleep(1)
        failed = []
        for process, log, task in workers:
            log.close()
            if process.returncode:
                failed.append(task)
        if failed:
            raise RuntimeError('Inference workers failed: '+','.join(failed)+'; see '+str(logs))
        print('STAGE_DONE', phase, flush=True)
    subprocess.run([sys.executable, '-u', '-m', 'improve.analyze_tabular_foundation',
                    '--out-dir', str(out), '--device', devices[0]], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'run', 'select', 'pipeline'))
    parser.add_argument('--out-dir', default=DEFAULT_OUT)
    parser.add_argument('--reference-dir', default=DEFAULT_REFERENCE)
    parser.add_argument('--checkpoint', default=DEFAULT_CHECKPOINT)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--phase', choices=('inner', 'outer'), default='inner')
    parser.add_argument('--task', choices=('LncDrug', 'MiDrug'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--devices', default='cuda:0,cuda:1')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'pipeline':
        pipeline(Path(args.out_dir).resolve(), args.devices.split(','))
    else:
        out = Path(args.out_dir).resolve()
        p = protocol(out)
        if args.command == 'select':
            select(out, p)
        else:
            for task, cfg, seed in jobs(out, p, args.phase):
                if args.task is None or args.task == task:
                    run_job(out, p, args.phase, task, cfg, seed, args.device)


if __name__ == '__main__':
    main()
