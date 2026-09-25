"""Run the frozen TabDPT (1.3) in-context classifier on the augmented feature banks.

This runner is intentionally dependency-light: it only needs numpy, hashlib and
the official `tabdpt` package (available in the dedicated frontier environment).
It reads the hash-sealed feature banks produced by `foundation_transfer_features`
and writes one score file per (bank, task, seed).  The checkpoint is verified by
SHA-256 before any prediction is produced.
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CHECKPOINT = '/data1/licheng/incrd-hg/data_external/frontier_models_20260914/tabdpt1_3/tabdpt1_3.safetensors'
CHECKPOINT_SHA256 = '97dc3b60bfad6b42ec1a07b7e121d86b0fc7c9fd67c8d2eaac3d815da197eacb'
DEFAULT_INPUTS = 'improve/results/foundation_transfer_20260914/inputs'
DEFAULT_OUT = 'improve/results/foundation_transfer_20260914/tabdpt'
SEEDS = (666, 0, 1)
N_ENSEMBLES = 8
BANKS = ('base36', 'tiered62', 'context76')


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2 ** 20), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs-dir', default=DEFAULT_INPUTS)
    parser.add_argument('--out-dir', default=DEFAULT_OUT)
    parser.add_argument('--task', required=True, choices=('MiDrug', 'LncDrug'))
    parser.add_argument('--device', required=True)
    parser.add_argument('--checkpoint', default=CHECKPOINT)
    args = parser.parse_args()

    if digest(args.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('TabDPT checkpoint hash mismatch')
    inputs, out = Path(args.inputs_dir), Path(args.out_dir)
    manifest = read_json(inputs / 'INPUTS.json')
    if manifest['sha256'] != hashlib.sha256(json.dumps(
            {k: v for k, v in manifest.items() if k != 'sha256'},
            sort_keys=True, separators=(',', ':')).encode()).hexdigest():
        raise ValueError('Unsealed input manifest')
    out.mkdir(parents=True, exist_ok=True)

    import torch
    from tabdpt import TabDPTClassifier
    torch.use_deterministic_algorithms(False)

    for bank in BANKS:
        source = 'tiered62' if bank == 'base36' else bank
        name = '%s_%s.npz' % (source, args.task)
        expected = manifest['files'][name]
        if digest(inputs / name) != expected:
            raise ValueError('Frozen bank changed: ' + name)
        with np.load(inputs / name, allow_pickle=False) as archive:
            x_train = archive['x_train']
            x_val = archive['x_val']
            y_train = archive['y_train']
            y_val = archive['y_val']
            pairs_val = archive['pairs_val']
        width = 36 if bank == 'base36' else (62 if bank == 'tiered62' else 76)
        x_train, x_val = x_train[:, :width], x_val[:, :width]
        for seed in SEEDS:
            dest = out / ('%s_%s_s%d.npz' % (bank, args.task, seed))
            if dest.exists():
                continue
            classifier = TabDPTClassifier(model_weight_path=args.checkpoint, device=args.device,
                                          context_reduction='subsample', compile=False, verbose=False)
            classifier.fit(x_train, y_train)
            probability = classifier.ensemble_predict_proba(
                x_val, n_ensembles=N_ENSEMBLES, seed=seed)[:, 1]
            np.savez_compressed(dest, score=probability, y_val=y_val, pairs_val=pairs_val,
                                bank=bank, task=args.task, seed=seed,
                                n_ensembles=N_ENSEMBLES, checkpoint_sha256=CHECKPOINT_SHA256)
            write_json(dest.with_suffix('.json'), dict(
                created_at_utc=datetime.now(timezone.utc).isoformat(), bank=bank, task=args.task,
                seed=seed, n_ensembles=N_ENSEMBLES, checkpoint_sha256=CHECKPOINT_SHA256,
                feature_file=name, sha256=digest(dest)))
            print('SCORED', bank, args.task, seed, flush=True)
            del classifier


if __name__ == '__main__':
    main()
