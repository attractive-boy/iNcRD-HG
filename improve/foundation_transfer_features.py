"""Reconstruct the frozen evidence/context-augmented feature banks for foundation-model transfer.

The augmented banks reuse the accepted pathway-context encoders without refitting
anything: the same 36 graph columns plus either the 26 tiered evidence columns
(62 total) or the 14 DepMap expression-context columns (76 total).  Nothing here
reads validation labels or the reserved test arrays.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil

import numpy as np

from improve import tabular_foundation as base
from improve import tabicl_graph_context as graph
from improve import tabicl_context_program_modules as feature


OWN = ('foundation_transfer_features.py',)
DEFAULT_INPUTS = 'improve/results/tabicl_pathway_context_inputs_20260913'
DEFAULT_PARENT = 'improve/results/tabicl_graph_context_20260912'
DEFAULT_OUT = 'improve/results/foundation_transfer_20260914/inputs'
TASKS = ('MiDrug', 'LncDrug')
BANKS = ('tiered62', 'context76')


def sources():
    return {name: base.digest(Path(__file__).with_name(name)) for name in OWN}


def build(args):
    out, inputs, parent = [Path(x).resolve() for x in (args.out_dir, args.inputs_dir, args.parent_dir)]
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new input directory')
    gp = graph.protocol(parent)
    manifest = base.read_json(inputs / 'INPUTS.json')
    for name, expected in manifest['files'].items():
        if base.digest(inputs / name) != expected:
            raise ValueError('Frozen context input changed: ' + name)
    if manifest['labels_read']:
        raise ValueError('External feature construction used labels')
    encoder = feature.ContextPrograms.load(inputs)
    out.mkdir(parents=True)
    files = {}
    for task in TASKS:
        a = graph.features(parent, gp, task, 'inner')
        prefix = 'raw' if gp['input_configs'][task]['input_view'] == 'raw' else 'x'
        base_graph_train, base_graph_val = a[prefix + '_train'], a[prefix + '_val']
        for bank, variant in (('tiered62', 'tiered'), ('context76', 'context')):
            extra_train = encoder.transform(task, a['pairs_train'], variant).astype(np.float32)
            extra_val = encoder.transform(task, a['pairs_val'], variant).astype(np.float32)
            x_train = np.concatenate([base_graph_train, extra_train], axis=1).astype(np.float32)
            x_val = np.concatenate([base_graph_val, extra_val], axis=1).astype(np.float32)
            if not np.isfinite(x_train).all() or not np.isfinite(x_val).all():
                raise ValueError('Non-finite augmented features')
            name = '%s_%s.npz' % (bank, task)
            np.savez_compressed(out / name, x_train=x_train, x_val=x_val,
                                y_train=a['y_train'], y_val=a['y_val'],
                                pairs_train=a['pairs_train'], pairs_val=a['pairs_val'])
            files[name] = base.digest(out / name)
            print('BANK', name, x_train.shape, x_val.shape, flush=True)
    doc = base.seal(dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        tasks=list(TASKS), banks=list(BANKS), seeds=[666, 0, 1],
        graph_parent=str(parent), graph_parent_protocol_sha256=gp['sha256'],
        inputs_dir=str(inputs), input_manifest_sha256=base.digest(inputs / 'INPUTS.json'),
        graph_columns=36, tiered_extra_columns=26, context_extra_columns=14,
        base_input_view={'MiDrug': 'raw', 'LncDrug': 'x'},
        source_sha256=sources(), files=files,
        label_policy='Only frozen support/validation pairs and labels are copied; no new labels are read.',
        limitations=['Historical inner partition repeatedly used in development.',
                     'The augmented columns are annotation-derived, not experimentally measured function.',
                     'No outer/test/source-holdout array is touched.']))
    base.write_json(out / 'INPUTS.json', doc)
    (out / 'source_snapshot').mkdir()
    for name in OWN:
        shutil.copy2(Path(__file__).with_name(name), out / 'source_snapshot' / name)
    print('INPUTS_FROZEN', doc['sha256'], flush=True)


def verify(directory):
    directory = Path(directory)
    doc = base.check_seal(base.read_json(directory / 'INPUTS.json'))
    for name, expected in doc['files'].items():
        if base.digest(directory / name) != expected:
            raise ValueError('Frozen feature bank changed: ' + name)
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', default=DEFAULT_OUT)
    parser.add_argument('--inputs-dir', default=DEFAULT_INPUTS)
    parser.add_argument('--parent-dir', default=DEFAULT_PARENT)
    args = parser.parse_args()
    build(args)


if __name__ == '__main__':
    main()
