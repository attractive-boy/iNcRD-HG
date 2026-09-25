"""Re-test the frozen path hypothesis after diagnosing a dead ReLU readout.

This is a repair experiment within the BioPathNet direction, not a new method
count. Only the hidden readout activation changes. Existing runs stay frozen.
"""
import argparse
import json
from pathlib import Path
import shutil

import torch
from torch import nn

from improve import biopath_experiment as experiment
from improve import biopath_models as original


OUT = experiment.base.ROOT / 'results' / 'biopath_head_recovery_20260915'
OriginalResidual = original.PathResidual


class LivePathResidual(OriginalResidual):
    """Preserve the zero starting correction without a dead negative half-axis."""

    def __init__(self, relations, width=32, depth=6):
        super().__init__(relations, width, depth)
        self.mlp[1] = nn.LeakyReLU(negative_slope=0.1)


def prepare(out):
    out = Path(out)
    experiment.prepare(out)
    p = json.loads((out / 'protocol.json').read_text())
    p['amendment'] = {
        'reason': 'All six previous validation corrections were constant; sampled '
                  'readout activations were zero (or effectively zero).',
        'diagnostic': str(experiment.OUT / 'POSTHOC_HEAD_DIAGNOSTIC.json'),
        'diagnostic_sha256': experiment.base.digest(
            experiment.OUT / 'POSTHOC_HEAD_DIAGNOSTIC.json'),
        'change': 'Replace only readout ReLU with LeakyReLU(0.1). No propagation, '
                  'data, objective, optimizer, training budget, or gate changes.',
        'scope': 'Same path-reasoning hypothesis; post-hoc recovery on reused '
                 'development data, not a seventh independent technique.',
    }
    p['architecture'] += ' Recovery uses LeakyReLU(0.1) in the readout only.'
    p['created_by'] = 'improve.biopath_head_recovery'
    for name in ('biopath_head_recovery.py', 'test_biopath_head_recovery.py'):
        p['source_sha256'][name] = experiment.base.digest(experiment.base.ROOT / name)
        shutil.copy2(experiment.base.ROOT / name, out / 'source_snapshot' / name)
    experiment.base.write_json(out / 'protocol.json', p)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'run', 'report'])
    parser.add_argument('--out-dir', default=str(OUT))
    parser.add_argument('--task', choices=experiment.base.TASKS)
    parser.add_argument('--kind', choices=experiment.KINDS)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args.out_dir)
    elif args.command == 'run':
        # The frozen runner resolves this class at fit and replay time. The new
        # launcher is hashed and archived by this experiment's own protocol.
        original.PathResidual = LivePathResidual
        experiment.run(args.out_dir, args.task, args.device, args.kind)
    else:
        experiment.report(args.out_dir)


if __name__ == '__main__':
    main()
