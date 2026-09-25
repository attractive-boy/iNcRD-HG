"""Reproduce graph-context selection with the explicit numerical audit adapter.

This orchestration entry freezes the existing audit policy immediately after
inner selection, before this run builds any outer graph-query features.
"""
import argparse
from pathlib import Path
import sys

from improve import tabicl_graph_context as experiment
from improve import tabicl_graph_context_audit as audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--devices', default='cuda:0,cuda:1')
    args = parser.parse_args()
    out = Path(args.out_dir).resolve()
    p = experiment.protocol(out)
    devices = args.devices.split(',')
    experiment.workers([[p['graph_python'], '-u', '-m', 'improve.tabicl_graph_context_features',
        '--out-dir', str(out), '--phase', 'inner', '--task', task] for task in p['tasks']], 'features_inner')
    experiment.workers([[sys.executable, '-u', '-m', 'improve.tabicl_graph_context', 'run',
        '--out-dir', str(out), '--phase', 'inner', '--task', task, '--device', devices[i % len(devices)]]
        for i, task in enumerate(p['tasks'])], 'inner')
    experiment.select(out)
    audit.freeze(out)
    experiment.workers([[p['graph_python'], '-u', '-m', 'improve.tabicl_graph_context_features',
        '--out-dir', str(out), '--phase', 'outer', '--task', task] for task in p['tasks']], 'features_outer')
    for stage in ('resume', 'verify'):
        experiment.workers([[sys.executable, '-u', '-m', 'improve.tabicl_graph_context_audit', stage,
            '--out-dir', str(out), '--task', task, '--device', devices[i % len(devices)]]
            for i, task in enumerate(p['tasks'])], stage)
    audit.report(out)


if __name__ == '__main__':
    main()
