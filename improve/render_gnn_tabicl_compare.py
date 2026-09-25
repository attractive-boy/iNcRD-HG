"""Repair a frozen report's display without changing its training sources.

Pandas Series.view shadows the input condition column in the frozen renderer.
Rebuild that table from verified CSV records and retain the original report.
"""
import argparse
from pathlib import Path

import pandas as pd

from improve import gnn_tabicl_compare as comparison
from improve import tabular_foundation as base


def render(out):
    out=Path(out);p=comparison.protocol(out)
    validation=base.read_json(out/'VALIDATION.json')
    if not validation['passed'] or validation['protocol_sha256']!=p['sha256']:
        raise ValueError('A complete verified comparison is required')
    for name,expected in validation['artifacts'].items():
        if base.digest(out/name)!=expected:
            raise ValueError('Changed comparison artifact: '+name)
    report=out/'REPORT.md'
    original=report.read_text()
    header='|任务|模型|预算|输入|AUC|AP|药物宏 AP|'
    next_section='**配对变化（均值；推理／初始化种子不代表独立划分）。**'
    intro,body=original.split(header,1)
    _,tail=body.split(next_section,1)
    summary=pd.read_csv(out/'summary.csv')
    lines=[header,'|---|---|---|---|---:|---:|---:|']
    for row in summary.to_dict('records'):
        lines.append('|{task}|{learner}|{budget}|{view}|{auc_mean:.6f}|{ap_mean:.6f}|{macro_drug_ap_mean:.6f}|'.format(**row))
    revised=intro+'\n'.join(lines)+'\n\n'+next_section+tail
    if '<bound method' in revised or len(lines)!=len(summary)+2:
        raise ValueError('Invalid comparison table rendering')
    archive=out/'REPORT_frozen_renderer.md'
    if not archive.exists():
        archive.write_text(original)
    elif '<bound method' in original and archive.read_text()!=original:
        raise ValueError('Original frozen rendering differs from its archive')
    report.write_text(revised)
    for name in ('REPORT.md','REPORT_frozen_renderer.md'):
        validation['artifacts'][name]=base.digest(out/name)
    validation['report_rendering']=dict(source=str(Path(__file__).resolve()),
        source_sha256=base.digest(Path(__file__)),rows=len(summary),
        reason='Read the view column by key instead of the pandas Series.view method.',
        training_sources_unchanged=True,metrics_and_predictions_unchanged=True)
    base.write_json(out/'VALIDATION.json',validation)
    print('REPORT_RENDERED',len(summary),'rows; numerical artifacts unchanged',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',default=comparison.DEFAULT_OUT)
    render(parser.parse_args().out_dir)
