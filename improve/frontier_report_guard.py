"""Report-only guard for empty/single-class diagnostic subsets.

Keep the already frozen experiment implementation and all model states intact.
The original matching helper requires two classes even for diagnostic subsets.
With fewer classes, no group can have both classes, so matching weights are zero
and existing weighted_metrics reports NA. Primary covered comparisons are intact.
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from improve import tabular_foundation as base
from improve import frozen_feature_validation as experiment
from improve.sampling_bias_audit import matched_weights as original_matched_weights


def diagnostic_matched_weights(y,groups):
    y,groups = np.asarray(y),np.asarray(groups)
    if y.ndim!=1 or groups.shape!=y.shape or not set(np.unique(y))<={0,1}:
        raise ValueError('Expected binary labels and aligned diagnostic groups')
    if len(np.unique(y))<2:
        return np.zeros(len(y),np.float64),dict(retained_pairs=0,retained_entities=0,
            reason='No group can contain both classes in an empty/single-class subset')
    return original_matched_weights(y,groups)


def main(args):
    out = Path(args.out_dir)
    p = experiment.protocol(out)
    file = out/'REPORTING_AMENDMENT.json'
    if file.exists():
        raise ValueError('Reporting amendment already recorded')
    calls = []
    def guarded(y,groups):
        weights,metadata = diagnostic_matched_weights(y,groups)
        if 'reason' in metadata:
            calls.append(dict(rows=len(y),classes=np.unique(y).tolist(),groups=len(np.unique(groups))))
        return weights,metadata
    amendment = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=p['sha256'],
        reporting_source_sha256=base.digest(__file__),test_source_sha256=base.digest(Path(__file__).with_name('test_frontier_report_guard.py')),
        reason='The original report stopped on a one-class uncovered diagnostic subset. Matching on such a subset has no eligible entities and its AUC/AP must be NA.',
        correction='Return zero matching weights for fewer than two observed classes; reuse original matching for all two-class subsets.',
        model_predictions_changed=False,model_inference_repeated=False,primary_recipes_changed=False,
        acceptance_thresholds_changed=False,original_report_source_changed=False,original_failure_log='report.log')
    base.write_json(file,amendment)
    saved = experiment.matched_weights
    experiment.matched_weights = guarded
    try:
        experiment.report(args)
    finally:
        experiment.matched_weights = saved
    base.write_json(file,dict(amendment,completed=True,single_class_diagnostic_calls=calls,
        validation_sha256=base.digest(out/'VALIDATION.json')))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir',required=True)
    main(parser.parse_args())
