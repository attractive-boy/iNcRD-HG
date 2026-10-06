"""Derive a separate optimizer-aware partition executor without editing ancestors."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parents[2]
source = root / 'anchor_plain_validate_v3.py'
target = root / 'anchor_eigenbasis_validate_v1.py'
if target.exists():
    raise ValueError('Preserve the derived executor')
s = source.read_text()
changes = [
    ('Version 3: fixed-partition validation of declared priority graph mechanisms.', 'Fixed-partition validation of the registered SOAP optimizer hypothesis.'),
    ('from improve import anchor_priority_graph_v1 as experiment', 'from improve import anchor_eigenbasis_experiment_v1 as experiment'),
    ("OWN = ('anchor_plain_validate_v3.py', 'test_anchor_plain_validate_v3.py', 'graph_mechanism_validate.py',\n       'anchor_priority_graph_v1.py', 'assess_declared_graph.py')", "OWN = ('anchor_eigenbasis_validate_v1.py', 'test_anchor_eigenbasis_validate_v1.py', 'graph_mechanism_validate.py',\n       'anchor_eigenbasis_experiment_v1.py', 'assess_declared_graph.py')"),
    ("'differential_attention': ('anchor_priority_differential_attention_v1', experiment)", "'eigenbasis_adam': ('anchor_priority_eigenbasis_adam_v1', experiment)"),
    ("if record['task'] != 'MiDrug' or record['training_steps'] != 400 or record['gradient_evaluations'] != 400:", "if record['task'] != 'MiDrug' or record['training_steps'] != 400 or record['gradient_evaluations'] != 401 or record.get('initialization_gradients') != 1:"),
    ("mechanism_family='cooperative' if family == 'cooperative_local' else family,\n        full_screen_previously_observed=(family == 'cooperative_local'),", "mechanism_family='matrix_preconditioning',\n        full_screen_previously_observed=True, independent_screen=False,\n        this_family_partition_predictions_previously_observed=False,"),
    ("training_policy='Fixed median historical steps;", "training_policy='One initialization gradient without a parameter update followed by fixed median historical parameter updates;"),
    ("    torch.set_num_threads(4)\n    for split", "    torch.set_num_threads(4)\n    torch.backends.cuda.matmul.allow_tf32 = False\n    torch.backends.cudnn.allow_tf32 = False\n    for split"),
    ("                shared.verify_artifacts(dest, r)\n                continue\n            model", "                shared.verify_artifacts(dest, r)\n                validate_budget(r, recipe['epochs'], base.read_json(dest / 'trace.json'))\n                continue\n            if any(dest.iterdir()):\n                raise ValueError('Preserve partial partition artifacts: ' + str(dest))\n            model"),
    ("            torch.save(model.state_dict(), dest / 'model.pt')\n            base.write_json", "            validate_budget(detail, recipe['epochs'], trace)\n            if any('anchor_composed' in row for row in trace):\n                raise ValueError('Fixed-step fit accessed evaluation labels')\n            torch.save(model.state_dict(), dest / 'model.pt')\n            torch.save(model.optimization_final_state, dest / 'optimizer_final.pt')\n            base.write_json"),
    ("for n in ('model.pt', 'trace.json', 'scores.npz')", "for n in ('model.pt', 'optimizer_final.pt', 'trace.json', 'scores.npz')"),
    ("                shared.verify_artifacts(dest, r)\n                if r.get('trained_neural_model', True):", "                shared.verify_artifacts(dest, r)\n                validate_budget(r, cp['recipes'][task][kind]['epochs'], base.read_json(dest / 'trace.json'))\n                if r.get('trained_neural_model', True):")]
for old, new in changes:
    if s.count(old) != 1:
        raise ValueError('Ambiguous derivation: '+old)
    s = s.replace(old,new,1)
helper = '''def validate_budget(record, updates, trace):
    if record.get('trained_neural_model') is False:
        if updates != 0 or trace or any(record.get(k) != 0 for k in
                ('epoch', 'parameters', 'trainable_parameters', 'training_steps',
                 'gradient_evaluations', 'initialization_gradients')):
            raise ValueError('Identity budget contains training')
        return
    if (record.get('trained_neural_model') is not True or updates < 1 or
            record['training_steps'] != updates or record['gradient_evaluations'] != updates+1 or
            record.get('initialization_gradients') != 1 or len(trace) != updates+1):
        raise ValueError('Optimizer initialization/update budget mismatch')
    if not trace[0]['initialization_only'] or any(row['initialization_only'] for row in trace[1:]):
        raise ValueError('Wrong initialization trace')
    for step, row in enumerate(trace):
        if (row['epoch'] != step or not row['state_finite'] or
                row['parameter_updates_min'] != step or row['parameter_updates_max'] != step or
                row['parameter_gradient_calls_min'] != step+1 or row['parameter_gradient_calls_max'] != step+1):
            raise ValueError('Invalid paired optimizer trace')
    if not record['final_optimizer_diagnostics']['state_finite']:
        raise ValueError('Nonfinite optimizer state')


'''
s = s.replace('def historical_counts(records):',helper+'def historical_counts(records):',1)
target.write_text(s)
manifest = dict(parent_path=str(source),parent_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                derived_path=str(target),derived_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                modifications='Dedicated SOAP fit, initialization-gradient accounting, optimizer-state artifacts, fixed-step boundary and partial-artifact guards; no partition adoption.')
(root/'research/eigenbasis_next_20260916/PARTITION_EXECUTOR_DERIVATION_V1.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(manifest)
