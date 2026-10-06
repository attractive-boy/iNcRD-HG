"""Create a new version while preserving the already sealed parent executor."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
source = ROOT / 'anchor_preconditioning_experiment_v1.py'
target = ROOT / 'anchor_eigenbasis_experiment_v1.py'
original = source.read_bytes()
assert hashlib.sha256(original).hexdigest() == 'd8f00300f33ac07f5336858b4495439ccfcf7dacedcc305c67b3b26802332bad'
if target.exists():
    raise ValueError('Preserve generated executor')
text = original.decode()
replacements = [
    ('anchor_preconditioning_experiment_v1', 'anchor_eigenbasis_experiment_v1'),
    ('test_anchor_preconditioning_experiment_v1', 'test_anchor_eigenbasis_experiment_v1'),
    ('kronecker_preconditioning', 'eigenbasis_adam'),
    ('preconditioned_graph_v1', 'eigenbasis_graph_v1'),
    ('All four arms', 'All three arms'),
    ('Four declared arms', 'Three declared arms'),
    ('Full must beat scalar_coupling, SOAP and AdamW', 'Full must beat separable_variance and AdamW'),
    ('primary_and_controls_fixed_before_screening=True,',
     'primary_and_controls_fixed_before_new_fits=True, parent_screening_results_seen=True,\n'
     '        this_revision_has_independent_screen=False, baseline_adoption=parent[\'baseline_adoption\'],'),
    ('\n\ndef prepare(out, family, cache, core):',
     '\n\nOWN += (\'kronecker_preconditioning_v1.py\', \'preconditioned_graph_v1.py\',\n'
     '        \'anchor_preconditioning_experiment_v1.py\')\n\ndef prepare(out, family, cache, core):'),
]
applied = []
for old, new in replacements:
    count = text.count(old)
    # The test-module name is already changed by the preceding substring rename.
    if count == 0 and old.startswith('test_anchor_'):
        continue
    if count < 1:
        raise ValueError('Missing exact parent fragment: '+old)
    text = text.replace(old, new)
    applied.append(dict(old=old, new=new, occurrences=count))
text = text.replace('Nested-reference graph residuals with paired Kronecker optimizer budgets.',
                    'SOAP follow-up with a declared separable-variance control and preserved parent fits.')
compile(text, str(target), 'exec')
with target.open('x') as stream:
    stream.write(text)
with (HERE / 'EXECUTOR_DERIVATION_V1.json').open('x') as stream:
    json.dump(dict(source=str(source), source_sha256=hashlib.sha256(original).hexdigest(),
        target=str(target), target_sha256=hashlib.sha256(text.encode()).hexdigest(), replacements=applied,
        parent_source_unchanged=True, training_loop_and_prediction_body_unchanged=True), stream, indent=2)
