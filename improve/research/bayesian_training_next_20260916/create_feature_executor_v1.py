"""Keep the full train/predict/run bodies while narrowing the tested claim."""
from pathlib import Path
import json,hashlib
r=Path(__file__).resolve().parents[2]
source=r/'anchor_variational_lastlayer_experiment_v1.py'
target=r/'anchor_variational_feature_experiment_v1.py'
if target.exists():
 raise ValueError('Preserve derived executor')
s=source.read_text()
start=s.index('OWN = ')
end=s.index('\n\n\ndef prepare',start)
names=('anchor_variational_feature_experiment_v1.py','test_variational_feature_learning_v1.py','variational_feature_learning_v1.py','adopt_variational_feature_v1.py','anchor_variational_lastlayer_experiment_v1.py','test_anchor_variational_lastlayer_experiment_v1.py','variational_lastlayer_graph_v1.py','test_variational_lastlayer_graph_v1.py','sharpness_graph_v1.py','overlap_block_models.py','anchor_priority_graph_v1.py','test_anchor_priority_graph_v1.py','assess_declared_graph.py','graph_prediction_precision.py','residual_graph_experiment.py','nested_anchor_inputs_v2.py')
s=s[:start]+'OWN = '+repr(names)+s[end:]
s=s.replace("family != 'variational_lastlayer'","family != 'variational_feature_learning'")
s=s.replace("default='variational_lastlayer'","default='variational_feature_learning'")
s=s.replace('Full covariance must beat diagonal covariance and MAP on both mean MiDrug metrics.', 'Diagonal variational feature learning must beat no_feature_variance and MAP on both mean MiDrug metrics.')
s=s.replace("primary_and_controls_fixed_before_screening=True, original_test_source_access=False,", "primary_and_controls_fixed_before_new_screening=True, parent_full_and_map_outcomes_previously_seen=True, independent_screen=False, original_test_source_access=False,")
s=s.replace("raise ValueError('Use a new variational classifier study')", "raise ValueError('Use a new variational feature-learning study')")
target.write_text(s)
manifest=dict(parent=str(source),parent_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),derived=str(target),derived_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),training_prediction_run_bodies_unchanged=True,independent_screen=False,correlated_covariance_claim_retained=False)
(r/'research/bayesian_training_next_20260916/FEATURE_EXECUTOR_DERIVATION_V1.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(manifest)
