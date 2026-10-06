"""Separate queue with explicit adopted/new record accounting."""
from pathlib import Path
h=Path(__file__).resolve().parent
source=h/'run_vbll_queue_v1.py'
target=h/'run_variational_feature_queue_v1.py'
if target.exists():
 raise ValueError('Preserve queue source')
s=source.read_text().replace('anchor_variational_lastlayer_experiment_v1','anchor_variational_feature_experiment_v1').replace("FAMILY = 'variational_lastlayer'","FAMILY = 'variational_feature_learning'").replace('anchor_priority_variational_lastlayer_v1_20260916','anchor_priority_variational_feature_learning_v1_20260916').replace('VBLL_','VARIATIONAL_FEATURE_')
old="""        if r.get('adopted_without_new_training',False):
            raise ValueError('No model adoption is registered')"""
new="""        adopted = r.get('adopted_without_new_training',False)
        if adopted and (r['new_training_steps'] != 0 or r['kind'] not in ('full','map') or r['seed'] != 666
                or base.digest(r['adoption_source_result']) != r['adoption_source_result_sha256']):
            raise ValueError('Invalid adoption evidence')"""
s=s.replace(old,new)
s=s.replace("lr=r['lr'],seed=r['seed'])", "lr=r['lr'],seed=r['seed'],adopted=adopted)")
s=s.replace("            if r['kind'] == 'full' and not any(row['covariance_offdiagonal_norm'] > 0 for row in trace):\n                raise ValueError('Full covariance never trained')", "            if any(row['covariance_offdiagonal_norm'] != 0 for row in trace):\n                raise ValueError('Unregistered within-class covariance')")
s=s.replace('cumulative_new_neural_fits=len(neural),cumulative_identity_scoring_records=len(identity)',"cumulative_new_neural_fits=sum(not row['adopted'] for row in neural),cumulative_identity_scoring_records=len(identity)")
s=s.replace('neural=neural,identity=identity,adopted_neural_fits=0,identity_records_count_as_training=False',"neural=neural,identity=identity,adopted_neural_fits=sum(row['adopted'] for row in neural),\n        adopted_identity_scoring_records=sum(row['adopted'] for row in identity),independent_screen=False,identity_records_count_as_training=False")
s=s.replace("        adopted_neural_fits=0,partition_verification_pending=bool(result['declared_gate']),", "        adopted_neural_fits=counts['adopted_neural_fits'],adopted_identity_records=counts['adopted_identity_scoring_records'],\n        independent_screen=False,partition_verification_pending=bool(result['declared_gate']),")
target.write_text(s)
print(target)
