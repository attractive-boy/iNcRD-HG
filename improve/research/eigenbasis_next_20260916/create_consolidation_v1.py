"""Build the read-only SOAP consolidation while retaining earlier reports."""
from pathlib import Path
root=Path(__file__).resolve().parents[2]
target=root/'research/eigenbasis_next_20260916/consolidate_eigenbasis_v1.py'
if target.exists():
 raise ValueError('Preserve consolidation source')
s=(root/'research/differential_next_20260916/consolidate_completed_v1.py').read_text()
s=s.replace('from improve import anchor_plain_validate_v3 as validation','from improve import anchor_eigenbasis_validate_v2 as validation')
s=s.replace("family = 'differential_attention'","family = 'eigenbasis_adam'")
s=s.replace("name='差分注意力（作者 V2 运算适配）'","name='SOAP 梯度特征基下逐坐标二阶矩'")
s=s.replace("previous = out / 'CONSOLIDATED_RESULTS_V5_20260916.csv'", "previous = out / 'CONSOLIDATED_RESULTS_V6_20260916.csv'")
s=s.replace("target = out / 'CONSOLIDATED_RESULTS_V6_20260916.csv'", "target = out / 'CONSOLIDATED_RESULTS_V7_20260916.csv'")
s=s.replace('assert len(old) == 30','assert len(old) == 32').replace('assert len(result) == 32','assert len(result) == 34')
s=s.replace("ROOT / 'research/feature_interaction_next_20260916/'\n        'COOPERATIVE_LOCAL_PARTITION_CONSOLIDATION_V1.json'", "ROOT / 'research/differential_next_20260916/'\n        'DIFFERENTIAL_ATTENTION_PARTITION_CONSOLIDATION_V1.json'")
s=s.replace("== assessment['identity_scoring_records'] == 6", "== assessment['identity_scoring_records'] == 9")
s=s.replace("HERE / 'DIFFERENTIAL_ATTENTION_PARTITION_CONSOLIDATION_V1.json'", "HERE / 'EIGENBASIS_ADAM_PARTITION_CONSOLIDATION_V1.json'")
s=s.replace('new_partition_neural_fits=6, new_partition_identity_records=6','new_partition_neural_fits=9, new_partition_identity_records=9')
s=s.replace("old_count['cumulative_fixed_partition_neural_fits'] + 6", "old_count['cumulative_fixed_partition_neural_fits'] + 9")
s=s.replace("decision='Stop this frozen candidate before outer recheck'", "decision='Eligible for a frozen outer recheck' if assessment['additional_development_gate'] else 'Stop this frozen candidate before outer recheck'")
s=s.replace("        r = base.read_json(path)\n        validation.shared.verify_artifacts(path.parent, r)", "        r = base.read_json(path)\n        validation.shared.verify_artifacts(path.parent, r)\n        validation.validate_budget(r, r['recipe']['epochs'], base.read_json(path.parent/'trace.json'))")
start=s.index("if __name__ == '__main__':")
s=s[:start]+"if __name__ == '__main__':\n    counts = historical('eigenbasis_adam', HERE, 'EIGENBASIS')\n    partitions(counts)\n"
target.write_text(s)
print(target)
