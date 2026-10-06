"""Handle the existing SOAP adoption schema; retain the failed v1 draft."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

root=Path(__file__).resolve().parents[2]
here=Path(__file__).resolve().parent
source=root/'anchor_eigenbasis_validate_v1.py'
target=root/'anchor_eigenbasis_validate_v2.py'
if target.exists():
 raise ValueError('Preserve executor')
s=source.read_text().replace('anchor_eigenbasis_validate_v1','anchor_eigenbasis_validate_v2')
old="""        if adopted and record.get('new_training_steps') != 0:
            raise ValueError('Adoption unexpectedly includes new training')"""
new="""        if adopted:
            if record.get('new_training_steps', 0) != 0:
                raise ValueError('Adoption unexpectedly includes new training')
            if (not record.get('adoption_input_arrays_identical') or
                    record.get('adoption_prediction_max_abs') != 0. or
                    not record.get('source_trace_and_weights_preserved')):
                raise ValueError('Incomplete exact adoption evidence')
            source = Path(record['adopted_from'])
            if base.digest(source) != record['adopted_from_sha256']:
                raise ValueError('Adoption source result changed')
            parent = base.read_json(source)
            shared.verify_artifacts(source.parent, parent)
            for name in ('model.pt', 'optimizer_final.pt', 'trace.json', 'scores.npz'):
                if record['files'][name] != parent['files'][name]:
                    raise ValueError('Adoption training or predictions changed')
            for name in ('training_steps', 'gradient_evaluations', 'initialization_gradients'):
                if record[name] != parent[name]:
                    raise ValueError('Adoption source budget changed')"""
if s.count(old)!=1:
 raise ValueError('Ambiguous repair')
s=s.replace(old,new,1)
target.write_text(s)
q=(here/'run_eigenbasis_partitions_v1.py').read_text().replace('anchor_eigenbasis_validate_v1','anchor_eigenbasis_validate_v2').replace('_V1.json','_V2.json').replace('_v1.log','_v2.log')
(here/'run_eigenbasis_partitions_v2.py').write_text(q)
p=(here/'prepare_eigenbasis_partitions_v1.py').read_text().replace('anchor_eigenbasis_validate_v1','anchor_eigenbasis_validate_v2').replace('EIGENBASIS_ADAM_PARTITION_QUEUE_PROTOCOL_V1','EIGENBASIS_ADAM_PARTITION_QUEUE_PROTOCOL_V2').replace('EIGENBASIS_PARTITION_UNIT_V1','EIGENBASIS_PARTITION_UNIT_V2').replace('run_eigenbasis_partitions_v1','run_eigenbasis_partitions_v2').replace('PARTITION_EXECUTOR_DERIVATION_V1','PARTITION_EXECUTOR_DERIVATION_V2')
(here/'prepare_eigenbasis_partitions_v2.py').write_text(p)
manifest=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),parent_path=str(source),parent_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),derived_path=str(target),derived_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),reason='The completed adoption uses adopted_from/adopted_from_sha256 and exact files, with no new_training_steps field. v1 rejected this before any candidate seal or partition fit. v2 checks the actual schema and byte-identical parent artifacts.',v1_new_partition_fits=0,original_records_changed=False)
(here/'PARTITION_EXECUTOR_DERIVATION_V2.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(manifest)
