from pathlib import Path
root=Path(__file__).resolve().parents[2]
target=root/'research/bayesian_training_next_20260916/stochastic_smoke_v1.py'
if target.exists():
 raise ValueError('Preserve smoke source')
s=(root/'research/preconditioning_next_20260916/preconditioning_smoke_v1.py').read_text()
s=s.replace('anchor_preconditioning_experiment_v1','anchor_stochastic_augmentation_v1').replace('preconditioned_graph_v1','stochastic_augmentation_graph_v1')
s=s.replace('from improve import kronecker_preconditioning_v1 as optimization\n','')
s=s.replace('import gc\n','import gc\nimport hashlib\n')
s=s.replace('kronecker_preconditioning','stochastic_augmentation').replace('PRECONDITIONING','STOCHASTIC_AUGMENTATION')
s=s.replace("core_parameters=core.parameters(), optimizer_parameters=optimization.parameters(),", "core_parameters=core.parameters(), private_rng_offset=29015,")
s=s.replace("core_parameters=core.parameters(), optimizer_parameters=optimization.parameters(),", "core_parameters=core.parameters(),")
s=s.replace('epochs=25','epochs=5').replace('fixed_epochs=25','fixed_epochs=5')
s=s.replace("optimizer_final.pt","training_final.pt").replace('model.optimization_final_state','model.training_final_state')
s=s.replace("            dest = OUT / task / kind", "            rng_hash = hashlib.sha256(torch.cuda.get_rng_state('cuda:1').numpy().tobytes()).hexdigest()\n            dest = OUT / task / kind")
start=s.index("                if detail['gradient_evaluations'] != 26")
end=s.index("                if any('anchor_composed' in row for row in trace):",start)
s=s[:start]+'''                if detail['gradient_evaluations'] != 10 or detail['training_steps'] != 5:
                    raise ValueError('Unmatched score/encoder gradient budget')
                state = torch.load(dest/'training_final.pt',map_location='cpu',weights_only=True)
                for group in ('encoder_optimizer','score_optimizer'):
                    if not state[group]['state'] or any(int(item['step']) != 5 for item in state[group]['state'].values()):
                        raise ValueError('Saved optimizer has inconsistent update budget')
                private_hash = hashlib.sha256(state['private_rng'].numpy().tobytes()).hexdigest()
'''+s[end:]
s=s.replace("            record = base.seal(dict(task=task, kind=kind, **detail,", "            record = base.seal(dict(task=task, kind=kind, **detail,\n                dropout_rng_sha256=rng_hash,private_rng_sha256=private_hash if task == 'MiDrug' else None,")
s=s.replace("Path(core.__file__), Path(optimization.__file__), Path(experiment.__file__)", "Path(core.__file__), Path(experiment.__file__)")
s=s.replace('new_short_neural_fits=4, identity_replays=4','new_short_neural_fits=3, identity_replays=3')
s=s.replace("    result = base.seal(dict(created_at_utc=", "    neural = [r for r in records if r['task'] == 'MiDrug']\n    if len({r['dropout_rng_sha256'] for r in neural}) != 1 or len({r['private_rng_sha256'] for r in neural}) != 1:\n        raise ValueError('Matched arms consumed different random-number streams')\n    result = base.seal(dict(created_at_utc=")
target.write_text(s)
print(target)
