"""Anchor-aligned predictive integration in the fixed-step partition executor."""
from pathlib import Path
import json,hashlib
r=Path(__file__).resolve().parents[2]
source=r/'anchor_plain_validate_v3.py'
target=r/'anchor_variational_validate_v1.py'
if target.exists():
 raise ValueError('Preserve validator')
s=source.read_text().replace('anchor_plain_validate_v3','anchor_variational_validate_v1').replace('anchor_priority_graph_v1 as experiment','anchor_variational_feature_experiment_v1 as experiment')
s=s.replace("'anchor_priority_graph_v1.py', 'assess_declared_graph.py'", "'anchor_variational_feature_experiment_v1.py', 'assess_declared_graph.py'")
s=s.replace("'differential_attention': ('anchor_priority_differential_attention_v1', experiment)","'variational_feature_learning': ('anchor_priority_variational_feature_learning_v1', experiment)")
s=s.replace("prediction_interface='unconditional_raw_residual'", "prediction_interface='evaluation_reference_conditioned_gaussian_integral'")
s=s.replace("mechanism_family='cooperative' if family == 'cooperative_local' else family,\n        full_screen_previously_observed=(family == 'cooperative_local'),", "mechanism_family='variational_classifier_weight_learning',\n        full_screen_previously_observed=True,independent_screen=False,")
s=s.replace('The reference plus 0.1 mean raw residual formula is frozen.', 'Prediction integrates the binary Gaussian logit with the aligned evaluation reference, subtracts that reference and averages residuals across supports; the 0.1 composition is frozen. A 128-point label-free quadrature reference audits every query.')
s=s.replace("    torch.set_num_threads(4)\n    for split", "    torch.set_num_threads(4)\n    torch.backends.cuda.matmul.allow_tf32 = False\n    torch.backends.cudnn.allow_tf32 = False\n    for split")
s=s.replace("                continue\n            model, _, trace, detail", "                continue\n            if any(dest.iterdir()):\n                raise ValueError('Preserve partial artifacts')\n            model, _, trace, detail")
s=s.replace("            torch.save(model.state_dict(), dest / 'model.pt')", "            if detail['training_steps'] != recipe['epochs'] or detail['gradient_evaluations'] != recipe['epochs']:\n                raise ValueError('Wrong fixed training budget')\n            if any('anchor_composed' in row for row in trace):\n                raise ValueError('Fixed fitting used evaluation labels')\n            torch.save(model.state_dict(), dest / 'model.pt')")
start=s.index("            value = experiment.predict(model, xs, graphs, query, p).cpu().numpy()")
end=s.index("            for actual in (replay, chunks):",start)
s=s[:start]+'''            anchor = torch.as_tensor(reference['anchor'],device=device,dtype=torch.float64)
            components = experiment.predict_components(model,xs,graphs,query,p,anchor)
            value = components['residual'].cpu().numpy()
            replay = experiment.predict(loaded,xs,graphs,query.flip(1),p,anchor.flip(0)).cpu().numpy()[::-1]
            chunks = np.concatenate([experiment.predict(loaded,xs,graphs,query[:,i:i+800],p,anchor[i:i+800]).cpu().numpy()
                                     for i in range(0,query.shape[1],800)])
            numerical_reference = experiment.predict(loaded,xs,graphs,query,p,anchor,128).cpu().numpy()
            integration_error = float(abs(value-numerical_reference).max())
            if integration_error > p['core_parameters']['quadrature_tolerance']:
                raise ValueError('Frozen quadrature is insufficient')
            np.savez_compressed(dest/'posterior_diagnostics.npz',**{k:v.cpu().numpy() for k,v in components.items()},
                                reference_residual=numerical_reference)
'''+s[end:]
s=s.replace("            np.savez_compressed(dest / 'scores.npz', **scores)", "            scores['mean_only_composed'] = scores['anchor']+.1*components['mean_residual'].cpu().numpy()\n            np.savez_compressed(dest / 'scores.npz', **scores)")
s=s.replace("task=task, split=split, kind=kind, seed=cp['model_seed'], recipe=recipe, **detail,", "task=task, split=split, kind=kind, seed=cp['model_seed'], recipe=recipe, **detail,\n                quadrature_reference_max_abs=integration_error,mean_only_is_same_selected_weights=True,")
s=s.replace("('model.pt', 'trace.json', 'scores.npz')","('model.pt', 'trace.json', 'scores.npz', 'posterior_diagnostics.npz')")
s=s.replace("                shared.verify_artifacts(dest, r)\n                if r.get('trained_neural_model', True):", "                shared.verify_artifacts(dest, r)\n                if r['training_steps'] != cp['recipes'][task][kind]['epochs'] or r['gradient_evaluations'] != r['training_steps']:\n                    raise ValueError('Changed training budget')\n                if r['quadrature_reference_max_abs'] > cp['training_parameters']['core_parameters']['quadrature_tolerance']:\n                    raise ValueError('Quadrature tolerance failed')\n                if r.get('trained_neural_model', True):")
target.write_text(s)
manifest=dict(parent=str(source),parent_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),derived=str(target),derived_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),changes='Dedicated variational fit with fixed updates; aligned anchor in reordered/batched prediction; 128-point integration audit without labels; same-weight mean-only diagnostic.')
(r/'research/bayesian_training_next_20260916/VARIATIONAL_PARTITION_EXECUTOR_DERIVATION_V1.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(manifest)
