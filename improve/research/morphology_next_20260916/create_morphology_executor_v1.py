"""Derive an external-morphology executor without modifying frozen executors."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'anchor_contextual_rule_v1.py'
DEST = ROOT / 'anchor_morphological_alignment_v1.py'
REPORT = HERE / 'MORPHOLOGY_EXECUTOR_DERIVATION_V1.json'
if DEST.exists() or REPORT.exists():
    raise ValueError('Preserve executor derivation')
original = SOURCE.read_text()
text = original.replace('anchor_contextual_rule_v1.py', 'anchor_morphological_alignment_v1.py')
text = text.replace('contextual_rule_graph_v1.py', 'morphological_alignment_v1.py')
text = text.replace('test_contextual_rule_v1.py', 'test_morphological_alignment_v1.py')
text = text.replace("'sharpness_graph_v1.py', ", "'morphology_inputs_v1.py', ")
text = text.replace('private_rng_offset=29015', 'private_rng_offset=32017')
tree = ast.parse(text)
functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
lines = text.splitlines(keepends=True)
FIT = '''def fit(a,p,kind,lr,seed,device,fixed_epochs=None):
    if device != 'cuda:1' and not (device == 'cpu' and p.get('smoke',False)):
        raise ValueError('Formal fitting is restricted to cuda:1')
    if fixed_epochs is not None and set(a)-TRAIN_KEYS:
        raise ValueError('Fixed-step fitting accepts training arrays only')
    if kind not in p['kinds']:
        raise ValueError('Unregistered morphology arm')
    if a['anchor_oof'].shape != a['y_train'].shape or not np.isfinite(a['anchor_oof']).all():
        raise ValueError('Invalid nested training reference')
    if device != 'cpu':
        torch.cuda.set_device(device)
    torch.random.default_generator.manual_seed(seed)
    if device != 'cpu':
        torch.cuda.manual_seed(seed)
    training = {k:v for k,v in a.items() if k in TRAIN_KEYS}
    model,xs,graphs,folds = setup(training,p,kind,device)
    core = library(p)
    if core.identity_task(training,p['core_parameters']):
        model.training_final_state = None
        value = None if fixed_epochs is not None else np.zeros(a['pairs_val'].shape[1],dtype=np.float64)
        return model,value,[],dict(epoch=0,seconds=0.,parameters=0,trainable_parameters=0,
            fixed_step_training=fixed_epochs is not None,trained_neural_model=False,training_steps=0,
            gradient_evaluations=0,optimizer_steps=0,dti_pairs_per_update=0,
            identity_preserves_anchor=True,no_new_lnc_training=True,query_outcomes_used_for_offset_fitting=False)
    c = p['core_parameters']
    epochs = p['epochs'] if fixed_epochs is None else int(fixed_epochs)
    if epochs < 1:
        raise ValueError('At least one update is required')
    pairs = torch.as_tensor(a['pairs_train']+a['pair_offset'][:,None],device=device)
    labels = torch.as_tensor(a['y_train'],device=device,dtype=torch.float64)
    offset = torch.as_tensor(a['anchor_oof'],device=device,dtype=torch.float64)
    query = None if fixed_epochs is not None else torch.as_tensor(a['pairs_val']+a['pair_offset'][:,None],device=device)
    covered = model.covered(xs,pairs)
    eligible_folds = [ids[covered[ids]] for ids in folds]
    if any(len(ids)==0 for ids in eligible_folds) or any(graph is not None for graph in graphs):
        raise ValueError('Empty covered training fold or forbidden association graph')
    generator = torch.Generator(device=device)
    generator.manual_seed(seed+p['private_rng_offset'])
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=p['weight_decay'])
    best,selected,state,trace = -np.inf,epochs,None,[]
    multiplier = 0. if kind=='no_alignment' else c['alignment_weight']
    start = time.monotonic()
    for step in range(1,epochs+1):
        fold = (step-1)%len(folds)
        ids = eligible_folds[fold]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        encoded = model.encode(xs)
        residual = model.residual(encoded,xs,pairs[:,ids])
        bce = F.binary_cross_entropy_with_logits(offset[ids]+residual,labels[ids])
        align,detail = core.alignment_loss(encoded,xs,c,generator)
        loss = bce+multiplier*align
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite morphological-alignment loss')
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        if not torch.isfinite(gradient):
            raise FloatingPointError('Nonfinite morphological-alignment gradient')
        optimizer.step()
        row = dict(epoch=step,training_fold=fold,covered_query_count=len(ids),
            full_fold_query_count=len(folds[fold]),positive_queries=int(labels[ids].sum()),
            loss=float(loss.detach()),bce=float(bce.detach()),alignment_loss=float(align.detach()),
            alignment_multiplier=multiplier,**detail,gradient_norm=float(gradient),
            optimizer_steps=step,gradient_evaluations=step,association_graph_used=False,
            maximum_absolute_residual=float(residual.detach().abs().max()))
        if (step%p['check_interval']==0 or step==epochs) and query is not None:
            value = predict(model,xs,graphs,query,p).cpu().numpy()
            metrics = common.base.metrics(a['y_val'],a['anchor_'+str(seed)]+p['strong_weight']*value,a['pairs_val'])
            row['anchor_composed'] = metrics
            criterion = (metrics['auc']+metrics['ap'])/2
            if criterion > best:
                best,selected = criterion,step
                state = cpu_copy(model.state_dict())
            print('TRAIN_MORPHOLOGY',kind,lr,seed,step,round(metrics['auc'],9),round(metrics['ap'],9),flush=True)
        trace.append(row)
    model.training_final_state = cpu_copy(dict(optimizer=optimizer.state_dict(),epoch=epochs,
        private_rng=generator.get_state(),input_manifest_sha256=c['input_manifest_sha256'],
        covered_queries_per_fold=[len(ids) for ids in eligible_folds],association_graph_used=False))
    if state is not None:
        model.load_state_dict(state)
    value = None if query is None else predict(model,xs,graphs,query,p).cpu().numpy()
    return model,value,trace,dict(epoch=selected,seconds=time.monotonic()-start,
        parameters=sum(v.numel() for v in model.parameters()),trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
        fixed_step_training=fixed_epochs is not None,trained_neural_model=True,training_steps=epochs,
        gradient_evaluations=epochs,optimizer_steps=epochs,dti_pairs_per_update=c['dti_batch'],
        final_optimizer_state_epoch=epochs,identity_preserves_anchor=False,all_device_seeding_used=False,
        mean_raw_residual=True,query_outcomes_used_for_offset_fitting=False,association_graph_used=False,
        external_positive_dti_pairs=xs['dti_pairs'].shape[1],covered_training_queries=int(covered.sum()),
        uncovered_pairs_exactly_preserve_anchor=True)
'''
node = functions['fit']
lines[node.lineno-1:node.end_lineno] = [FIT]
text = ''.join(lines)
tree = ast.parse(text)
prepare = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'prepare')
replacements = {
    'likelihood': 'BCE on covered nested-OOF reference queries plus a bounded learned morphological residual, jointly with symmetric multi-positive contrastive loss on the independent author-released positive image-DTI table. Each update uses one rotating project training fold; no association graph or project attributes enter the phenotype encoder.',
    'training_budget': '400 AdamW updates and 400 backward evaluations. Every arm encodes the same frozen external profiles and draws the same 256 positive DTI indices per step. no_alignment computes the auxiliary loss with multiplier zero; shuffled_rna_targets changes only the fixed RNA-to-gene identity mapping. All twenty checkpoints are evaluated.',
    'prediction': 'MiDrug: unchanged evaluation reference + 0.1 bounded morphology residual. Pairs lacking a measured drug profile or functional RNA target have exactly zero residual. Five support slots contain no graph and produce identical phenotype predictions. LncDrug: exact reference identity without training.',
}
segments = []
for node in ast.walk(prepare):
    if isinstance(node, ast.keyword) and node.arg in replacements:
        value = node.value
        if not isinstance(value, ast.Constant):
            raise ValueError('Expected literal protocol description')
        segments.append((value.lineno, value.col_offset, value.end_lineno, value.end_col_offset, repr(replacements[node.arg])))
lines = text.splitlines(keepends=True)
for start, col, end, end_col, replacement in sorted(segments, reverse=True):
    if start != end:
        raise ValueError('Protocol literal spans lines')
    lines[start-1] = lines[start-1][:col] + replacement + lines[start-1][end_col:]
text = ''.join(lines)
ast.parse(text)
DEST.write_text(text)
def bodies(source):
    rows = source.splitlines(keepends=True)
    return {node.name: ''.join(rows[node.lineno-1:node.end_lineno]) for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
old, new = bodies(original), bodies(text)
shared = ['library', 'offset_arrays', 'setup', 'predict', 'cpu_copy', 'score_arrays', 'run', 'select']
if any(old[name] != new[name] for name in shared):
    raise ValueError('Unrelated executor behavior changed')
REPORT.write_text(json.dumps(dict(source=str(SOURCE), source_sha256=hashlib.sha256(original.encode()).hexdigest(),
    output=str(DEST), output_sha256=hashlib.sha256(text.encode()).hexdigest(), unchanged_function_bodies=shared,
    replaced_fit=True, independent_external_morphology=True), indent=2) + '\n')
print('MORPHOLOGY_EXECUTOR_DERIVED', flush=True)
