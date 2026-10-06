"""Derive a new executor without changing any sealed historical implementation."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT/'anchor_stochastic_augmentation_v1.py'
TARGET = ROOT/'anchor_risk_distribution_v1.py'
if TARGET.exists():
    raise ValueError('Preserve executor source')
text = SOURCE.read_text()
original = text


def replace_once(old, new):
    global text
    if text.count(old) != 1:
        raise ValueError('Unexpected template: '+old[:80])
    text = text.replace(old, new)


replace_once("OWN = ('anchor_stochastic_augmentation_v1.py', 'stochastic_augmentation_graph_v1.py', 'test_stochastic_augmentation_v1.py',",
             "OWN = ('anchor_risk_distribution_v1.py', 'risk_distribution_graph_v1.py', 'test_risk_distribution_v1.py',")
replace_once("likelihood='BCE on OOF-reference-plus-residual, plus 0.1 BPR on same-RNA actual training pairs with latent synthetic negative embeddings and 0.05 symmetric InfoNCE on noisy/reversed node views. The synthetic negative retains its original training-pair reference logit; it is a regularizer, not an observed new association.',",
             "likelihood='Global BCE on nested OOF reference plus graph residual; after 100 warmup updates add label-conditional risk distribution matching across four training-drug attribute clusters. Full uses summed eight-RBF empirical MMD; moment_matching uses squared mean and population-variance differences; erm has zero matching penalty. No prediction-time domain input.',")
replace_once("training_budget='400 encoder AdamW updates and 400 detached-feature denoising score-network updates; two backwards per step. Generator parameters are frozen during differentiable reverse-sampling gradients to encoder features. All arms train the same score network and draw the same private random tensors.',",
             "training_budget='400 AdamW updates and 400 backward evaluations, first 100 global-BCE warmup updates, constant learning rate, no optimizer reset. All arms share initialization, dropout stream, support folds and checkpoint budget. Only post-warmup checkpoints at 120 through 400 are eligible.',")
start = text.index('def fit(')
end = text.index('\n\ndef score_arrays', start)
fit = '''def fit(a,p,kind,lr,seed,device,fixed_epochs=None):
    if device != 'cuda:1' and not (device == 'cpu' and p.get('smoke',False)):
        raise ValueError('Formal fitting is restricted to cuda:1')
    if fixed_epochs is not None and set(a)-TRAIN_KEYS:
        raise ValueError('Fixed-step fitting accepts training arrays only')
    if kind not in p['kinds']:
        raise ValueError('Unregistered risk-matching arm')
    if a['anchor_oof'].shape != a['y_train'].shape or not np.isfinite(a['anchor_oof']).all():
        raise ValueError('Invalid nested training reference')
    if device != 'cpu':
        torch.cuda.set_device(device)
    torch.random.default_generator.manual_seed(seed)
    if device != 'cpu':
        torch.cuda.manual_seed(seed)
    training = {k:v for k,v in a.items() if k in TRAIN_KEYS}
    model,xs,graphs,folds = setup(training,p,kind,device)
    if library(p).identity_task(training,p['core_parameters']):
        model.training_final_state = None
        value = None if fixed_epochs is not None else np.zeros(a['pairs_val'].shape[1],dtype=np.float64)
        return model,value,[],dict(epoch=0,seconds=0.,parameters=0,trainable_parameters=0,
            fixed_step_training=fixed_epochs is not None,trained_neural_model=False,training_steps=0,
            gradient_evaluations=0,optimizer_steps=0,matching_active_steps=0,
            identity_preserves_anchor=True,no_new_lnc_training=True,query_outcomes_used_for_offset_fitting=False)
    c = p['core_parameters']
    epochs = p['epochs'] if fixed_epochs is None else int(fixed_epochs)
    if epochs < 1 or (fixed_epochs is None and epochs <= c['warmup_steps']):
        raise ValueError('Historical selection requires post-warmup updates')
    pairs = torch.as_tensor(a['pairs_train']+a['pair_offset'][:,None],device=device)
    labels = torch.as_tensor(a['y_train'],device=device,dtype=torch.float64)
    offset = torch.as_tensor(a['anchor_oof'],device=device,dtype=torch.float64)
    domains = model.drug_domains[torch.as_tensor(a['pairs_train'][1],device=device)]
    query = None if fixed_epochs is not None else torch.as_tensor(a['pairs_val']+a['pair_offset'][:,None],device=device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=p['weight_decay'])
    best,selected,state,trace = -np.inf,epochs,None,[]
    start = time.monotonic()
    for step in range(1,epochs+1):
        fold = (step-1)%len(graphs)
        ids = folds[fold]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        residual = model(xs,graphs[fold],pairs[:,ids])
        risks = F.binary_cross_entropy_with_logits(offset[ids]+residual.double(),labels[ids],reduction='none')
        bce = risks.mean()
        distribution,moment,domain_info = library(p).matching_terms(risks,labels[ids],domains[ids],c)
        weight = c['matching_weight'] if step > c['warmup_steps'] else 0.
        term = distribution if kind == 'full' else moment if kind == 'moment_matching' else distribution*0.
        loss = bce+weight*term
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite risk-matching loss')
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        if not torch.isfinite(gradient):
            raise FloatingPointError('Nonfinite risk-matching gradient')
        optimizer.step()
        row = dict(epoch=step,omitted_fold=fold,query_count=len(ids),loss=float(loss.detach()),
            bce=float(bce.detach()),distribution_penalty=float(distribution.detach()),
            moment_penalty=float(moment.detach()),matching_weight=weight,
            effective_penalty=float((weight*term).detach()),domain_info=domain_info,
            gradient_norm=float(gradient),optimizer_steps=step,gradient_evaluations=step,
            checkpoint_eligible=step > c['warmup_steps'])
        if step > c['warmup_steps'] and (step%p['check_interval']==0 or step==epochs) and query is not None:
            value = predict(model,xs,graphs,query,p).cpu().numpy()
            metrics = common.base.metrics(a['y_val'],a['anchor_'+str(seed)]+p['strong_weight']*value,a['pairs_val'])
            row['anchor_composed'] = metrics
            criterion = (metrics['auc']+metrics['ap'])/2
            if criterion > best:
                best,selected = criterion,step
                state = cpu_copy(model.state_dict())
            print('TRAIN_RISK_DISTRIBUTION',kind,lr,seed,step,round(metrics['auc'],9),round(metrics['ap'],9),flush=True)
        trace.append(row)
    model.training_final_state = cpu_copy(dict(optimizer=optimizer.state_dict(),epoch=epochs,
        drug_domains=model.drug_domains,cpu_rng=torch.random.get_rng_state(),
        cuda_rng=torch.cuda.get_rng_state(device) if device != 'cpu' else None))
    if state is not None:
        model.load_state_dict(state)
    value = None if query is None else predict(model,xs,graphs,query,p).cpu().numpy()
    return model,value,trace,dict(epoch=selected,seconds=time.monotonic()-start,
        parameters=sum(v.numel() for v in model.parameters()),trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
        fixed_step_training=fixed_epochs is not None,trained_neural_model=True,training_steps=epochs,
        gradient_evaluations=epochs,optimizer_steps=epochs,matching_active_steps=max(0,epochs-c['warmup_steps']),
        final_optimizer_state_epoch=epochs,identity_preserves_anchor=False,all_device_seeding_used=False,
        mean_raw_residual=True,query_outcomes_used_for_offset_fitting=False,nested_feature_crossfit=True)
'''
text = text[:start]+fit+text[end:]
ast.parse(text)
old_ast, new_ast = ast.parse(original), ast.parse(text)
def functions(tree, source):
    return {node.name:ast.get_source_segment(source,node) for node in tree.body if isinstance(node,ast.FunctionDef)}
old, new = functions(old_ast,original), functions(new_ast,text)
unchanged = ['library','offset_arrays','setup','predict','cpu_copy','score_arrays','run','select']
if any(old[name] != new[name] for name in unchanged) or 'torch.manual_seed(' in text:
    raise ValueError('Unexpected executor derivation')
TARGET.write_text(text)
report = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
              parent=str(SOURCE),parent_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
              source=str(TARGET),source_sha256=hashlib.sha256(TARGET.read_bytes()).hexdigest(),
              unchanged_function_bodies=unchanged,
              generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              previous_sources_modified=False)
path = HERE/'RISK_EXECUTOR_DERIVATION_V1.json'
with path.open('x') as stream:
    json.dump(report,stream,indent=2)
print(json.dumps(report,indent=2))
