"""New SLogic-inspired executor; preserve all previous sealed experiments."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARENT = ROOT/'anchor_risk_distribution_v1.py'
TARGET = ROOT/'anchor_contextual_rule_v1.py'
if TARGET.exists():
    raise ValueError('Preserve contextual-rule executor')
original = PARENT.read_text()
text = original


def replace_once(old,new):
    global text
    if text.count(old)!=1:
        raise ValueError('Unexpected executor template: '+old[:90])
    text = text.replace(old,new)


replace_once("OWN = ('anchor_risk_distribution_v1.py', 'risk_distribution_graph_v1.py', 'test_risk_distribution_v1.py',",
             "OWN = ('anchor_contextual_rule_v1.py', 'contextual_rule_graph_v1.py', 'test_contextual_rule_v1.py',")
replace_once("likelihood='Global BCE on nested OOF reference plus graph residual; after 100 warmup updates add label-conditional risk distribution matching across four training-drug attribute clusters. Full uses summed eight-RBF empirical MMD; moment_matching uses squared mean and population-variance differences; erm has zero matching penalty. No prediction-time domain input.',",
             "likelihood='BCE on nested OOF reference plus a learned rule-potential residual, plus 0.1 margin ranking between grounded and non-grounded locally applicable rules for actual positive queries in the omitted training fold. All symbolic counts, rule confidence and context graphs are reconstructed after deleting that entire fold.',")
replace_once("training_budget='400 AdamW updates and 400 backward evaluations, first 100 global-BCE warmup updates, constant learning rate, no optimizer reset. All arms share initialization, dropout stream, support folds and checkpoint budget. Only post-warmup checkpoints at 120 through 400 are eligible.',",
             "training_budget='400 AdamW updates and 400 backward evaluations. Full and global_rules share initialization, dropout stream, 256 private-RNG rule triplets per update, symbolic supports, optimizer, and all twenty checkpoint evaluations. Global_rules replaces head-specific graph contexts with their common mean; all other operations are identical.',")
start = text.index('def fit(')
end = text.index('\n\ndef score_arrays',start)
fit = '''def fit(a,p,kind,lr,seed,device,fixed_epochs=None):
    if device != 'cuda:1' and not (device == 'cpu' and p.get('smoke',False)):
        raise ValueError('Formal fitting is restricted to cuda:1')
    if fixed_epochs is not None and set(a)-TRAIN_KEYS:
        raise ValueError('Fixed-step fitting accepts training arrays only')
    if kind not in p['kinds']:
        raise ValueError('Unregistered contextual-rule arm')
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
            gradient_evaluations=0,optimizer_steps=0,rule_triplets_per_update=0,
            identity_preserves_anchor=True,no_new_lnc_training=True,query_outcomes_used_for_offset_fitting=False)
    c = p['core_parameters']
    epochs = p['epochs'] if fixed_epochs is None else int(fixed_epochs)
    if epochs < 1:
        raise ValueError('At least one update is required')
    pairs = torch.as_tensor(a['pairs_train']+a['pair_offset'][:,None],device=device)
    local_heads = torch.as_tensor(a['pairs_train'][0],device=device)
    local_tails = torch.as_tensor(a['pairs_train'][1],device=device)
    labels = torch.as_tensor(a['y_train'],device=device,dtype=torch.float64)
    offset = torch.as_tensor(a['anchor_oof'],device=device,dtype=torch.float64)
    query = None if fixed_epochs is not None else torch.as_tensor(a['pairs_val']+a['pair_offset'][:,None],device=device)
    pools = [library(p).rule_pool(training,graph,fold,device) for fold,graph in enumerate(graphs)]
    generator = torch.Generator(device=device)
    generator.manual_seed(seed+p['private_rng_offset'])
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=p['weight_decay'])
    best,selected,state,trace = -np.inf,epochs,None,[]
    start = time.monotonic()
    for step in range(1,epochs+1):
        fold = (step-1)%len(graphs)
        ids,graph = folds[fold],graphs[fold]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        rule_scores = model.rule_scores(graph)
        residual = model.residual(rule_scores,graph,pairs[:,ids])
        bce = F.binary_cross_entropy_with_logits(offset[ids]+residual,labels[ids])
        selected_queries,positive,negative = library(p).sample_rules(pools[fold],c['triplet_batch'],generator)
        heads,tails = local_heads[selected_queries],local_tails[selected_queries]
        valid = (bool(torch.all(labels[selected_queries]==1))
                 and bool(torch.all(graph['grounding'][heads,positive,tails]>0))
                 and bool(torch.all(graph['grounding'][heads,negative,tails]==0)))
        if not valid:
            raise ValueError('Incorrect positive or negative rule supervision')
        margin = F.relu(c['margin']-rule_scores[heads,positive]+rule_scores[heads,negative]).mean()
        loss = bce+c['ranking_weight']*margin
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite contextual-rule loss')
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
        if not torch.isfinite(gradient):
            raise FloatingPointError('Nonfinite contextual-rule gradient')
        optimizer.step()
        row = dict(epoch=step,omitted_fold=fold,query_count=len(ids),loss=float(loss.detach()),
            bce=float(bce.detach()),ranking_loss=float(margin.detach()),triplet_count=len(selected_queries),
            eligible_positive_queries=len(pools[fold]['queries']),rule_supervision_verified=valid,
            gradient_norm=float(gradient),optimizer_steps=step,gradient_evaluations=step,
            maximum_absolute_residual=float(residual.detach().abs().max()))
        if (step%p['check_interval']==0 or step==epochs) and query is not None:
            value = predict(model,xs,graphs,query,p).cpu().numpy()
            metrics = common.base.metrics(a['y_val'],a['anchor_'+str(seed)]+p['strong_weight']*value,a['pairs_val'])
            row['anchor_composed'] = metrics
            criterion = (metrics['auc']+metrics['ap'])/2
            if criterion > best:
                best,selected = criterion,step
                state = cpu_copy(model.state_dict())
            print('TRAIN_CONTEXTUAL_RULE',kind,lr,seed,step,round(metrics['auc'],9),round(metrics['ap'],9),flush=True)
        trace.append(row)
    model.training_final_state = cpu_copy(dict(optimizer=optimizer.state_dict(),epoch=epochs,
        private_rng=generator.get_state(),support_audits=[graph['audit'] for graph in graphs]))
    if state is not None:
        model.load_state_dict(state)
    value = None if query is None else predict(model,xs,graphs,query,p).cpu().numpy()
    return model,value,trace,dict(epoch=selected,seconds=time.monotonic()-start,
        parameters=sum(v.numel() for v in model.parameters()),trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
        fixed_step_training=fixed_epochs is not None,trained_neural_model=True,training_steps=epochs,
        gradient_evaluations=epochs,optimizer_steps=epochs,rule_triplets_per_update=c['triplet_batch'],
        final_optimizer_state_epoch=epochs,identity_preserves_anchor=False,all_device_seeding_used=False,
        mean_raw_residual=True,query_outcomes_used_for_offset_fitting=False,nested_feature_crossfit=True)
'''
text = text[:start]+fit+text[end:]
def functions(source):
    return {node.name:ast.get_source_segment(source,node) for node in ast.parse(source).body if isinstance(node,ast.FunctionDef)}
old,new = functions(original),functions(text)
unchanged = ['library','offset_arrays','setup','predict','cpu_copy','score_arrays','run','select']
if any(old[name]!=new[name] for name in unchanged) or 'torch.manual_seed(' in text:
    raise ValueError('Unexpected executor change')
TARGET.write_text(text)
record = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),parent=str(PARENT),
    parent_sha256=hashlib.sha256(PARENT.read_bytes()).hexdigest(),source=str(TARGET),
    source_sha256=hashlib.sha256(TARGET.read_bytes()).hexdigest(),unchanged_functions=unchanged,
    generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),sealed_sources_modified=False)
with (HERE/'CONTEXTUAL_RULE_EXECUTOR_DERIVATION_V1.json').open('x') as stream:
    json.dump(record,stream,indent=2)
print(json.dumps(record,indent=2))
