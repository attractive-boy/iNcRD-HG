"""Derive an executor that changes training only, retaining scoring infrastructure."""
from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parents[2]
source=root/'anchor_priority_graph_v1.py'
target=root/'anchor_stochastic_augmentation_v1.py'
if target.exists():
 raise ValueError('Preserve executor')
s=source.read_text()
start=s.index('OWN = ');end=s.index('\n\n\ndef library',start)
names=('anchor_stochastic_augmentation_v1.py','stochastic_augmentation_graph_v1.py','test_stochastic_augmentation_v1.py','sharpness_graph_v1.py','overlap_block_models.py','overlap_block_experiment.py','non_tabular_experiment.py','anchor_priority_graph_v1.py','assess_declared_graph.py','graph_prediction_precision.py','residual_graph_experiment.py','nested_anchor_inputs_v2.py')
s=s[:start]+'OWN = '+repr(names)+s[end:]
start=s.index('def fit(');end=s.index('\n\ndef score_arrays',start)
fit='''def cpu_copy(value):
    if isinstance(value,torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value,dict):
        return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return type(value)(cpu_copy(v) for v in value)
    return value


def fit(a,p,kind,lr,seed,device,fixed_epochs=None):
    if device != 'cuda:1' and not (device == 'cpu' and p.get('smoke',False)):
        raise ValueError('Formal fitting is restricted to cuda:1')
    if fixed_epochs is not None and set(a)-TRAIN_KEYS:
        raise ValueError('Fixed-step fitting accepts training arrays only')
    if kind not in p['kinds']:
        raise ValueError('Unregistered augmentation arm')
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
        value = None if fixed_epochs is not None else np.zeros(len(a['y_val']),dtype=np.float64)
        return model,value,[],dict(epoch=0,seconds=0.,parameters=0,trainable_parameters=0,
            fixed_step_training=fixed_epochs is not None,trained_neural_model=False,training_steps=0,
            gradient_evaluations=0,encoder_optimizer_steps=0,score_optimizer_steps=0,
            identity_preserves_anchor=True,no_new_lnc_training=True,query_outcomes_used_for_offset_fitting=False)
    c = p['core_parameters']
    epochs = p['epochs'] if fixed_epochs is None else int(fixed_epochs)
    if epochs < 1:
        raise ValueError('At least one training update is required')
    pairs = torch.as_tensor(a['pairs_train']+a['pair_offset'][:,None],device=device)
    labels = torch.as_tensor(a['y_train'],device=device,dtype=torch.float64)
    offset = torch.as_tensor(a['anchor_oof'],device=device,dtype=torch.float64)
    query = None if fixed_epochs is not None else torch.as_tensor(a['pairs_val']+a['pair_offset'][:,None],device=device)
    pools = library(p).training_triplets(training,device,c['folds'])
    generator = torch.Generator(device=device)
    generator.manual_seed(seed+p['private_rng_offset'])
    encoder_parameters = [v for name,v in model.named_parameters() if not name.startswith('score_network.')]
    optimizer = torch.optim.AdamW(encoder_parameters,lr=lr,weight_decay=p['weight_decay'])
    score_optimizer = torch.optim.AdamW(model.score_network.parameters(),lr=c['score_lr'],weight_decay=c['score_weight_decay'])
    best,selected,state,trace = -np.inf,epochs,None,[]
    start = time.monotonic()
    for step in range(1,epochs+1):
        fold = (step-1)%len(graphs)
        ids = folds[fold]
        model.train()
        h = model.encode(xs,graphs[fold])
        score_optimizer.zero_grad(set_to_none=True)
        time_sample = .00001+.99999*torch.rand(len(h),device=device,generator=generator)
        noise = torch.randn(h.shape,device=device,dtype=h.dtype,generator=generator)
        score_loss = library(p).denoising_loss(model.score_network,h,c,time_sample,noise)
        if not torch.isfinite(score_loss):
            raise FloatingPointError('Nonfinite score-matching loss')
        score_loss.backward()
        score_gradient = torch.nn.utils.clip_grad_norm_(model.score_network.parameters(),5.)
        if not torch.isfinite(score_gradient):
            raise FloatingPointError('Nonfinite score-matching gradient')
        score_optimizer.step()
        score_optimizer.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        positive,negative = library(p).sample_triplets(pools[fold],c['triplet_batch'],generator)
        rnas = torch.unique(pairs[0,ids])
        drugs = torch.unique(pairs[1,ids])
        nodes = torch.cat((rnas,drugs))
        view_noise = torch.randn((c['reverse_steps']+1,len(nodes),c['width']),device=device,generator=generator)
        hard_noise = torch.randn((c['reverse_steps']+1,2*c['triplet_batch'],c['width']),device=device,generator=generator)
        for parameter in model.score_network.parameters():
            parameter.requires_grad_(False)
        try:
            view1,view2 = library(p).reverse_sample(model.score_network.score,h[nodes],c,view_noise,kind)
            latent = torch.cat((h[pairs[1,positive]],h[pairs[1,negative]]),0)
            _,synthetic = library(p).reverse_sample(model.score_network.score,latent,c,hard_noise,kind,inject=True)
            hard = synthetic[c['triplet_batch']:]
            cl = .5*(library(p).contrastive(view1[:len(rnas)],view2[:len(rnas)],c['contrastive_temperature'])+
                       library(p).contrastive(view1[len(rnas):],view2[len(rnas):],c['contrastive_temperature']))
            residual = model.decode(h[pairs[0,ids]],h[pairs[1,ids]])
            bce = F.binary_cross_entropy_with_logits(offset[ids]+residual.double(),labels[ids])
            pos_score = offset[positive]+model.decode(h[pairs[0,positive]],h[pairs[1,positive]]).double()
            neg_score = offset[negative]+model.decode(h[pairs[0,negative]],hard).double()
            ranking = F.softplus(neg_score-pos_score).mean()
            loss = bce+c['ranking_weight']*ranking+c['contrastive_weight']*cl
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite augmented training objective')
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_(encoder_parameters,5.)
            if not torch.isfinite(gradient):
                raise FloatingPointError('Nonfinite augmented training gradient')
            optimizer.step()
        finally:
            for parameter in model.score_network.parameters():
                parameter.requires_grad_(True)
        row = dict(epoch=step,omitted_fold=fold,query_count=len(ids),triplet_count=len(positive),
            loss=float(loss.detach()),bce=float(bce.detach()),ranking=float(ranking.detach()),
            contrastive=float(cl.detach()),score_loss=float(score_loss.detach()),
            encoder_gradient_norm=float(gradient),score_gradient_norm=float(score_gradient),
            encoder_optimizer_steps=step,score_optimizer_steps=step,gradient_evaluations=2*step,
            synthetic_displacement=float((hard.detach()-h[pairs[1,negative]].detach()).square().mean().sqrt()),
            same_rna_triplets=bool(torch.equal(pairs[0,positive],pairs[0,negative])))
        if not row['same_rna_triplets']:
            raise ValueError('Triplets mix RNA identities')
        if (step%p['check_interval']==0 or step==epochs) and query is not None:
            value = predict(model,xs,graphs,query,p).cpu().numpy()
            metrics = common.base.metrics(a['y_val'],a['anchor_'+str(seed)]+p['strong_weight']*value,a['pairs_val'])
            row['anchor_composed'] = metrics
            criterion = (metrics['auc']+metrics['ap'])/2
            if criterion > best:
                best,selected = criterion,step
                state = cpu_copy(model.state_dict())
            print('TRAIN_STOCHASTIC_AUGMENTATION',kind,lr,seed,step,round(metrics['auc'],9),round(metrics['ap'],9),flush=True)
        trace.append(row)
    model.training_final_state = cpu_copy(dict(encoder_optimizer=optimizer.state_dict(),score_optimizer=score_optimizer.state_dict(),private_rng=generator.get_state(),epoch=epochs))
    if state is not None:
        model.load_state_dict(state)
    value = None if query is None else predict(model,xs,graphs,query,p).cpu().numpy()
    return model,value,trace,dict(epoch=selected,seconds=time.monotonic()-start,
        parameters=sum(v.numel() for v in model.parameters()),trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
        fixed_step_training=fixed_epochs is not None,trained_neural_model=True,training_steps=epochs,
        gradient_evaluations=2*epochs,encoder_optimizer_steps=epochs,score_optimizer_steps=epochs,
        final_optimizer_state_epoch=epochs,identity_preserves_anchor=False,all_device_seeding_used=False,
        mean_raw_residual=True,query_outcomes_used_for_offset_fitting=False,nested_feature_crossfit=True)
'''
s=s[:start]+fit+s[end:]
s=s.replace("def setup(a, p, kind, device):\n    model", "def setup(a, p, kind, device):\n    if set(a)-TRAIN_KEYS:\n        raise ValueError('Setup accepts training arrays only')\n    model")
s=s.replace("                dest.mkdir(parents=True, exist_ok=True)\n                model", "                if dest.exists() and any(dest.iterdir()):\n                    raise ValueError('Preserve partial fitting artifacts')\n                dest.mkdir(parents=True, exist_ok=True)\n                model")
s=s.replace("                torch.save(model.state_dict(), dest / 'model.pt')", "                torch.save(model.state_dict(), dest / 'model.pt')\n                torch.save(model.training_final_state, dest / 'training_final.pt')")
s=s.replace("loaded, xs, graphs, _ = setup(a, p, kind, device)","loaded, xs, graphs, _ = setup({k:v for k,v in a.items() if k in TRAIN_KEYS}, p, kind, device)")
s=s.replace("('model.pt', 'trace.json', 'scores.npz')","('model.pt', 'trace.json', 'scores.npz', 'training_final.pt')")
s=s.replace("likelihood='MiDrug uses BCEWithLogits(fixed nested OOF strong reference + learned graph residual). Every neural query fold is absent from its reference fit and all nested target-dependent feature supports. Marginal degree readout heads are identically zero and frozen.'", "likelihood='BCE on OOF-reference-plus-residual, plus 0.1 BPR on same-RNA actual training pairs with latent synthetic negative embeddings and 0.05 symmetric InfoNCE on noisy/reversed node views. The synthetic negative retains its original training-pair reference logit; it is a regularizer, not an observed new association.',\n        training_budget='400 encoder AdamW updates and 400 detached-feature denoising score-network updates; two backwards per step. Generator parameters are frozen during differentiable reverse-sampling gradients to encoder features. All arms train the same score network and draw the same private random tensors.',\n        local_adaptation=parent['local_adaptation']")
if 'torch.manual_seed(' in s:
 raise ValueError('Forbidden global seeding remained')
target.write_text(s)
manifest=dict(parent=str(source),parent_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),derived=str(target),derived_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),prediction_and_selection_unchanged=True,training='Score loss plus encoder BCE/ranking/contrastive losses with matched separate updates')
(root/'research/bayesian_training_next_20260916/STOCHASTIC_EXECUTOR_DERIVATION_V1.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(manifest)
