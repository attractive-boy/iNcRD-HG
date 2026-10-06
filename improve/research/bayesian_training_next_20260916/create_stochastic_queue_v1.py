from pathlib import Path
h=Path(__file__).resolve().parent
target=h/'run_stochastic_queue_v1.py'
if target.exists():
 raise ValueError('Preserve queue source')
s=(h/'run_vbll_queue_v1.py').read_text().replace('anchor_variational_lastlayer_experiment_v1','anchor_stochastic_augmentation_v1').replace("FAMILY = 'variational_lastlayer'","FAMILY = 'stochastic_augmentation'").replace('anchor_priority_variational_lastlayer_v1_20260916','anchor_priority_stochastic_augmentation_v1_20260916').replace('VBLL_','STOCHASTIC_AUGMENTATION_')
s=s.replace('import numpy as np\n','import numpy as np\nimport torch\n')
start=s.index('def classify(');end=s.index('\n\ndef main',start)
classify='''def classify(out,stage):
    p = experiment.common.protocol(out)
    ph = base.digest(out/'protocol.json')
    expected = {(task,kind,lr,p['screen_seed']) for task in p['tasks'] for kind in p['kinds'] for lr in p['lrs']}
    if stage == 'confirm':
        selection = base.read_json(out/'SELECTION.json')
        expected |= {(task,kind,selection['chosen'][kind]['lr'],seed) for task in p['tasks']
                     for kind in p['kinds'] for seed in p['confirm_seeds']}
    found,neural,identity = set(),[],[]
    for path in sorted((out/'runs').glob('*/*/*/*/result.json')):
        r = experiment.common.verify(path.parent,ph)
        found.add((r['task'],r['kind'],r['lr'],r['seed']))
        if r.get('adopted_without_new_training',False):
            raise ValueError('No model adoption is registered')
        item = dict(path=str(path),sha256=base.digest(path),task=r['task'],kind=r['kind'],lr=r['lr'],seed=r['seed'])
        trace = base.read_json(path.parent/'trace.json')
        state = torch.load(path.parent/'training_final.pt',map_location='cpu',weights_only=True)
        if r['trained_neural_model']:
            if (r['task'] != 'MiDrug' or r['training_steps'] != 400 or r['gradient_evaluations'] != 800
                    or r['encoder_optimizer_steps'] != 400 or r['score_optimizer_steps'] != 400):
                raise ValueError('Wrong two-optimizer training budget')
            if len(trace) != 400 or [row['epoch'] for row in trace] != list(range(1,401)):
                raise ValueError('Incomplete per-step trace')
            if [row['epoch'] for row in trace if 'anchor_composed' in row] != list(range(20,401,20)):
                raise ValueError('Changed checkpoint-selection budget')
            for row in trace:
                step = row['epoch']
                if (row['omitted_fold'] != (step-1)%5 or row['gradient_evaluations'] != 2*step
                        or row['encoder_optimizer_steps'] != step or row['score_optimizer_steps'] != step
                        or not row['same_rna_triplets'] or row['triplet_count'] != p['core_parameters']['triplet_batch']):
                    raise ValueError('Inconsistent stochastic training trace')
                if any(not np.isfinite(row[key]) for key in ('loss','score_loss','ranking','contrastive','bce')):
                    raise ValueError('Nonfinite training objective')
            for group in ('encoder_optimizer','score_optimizer'):
                if not state[group]['state']:
                    raise ValueError('Empty optimizer state')
                for item_state in state[group]['state'].values():
                    if int(item_state['step']) != 400:
                        raise ValueError('Saved optimizer budget changed')
                    if any(not torch.isfinite(value).all() for value in item_state.values() if isinstance(value,torch.Tensor)):
                        raise ValueError('Saved optimizer is nonfinite')
            neural.append(item)
        else:
            if r['task'] != 'LncDrug' or state is not None or trace or any(r[k] for k in
                    ('epoch','parameters','trainable_parameters','training_steps','gradient_evaluations','encoder_optimizer_steps','score_optimizer_steps')):
                raise ValueError('Invalid identity record')
            scores = experiment.common.load(path.parent/'scores.npz')
            np.testing.assert_array_equal(scores['standalone'],np.zeros_like(scores['standalone']))
            np.testing.assert_array_equal(scores['anchor_composed'],scores['anchor'])
            identity.append(item)
    count = len(p['kinds'])*(len(p['lrs'])+(len(p['confirm_seeds']) if stage == 'confirm' else 0))
    if found != expected or len(neural) != count or len(identity) != count:
        raise ValueError('Missing or unexpected records')
    result = base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),family=FAMILY,stage=stage,
        protocol_sha256=ph,cumulative_new_neural_fits=len(neural),cumulative_identity_scoring_records=len(identity),
        neural=neural,identity=identity,adopted_neural_fits=0,identity_records_count_as_training=False,
        encoder_updates=400,score_updates=400,total_backward_evaluations=800,confirmed_final_innovations=0))
    path = HERE/('STOCHASTIC_AUGMENTATION_RECORD_CLASSIFICATION_'+stage.upper()+'_V1.json')
    if path.exists():
        old = base.check_seal(base.read_json(path))
        if any(old[k] != result[k] for k in ('neural','identity')):
            raise ValueError('Existing classification disagrees')
    else:
        base.write_json(path,result)
    return path
'''
s=s[:start]+classify+s[end:]
target.write_text(s)
print(target)
