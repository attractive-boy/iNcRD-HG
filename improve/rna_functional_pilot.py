"""Verified inference-only pilots of local RNA functional representations."""
import argparse
import ast
import csv
import hashlib
import importlib.util
import logging
from pathlib import Path
import sys
import time
import types

import numpy as np
import torch

from improve import tabular_foundation as base
from improve.rna_functional_sources import ROOT,REPOS
from improve.rna_site_windows import RNA,DEFAULT_OUT as SITE_INPUTS


DEFAULT_OUT=Path('improve/results/rna_functional_pilot_20260913')


def verify_model_files(directory,name):
    manifest=base.read_json(directory/name)
    for item in manifest['files']:
        if base.digest(directory/item['path'])!=item['sha256']:
            raise ValueError('Official model file changed: '+item['path'])
    return manifest


def module_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def original_definition(name,path,definition,namespace):
    """Load one unchanged upstream AST definition without unrelated training imports."""
    tree=ast.parse(path.read_text(),filename=str(path))
    nodes=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name==definition]
    if len(nodes)!=1:
        raise ValueError('Required original definition is missing or ambiguous')
    module=types.ModuleType(name);module.__file__=str(path);module.__dict__.update(namespace)
    sys.modules[name]=module
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),module.__dict__)
    return module


def parnet_loader():
    directory=ROOT/'mhorlacher__parnet'
    compat=directory/('compat_'+REPOS['parnet_compat'][1][:12])
    verify_model_files(compat,'SOURCE_FILES.json');verify_model_files(directory,'WEIGHT_FILES.json')
    checkpoint=directory/'models/NewRBPNet_7M_Penalty-0.0_20250107.pt'
    tree={r['path']:r for r in base.read_json(compat/'discovery.json')['tree']['tree']}
    from improve.rna_functional_sources import blob_hash
    if blob_hash(checkpoint.read_bytes())!=tree['models/'+checkpoint.name]['sha']:
        raise ValueError('Checkpoint is not the one shipped with compatibility source')
    sys.path.insert(0,str(ROOT/'pilot_dependencies'))
    import gin
    if 'parnet' in sys.modules:
        raise ValueError('Run each model pilot in a fresh Python process')
    package=types.ModuleType('parnet');package.__path__=[str(compat/'parnet')];sys.modules['parnet']=package
    layers=module_file('parnet.layers',compat/'parnet/layers.py')
    utils=original_definition('parnet.utils',compat/'parnet/utils.py','sequence_to_onehot',
                              dict(torch=torch,F=torch.nn.functional))
    models=original_definition('parnet.models',compat/'parnet/models.py','RBPNet',
        dict(torch=torch,nn=torch.nn,gin=gin,logging=logging,StemConv1D=layers.StemConv1D,
             ResConvBlock1D=layers.ResConvBlock1D,sequence_to_onehot=utils.sequence_to_onehot))
    allowed=[set,models.RBPNet,*[getattr(layers,k) for k in
        ('LinearProjection','NewAdditiveMix','NewMixCoeffMLP','NewMixCoeffPenalty','ResConvBlock1D','StemConv1D')],
        torch.nn.GELU,torch.nn.BatchNorm1d,torch.nn.Sequential,torch.nn.Conv1d,torch.nn.Dropout1d,torch.nn.Linear]
    def load():
        with torch.serialization.safe_globals(allowed):
            return torch.load(checkpoint,map_location='cpu',weights_only=True).eval()
    return load,checkpoint,compat


def state_digest(model):
    h=hashlib.sha256()
    for key,value in model.state_dict().items():
        array=value.detach().cpu().contiguous().numpy()
        h.update(key.encode());h.update(str(array.shape).encode());h.update(str(array.dtype).encode());h.update(array.tobytes())
    return h.hexdigest()


def setup(out,protocol):
    out.mkdir(parents=True,exist_ok=True)
    if (out/'VALIDATION.json').exists():
        raise ValueError('Pilot has already completed')
    protocol=base.seal(protocol)
    if (out/'protocol.json').exists() and base.read_json(out/'protocol.json')!=protocol:
        raise ValueError('Pilot protocol changed; use a new output directory')
    base.write_json(out/'protocol.json',protocol)
    return protocol


def parnet(args):
    out=Path(args.out_dir)/'parnet';manifest=base.read_json(RNA/'inputs.json')
    if base.digest(RNA/'LncDrug_sequences.json')!=manifest['files']['LncDrug_sequences.json']:
        raise ValueError('Frozen lncRNA sequences changed')
    records=base.read_json(RNA/'LncDrug_sequences.json')
    eligible=[i for i,r in enumerate(records) if r['status']=='annotated_full_transcript'
        and 200<=len(r['sequence'])<=8192 and set(r['sequence'].upper())<=set('ACGTU')]
    ids=np.sort(np.random.default_rng(20260913).choice(eligible,size=min(32,len(eligible)),replace=False))
    loader,checkpoint,compat=parnet_loader()
    p=setup(out,dict(model='PARNET public NewRBPNet 7M, penalty 0.0',
        code_commit=REPOS['parnet_compat'][1],discovery_commit=REPOS['parnet'][1],
        source_manifest_sha256=base.digest(compat/'SOURCE_FILES.json'),checkpoint_sha256=base.digest(checkpoint),
        code_sha256=base.digest(Path(__file__)),sequence_sha256=base.digest(RNA/'LncDrug_sequences.json'),
        indices=ids.tolist(),selection='Seed 20260913, 32 complete ACGTU lncRNA transcripts of length 200..8192',
        extraction='Unchanged upstream RBPNet and sequence_to_onehot AST definitions, plus the original full layer module; unrelated TensorFlow/ByteNet training imports are not executed. Restricted torch weights-only loading with explicit class allowlist.',
        representation='Full transcript inference; collect hidden input to the prediction head, globally average and average within five contiguous sequence bins. No truncation, synthetic sequence padding, or downstream labels.',
        no_downstream_evaluation=True,no_test_arrays=True))
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=loader().to(args.device);before=state_digest(model);start=time.monotonic()
    captured={}
    def capture(module,inputs):
        captured['hidden']=inputs[0]
    handle=model.head.register_forward_pre_hook(capture)
    def extract(index):
        seq=records[index]['sequence'].upper().replace('U','T')
        from parnet.utils import sequence_to_onehot
        x=sequence_to_onehot(seq).unsqueeze(0).float().to(args.device)
        with torch.inference_mode():
            result=model({'sequence':x})
            hidden=captured.pop('hidden')
            mean=hidden.mean(-1)[0].cpu().numpy()
            local=torch.stack([part.mean(-1)[0] for part in torch.tensor_split(hidden,5,dim=-1)]).cpu().numpy()
            for track in ('target','control','total'):
                if not torch.isfinite(result[track]).all():
                    raise ValueError('Nonfinite RBP profile')
                torch.testing.assert_close(result[track].exp().sum(-1),torch.ones_like(result[track][:,:,0]),atol=2e-5,rtol=1e-5)
        return mean,local,int(result['target'].shape[1])
    values=[extract(int(i)) for i in ids]
    means=np.stack([v[0] for v in values]);local=np.stack([v[1] for v in values])
    reverse=[extract(int(i)) for i in ids[::-1]][::-1]
    np.testing.assert_allclose(means,np.stack([v[0] for v in reverse]),atol=1e-6,rtol=1e-6)
    np.testing.assert_allclose(local,np.stack([v[1] for v in reverse]),atol=1e-6,rtol=1e-6)
    if state_digest(model)!=before:
        raise ValueError('Inference modified pretrained state')
    handle.remove();del model
    model=loader().to(args.device)
    handle=model.head.register_forward_pre_hook(capture)
    replay=[extract(int(i)) for i in ids]
    np.testing.assert_allclose(means,np.stack([v[0] for v in replay]),atol=1e-6,rtol=1e-6)
    np.testing.assert_allclose(local,np.stack([v[1] for v in replay]),atol=1e-6,rtol=1e-6)
    handle.remove()
    np.savez_compressed(out/'representations.npz',rna_indices=ids,global_mean=means,local_bins=local)
    metadata=[dict(index=int(i),**{k:records[i][k] for k in ('name','status','transcript','gene_id')},
                   length=len(records[i]['sequence'])) for i in ids]
    base.write_json(out/'RNA_RECORDS.json',metadata)
    result=dict(protocol_sha256=p['sha256'],passed=True,pilot_rnas=len(ids),eligible_complete_rnas=len(eligible),
        input_length_range=[min(r['length'] for r in metadata),max(r['length'] for r in metadata)],
        embedding_dimensions=means.shape[1],local_shape=list(local.shape),rbp_tracks=values[0][2],
        parameters=sum(v.numel() for v in model.parameters()),state_sha256=before,
        replay_max_abs=float(np.max(np.abs(means-np.stack([v[0] for v in replay])))),
        query_order_verified=True,seconds=time.monotonic()-start,
        artifacts={name:base.digest(out/name) for name in ('representations.npz','RNA_RECORDS.json')},
        downstream_performance_claimed=False,limitations=['This verifies usability of a public historical checkpoint, not equivalence to every PARNET paper experiment.',
            'No evidence yet that these features improve ncRNA-drug prediction or outperform a matched generic RNA representation.',
            'Training is based on eCLIP in K562/HepG2; transcript embedding does not establish broad context generalization.'])
    base.write_json(out/'VALIDATION.json',result)
    print('PARNET_PILOT',result,flush=True)


def mirbind(args):
    directory=ROOT/'BioGeMT__miRBind_2.0';out=Path(args.out_dir)/'mirbind'
    verify_model_files(directory,'SOURCE_FILES.json');verify_model_files(directory,'WEIGHT_FILES.json')
    manifest=base.read_json(SITE_INPUTS/'INPUTS.json')
    for name,digest in manifest['files'].items():
        if base.digest(SITE_INPUTS/name)!=digest:
            raise ValueError('Verified site input changed')
    with (SITE_INPUTS/'sites.tsv').open() as f:
        rows=list(csv.DictReader(f,delimiter='\t'))
    shared=directory/'code/pairwise_binding_site_model/shared'
    models=module_file('mirbind_models',shared/'models.py')
    encoding=module_file('mirbind_encoding',shared/'encoding.py')
    constants=module_file('mirbind_constants',shared/'constants.py')
    checkpoint=directory/'models/pairwise_onehot_model_20260105_200141.pt'
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    params=saved['model_params']
    p=setup(out,dict(model='miRBind2 binding-site model',code_commit=REPOS['mirbind'][1],
        source_manifest_sha256=base.digest(directory/'SOURCE_FILES.json'),checkpoint_sha256=base.digest(checkpoint),
        code_sha256=base.digest(Path(__file__)),site_inputs_manifest_sha256=base.digest(SITE_INPUTS/'INPUTS.json'),
        model_params=params,representation='Binding score and 30-dimensional post-activation hidden vector. Original U-to-T encoding; mature miRNA N-padded to the trained 28 nt length. All mRNA targets have 50 verified real nucleotides.',
        no_downstream_evaluation=True,no_test_arrays=True))
    def load():
        model=models.PairwiseOneHotCNN(**params)
        model.load_state_dict(saved['model_state_dict'],strict=True)
        return model.eval().to(args.device)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    pair_to_index=constants.get_pair_to_index();encoded=[]
    for row in rows:
        target=row['target_sequence'];mirna=row['mirna_sequence']
        if len(target)!=params['target_length'] or len(mirna)>params['mirna_length']:
            raise ValueError('Pilot would truncate or pad a target site')
        mirna=encoding.pad_or_trim(mirna,params['mirna_length'])
        value=encoding.encode_complementarity(target,mirna,params['target_length'],params['mirna_length'],pair_to_index,len(pair_to_index))
        encoded.append(torch.nn.functional.one_hot(torch.tensor(value,dtype=torch.long),num_classes=len(pair_to_index)+1).float())
    x=torch.stack(encoded).to(args.device)
    model=load();before=state_digest(model);start=time.monotonic();captured={}
    def capture(module,inputs,output):
        captured['hidden']=output
    handle=model.dropout_fc.register_forward_hook(capture)
    def predict(values):
        with torch.inference_mode():
            score=model(values).ravel().cpu().numpy()
            hidden=captured.pop('hidden').cpu().numpy()
        return score,hidden
    scores,hidden=predict(x);reversed_scores,reversed_hidden=predict(x.flip(0))
    np.testing.assert_allclose(scores,reversed_scores[::-1],atol=1e-6,rtol=1e-6)
    np.testing.assert_allclose(hidden,reversed_hidden[::-1],atol=1e-6,rtol=1e-6)
    if state_digest(model)!=before:
        raise ValueError('Inference modified pretrained state')
    handle.remove();model=load();handle=model.dropout_fc.register_forward_hook(capture)
    replay,replay_hidden=predict(x)
    np.testing.assert_allclose(scores,replay,atol=1e-6,rtol=1e-6)
    np.testing.assert_allclose(hidden,replay_hidden,atol=1e-6,rtol=1e-6)
    handle.remove()
    np.savez_compressed(out/'representations.npz',binding_scores=scores,site_features=hidden,
        rna_indices=np.array([int(r['rna_index']) for r in rows]))
    with (out/'site_predictions.tsv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=[*rows[0],'binding_score'],delimiter='\t');writer.writeheader()
        writer.writerows([dict(r,binding_score=float(s)) for r,s in zip(rows,scores)])
    result=dict(protocol_sha256=p['sha256'],passed=True,sites=len(rows),mirnas=len({r['rna_name'] for r in rows}),
        genes=len({r['gene_id'] for r in rows}),embedding_dimensions=hidden.shape[1],
        parameters=sum(v.numel() for v in model.parameters()),binding_score_range=[float(scores.min()),float(scores.max())],
        state_sha256=before,replay_max_abs=float(np.max(np.abs(scores-replay))),query_order_verified=True,
        target_padding_used=False,seconds=time.monotonic()-start,
        artifacts={name:base.digest(out/name) for name in ('representations.npz','site_predictions.tsv')},
        downstream_performance_claimed=False,limitations=['Binding-model outputs are predictions, not measured binding or repression.',
            'Current RefSeq isoforms may differ from the original experiment; only exact observed sites and their current real flanks were used.',
            'This is a positive-evidence feasibility pilot, not an independent accuracy benchmark or ncRNA-drug improvement.'])
    base.write_json(out/'VALIDATION.json',result)
    print('MIRBIND_PILOT',result,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model',choices=('parnet','mirbind'))
    parser.add_argument('--out-dir',default=str(DEFAULT_OUT))
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    dict(parnet=parnet,mirbind=mirbind)[args.model](args)


if __name__=='__main__':
    main()
