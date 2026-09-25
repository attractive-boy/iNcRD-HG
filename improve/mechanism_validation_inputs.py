"""Fresh development partitions, prepared without evaluating predictions.

The only labels read from the historical outer archive are pairs_train/y_train.
Other-task relations retain the established single-task transductive protocol.
These overlapping partitions are sensitivity checks, not new biological data.
Preparation runs in the historical PyG environment; reference inference later
uses the isolated TabICL environment and requires a candidate freeze record.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import shutil

import numpy as np

from improve import tabular_foundation as base

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results/mechanism_validation_20260915'
SPLITS=(61453,7919,104729)
TASKS=('MiDrug','LncDrug')


def physical_arrays(graph, task, a):
    nts=list(graph.node_types)
    offsets={}; total=0
    for name in nts:
        offsets[name]=total
        total+=graph[name].num_nodes
    rna='miRNA' if task=='MiDrug' else 'lncRNA'
    edges,relations,codes,keys=[],[],[],[]
    for key,ix in graph.edge_index_dict.items():
        src,rel,dst=key
        if rel.startswith('rev_'):
            continue
        e=ix.cpu().numpy()
        if src==dst:
            e=e[:,e[0]<e[1]]
        if not e.shape[1]:
            raise ValueError('Empty physical relation would change relation numbering')
        code=np.full(e.shape[1],-1,dtype=np.int64)
        if key==(rna,task,'drug'):
            code=e[0]*graph['drug'].num_nodes+e[1]
        edges.append(e+np.array([offsets[src],offsets[dst]])[:,None])
        relations.append(np.full(len(code),len(keys),dtype=np.int64))
        codes.append(code); keys.append(list(key))
    result=dict(edges=np.concatenate(edges,1),relation=np.concatenate(relations),
                target_codes=np.concatenate(codes),
                node_counts=np.array([graph[nt].num_nodes for nt in nts],dtype=np.int64),
                pair_offset=np.array([offsets[rna],offsets['drug']],dtype=np.int64),
                nd=np.array(graph['drug'].num_nodes))
    for i,nt in enumerate(nts):
        result['x_'+str(i)]=graph[nt].x.cpu().numpy().copy()
    expected=a['pairs_train'][:,a['y_train']==1]
    expected=set(expected[0]*int(result['nd'])+expected[1])
    assert set(result['target_codes'][result['target_codes']>=0])==expected
    assert not np.isin(a['pairs_val'][0]*int(result['nd'])+a['pairs_val'][1],result['target_codes']).any()
    return result,dict(node_types=nts,physical_relations=keys,nodes=total,
                       physical_edges=result['edges'].shape[1],target_overlap=0)


def prepare(out):
    import torch
    from sklearn.model_selection import StratifiedKFold,train_test_split
    from improve.graph_tabm_modules import PairFeatureEncoder,crossfit_features,fit_transform
    from improve.tabicl_graph_context_features import build_views
    from improve.tabicl_context_program_modules import ContextPrograms
    out=Path(out).resolve()
    if out.exists():
        raise ValueError('Use a new validation directory')
    graph_root=ROOT/'results/backbone_replacement_20260910'
    context_root=ROOT/'results/tabicl_pathway_context_20260913'
    cp=base.read_json(context_root/'protocol.json')
    ip=base.read_json(ROOT/'results/tabicl_20260912/protocol.json')
    evidence=Path(cp['inputs_dir'])
    encoder=ContextPrograms.load(evidence)
    names=('mechanism_validation_inputs.py','validate_transfer_splits.py',
           'graph_tabm_modules.py','tabicl_graph_context_features.py',
           'tabicl_graph_context_modules.py','tabicl_context_program_modules.py',
           'tabicl_pathway_modules.py','network_feature_ablation.py','tabular_foundation.py',
           'context_anchor_verify.py','anchor_shrinkage.py')
    inputs=[context_root/'protocol.json',ROOT/'results/tabicl_20260912/protocol.json',
            evidence/'INPUTS.json',Path(ip['checkpoint'])]
    manifest=base.read_json(evidence/'INPUTS.json')
    if manifest['labels_read']:
        raise ValueError('Evidence inputs must not use task outcomes')
    inputs.extend(evidence/name for name in manifest['files'])
    for task in TASKS:
        inputs.extend(graph_root/'inputs'/task/name for name in ('outer.npz','train_graph.pth'))
    p=base.seal(dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        smoke=False,tasks=list(TASKS),split_seeds=list(SPLITS),model_seed=20260915,
        reference_seed=20260914,holdout_fraction=.2,members=32,
        feature_fold_seed=20260910,neural_fold_seed=20260915,folds=5,
        arms=['tabicl','graph_context','tiered','shared_anchor'],
        icl_protocol=ip,graph_recipes=cp['graph_recipes'],input_configs=cp['input_configs'],
        evidence_dir=str(evidence),
        anchor=dict(members=['hgb','et','mlp','lr'],seed=20260914,bank='context76',weight=.3),
        source_sha256={n:base.digest(ROOT/n) for n in names},
        input_sha256={str(path):base.digest(path) for path in inputs},
        preparation_only=True,no_prediction_metrics_during_preparation=True,
        candidate_freeze_required_before_prediction_evaluation=True,
        graph_policy='Rebuild target adjacency from each fitting partition; delete heldout query target facts before counts, inverse arcs and propagation. Other-task relations, fixed PCA and circRNA side information keep the historical protocol.',
        limitations=['New partitions of previously used development labels; no new biological data.',
                     'The three partitions overlap and are not independent experiments.',
                     'The fixed 0.001 LncDrug tolerance is an engineering gate, not statistical noninferiority.',
                     'No original reserved test or historical outer query labels are loaded.'],
        original_test_access=False,historical_outer_labels_access=False))
    out.mkdir(parents=True); (out/'source_snapshot').mkdir()
    base.write_json(out/'protocol.json',p)
    for n in names:
        shutil.copy2(ROOT/n,out/'source_snapshot'/n)
    for task in TASKS:
        directory=graph_root/'inputs'/task
        graph=torch.load(directory/'train_graph.pth',map_location='cpu')
        with np.load(directory/'outer.npz',allow_pickle=False) as z:
            pairs,labels=z['pairs_train'].copy(),z['y_train'].copy()
            for nt in graph.node_types:
                key=('pca_x_' if nt in ('lncRNA','miRNA') else 'x_')+nt
                graph[nt].x=torch.from_numpy(z[key].copy())
        assert len(set(map(tuple,pairs.T)))==len(labels)
        rna='miRNA' if task=='MiDrug' else 'lncRNA'
        target,reverse=(rna,task,'drug'),('drug','rev_'+task,rna)
        for split in p['split_seeds']:
            fit,query=train_test_split(np.arange(len(labels)),test_size=p['holdout_fraction'],
                                      random_state=split,stratify=labels)
            fit,query=np.sort(fit),np.sort(query)
            a=dict(pairs_train=pairs[:,fit],y_train=labels[fit],
                   pairs_val=pairs[:,query],y_val=labels[query])
            g=graph.clone()
            support=a['pairs_train'][:,a['y_train']==1]
            g[target].edge_index=torch.from_numpy(support.copy())
            g[reverse].edge_index=g[target].edge_index.flip(0).clone()
            for key in g.edge_types:
                for attr in ('edge_label','edge_label_index'):
                    if attr in g[key]:
                        del g[key][attr]
            raw,raw_val,columns,folds=crossfit_features(g,task,a['pairs_train'],a['y_train'],
                a['pairs_val'],p['folds'],p['feature_fold_seed'])
            views,view_names,checks,error=build_views(PairFeatureEncoder(g,task),a['pairs_train'],
                a['y_train'],folds,a['pairs_val'],raw,raw_val)
            assert columns==view_names and len(columns)==36
            x,xv,transform=fit_transform(raw,raw_val)
            neural_folds=np.zeros(len(fit),dtype=np.int64)
            for f,(_,ix) in enumerate(StratifiedKFold(p['folds'],shuffle=True,
                    random_state=p['neural_fold_seed']).split(a['pairs_train'].T,a['y_train'])):
                neural_folds[ix]=f
            a.update(raw_train=raw,raw_val=raw_val,raw_query_views=views,x_train=x,x_val=xv,
                     x_query_views=transform.transform(views.reshape(-1,36)).astype(np.float32).reshape(views.shape),
                     fold_ids=folds,folds=neural_folds)
            physical,description=physical_arrays(g,task,a)
            a.update(physical)
            prefix='raw' if p['input_configs'][task]['input_view']=='raw' else 'x'
            a['context76_train']=np.concatenate([a[prefix+'_train'],
                encoder.transform(task,a['pairs_train'],'context')],axis=1).astype(np.float32)
            a['context76_val']=np.concatenate([a[prefix+'_val'],
                encoder.transform(task,a['pairs_val'],'context')],axis=1).astype(np.float32)
            assert a['context76_train'].shape[1]==76
            dest=out/'inputs'/task/str(split);dest.mkdir(parents=True)
            np.savez_compressed(dest/'arrays.npz',**a)
            with (dest/'transform.pkl').open('wb') as stream:
                pickle.dump(transform,stream,protocol=4)
            base.write_json(dest/'graph.json',description)
            base.write_json(dest/'manifest.json',dict(protocol_sha256=p['sha256'],names=columns,
                fitting_pool_rows=fit.tolist(),holdout_pool_rows=query.tolist(),target_overlap=0,
                fold_audits=checks,full_view_reconstruction_error=error,
                files={n:base.digest(dest/n) for n in ('arrays.npz','transform.pkl','graph.json')}))
            print('PREPARED',task,split,len(fit),len(query),'metrics not evaluated',flush=True)
    base.write_json(out/'INPUTS_COMPLETE.json',dict(protocol_sha256=p['sha256'],
        manifests={str(path.relative_to(out)):base.digest(path) for path in (out/'inputs').glob('*/*/manifest.json')},
        metrics_evaluated=False,original_test_access=False,historical_outer_query_labels_access=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',default=str(OUT))
    a=parser.parse_args()
    prepare(a.out)
