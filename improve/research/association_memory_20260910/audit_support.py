"""Training-internal feasibility audit of association-instance evidence.

This is a fixed structural diagnostic, not a trained model or new validation
claim. Only the existing internal split is evaluated; no outer/test arrays are
loaded into the audit.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch

from improve import pair_exchange as parent
from improve.first_round_features import sha256


def incoming(graph, node_type):
    n=graph[node_type].num_nodes
    result=np.zeros((n,n),dtype=np.float64)
    for source,relation,target in graph.edge_types:
        if source==target==node_type:
            edge=graph[(source,relation,target)].edge_index.numpy()
            result[edge[1],edge[0]]=1.
    np.fill_diagonal(result,0.)
    return result


def normalize(x):
    return x/np.maximum(x.sum(axis=1,keepdims=True),1.)


def metrics(y,scores,pairs):
    aps=[]
    for drug in np.unique(pairs[1]):
        keep=pairs[1]==drug
        if len(np.unique(y[keep]))==2:
            aps.append(average_precision_score(y[keep],scores[keep]))
    return dict(auc=float(roc_auc_score(y,scores)),aupr=float(average_precision_score(y,scores)),
                macro_drug_aupr=float(np.mean(aps)),eligible_drugs=len(aps))


def main():
    torch.set_num_threads(4)
    reference=Path('improve/results/dual_pair_tuning_20260910')
    p=json.loads((reference/'protocol.json').read_text())
    out=Path(__file__).parent/'support_audit.json'
    if out.exists():
        raise ValueError('Preserve existing audit')
    result=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
                protocol_sha256=p['sha256'],audit_source_sha256=sha256(Path(__file__)),
                scope='Fixed untrained structural diagnostics on training-internal holdout only; not performance of the proposed neural architecture.',
                outer_evaluations=0,test_evaluations=0,tasks={})
    for task in p['tasks']:
        arrays,graph=parent.load_stage(reference,p,task,'inner')
        rna=p['training'][task]['rna']
        pair,y=arrays['pairs_train'],arrays['y_train']
        queries,truth=arrays['pairs_val'],arrays['y_val']
        r,d=queries
        nr,nd=graph[rna].num_nodes,graph['drug'].num_nodes
        a=np.zeros((nr,nd),dtype=np.float64)
        a[pair[0,y==1],pair[1,y==1]]=1.
        assert not np.any(a[r,d])
        row,col=a.sum(1),a.sum(0)
        null=row[:,None]*col[None,:]/a.sum()
        centered=a-null
        assert np.max(np.abs(centered.sum(0)))<1e-10
        assert np.max(np.abs(centered.sum(1)))<1e-10
        sr,sd=incoming(graph,rna),incoming(graph,'drug')
        r_only=sr@a
        d_only=a@sd.T
        both=r_only@sd.T
        kr,kd=normalize(sr+np.eye(nr)),normalize(sd+np.eye(nd))
        matrices=dict(endpoint_frequency=null,one_axis_paths=r_only+d_only,
                      joint_support=kr@a@kd.T,centered_joint_support=kr@centered@kd.T)
        counts=both[r,d]
        report=dict(n_rna=nr,n_drug=nd,fit_rows=len(y),fit_positive_pairs=int(a.sum()),
                    inner_queries=len(truth),inner_positives=int(truth.sum()),
                    same_rna_type_relation_edges=int(sr.sum()),drug_relation_edges=int(sd.sum()),
                    queries_with_one_axis_support=int(np.count_nonzero((r_only+d_only)[r,d])),
                    queries_with_two_changed_endpoints_support=int(np.count_nonzero(counts)),
                    two_changed_endpoints_support_fraction=float(np.mean(counts>0)),
                    support_count_quantiles={str(q):float(np.quantile(counts,q)) for q in (0,.25,.5,.75,1)},
                    centered_row_sum_max_abs=float(np.max(np.abs(centered.sum(1)))),
                    centered_col_sum_max_abs=float(np.max(np.abs(centered.sum(0)))),
                    fixed_diagnostics={name:metrics(truth,matrix[r,d],queries) for name,matrix in matrices.items()},
                    input_sha256={name:sha256(reference/'inputs'/task/name) for name in ('inner.npz','train_graph.pth')})
        result['tasks'][task]=report
        print(task,json.dumps(report,ensure_ascii=False),flush=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    main()
