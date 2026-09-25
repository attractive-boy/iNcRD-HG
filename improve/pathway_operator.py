"""A named-pathway interaction operator extending the product-kernel scorer.

f(r,d) = f_kernel(r,d) + sum_c sum_p w[c,p] R[c,r,p] D[d,p].
The induced pair covariance is a sum of matched-pathway products, generally
nonseparable into one RNA kernel times one drug kernel. All pathway identities
are retained. This is a structured linear/RKHS extension, not a new GP theorem
or a pretrained foundation model. Novelty and utility require empirical controls.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch import nn
from torch.nn import functional as F

from improve import non_tabular_experiment as base
from improve.non_tabular_models import KernelLogistic

OUT = base.ROOT/'results'/'pathway_operator_20260914'
EVIDENCE = base.EVIDENCE
SPLITS = base.ROOT/'results'/'transfer_splits_20260914'
VARIANTS = ('kernel','annotation','context','shuffled_context','pooled_context')


def unit_rows(matrix):
    x = sparse.csr_matrix(matrix, dtype=np.float64)
    norm = np.sqrt(np.asarray(x.multiply(x).sum(1)).ravel())
    return (sparse.diags(1/np.maximum(norm,1e-12))@x).tocsr()


def pathway_projection(gene_pathway):
    x = sparse.csr_matrix(gene_pathway,dtype=np.float64).copy()
    x.eliminate_zeros()
    x.data[:] = 1.
    a = np.asarray(x.sum(1)).ravel()
    b = np.asarray(x.sum(0)).ravel()
    return (sparse.diags(1/np.sqrt(np.maximum(a,1.)))@x@sparse.diags(1/np.sqrt(np.maximum(b,1.)))).tocsr()


def normalize_channels(rna, drug):
    """Expected feature norm under all entity pairs; no association outcomes."""
    rna = np.asarray(rna,dtype=np.float64)
    drug = np.asarray(drug,dtype=np.float64)
    if rna.ndim!=3 or rna.shape[2]!=drug.shape[1]:
        raise ValueError('Pathway dictionary mismatch')
    norm2 = (np.mean(rna*rna,axis=1)*np.mean(drug*drug,axis=0)[None,:]).sum(1)
    scale = np.sqrt(np.maximum(norm2,1e-12))
    return (rna/scale[:,None,None]).astype(np.float32),drug.astype(np.float32),scale


class PathwayOperator(nn.Module):
    def __init__(self,a,variant):
        super().__init__()
        self.variant=variant
        self.kernel=KernelLogistic(a['ur'],a['ud'],float(a['y_train'].mean()))
        r=np.array(a['program_rna'],copy=True)
        d=np.array(a['program_drug'],copy=True)
        if variant=='shuffled_context':
            # Same permutation for global, within-lineage and absolute context.
            # Keep the independently annotated RNA-gene channel untouched.
            permutation=np.random.default_rng(20260914).permutation(r.shape[2])
            r[1:]=r[1:][:,:,permutation]
        if variant not in VARIANTS:
            raise ValueError('Unknown operator')
        r,d,scales=normalize_channels(r,d)
        r/=np.sqrt(len(r))
        if variant=='annotation':
            # Retain exactly the annotation component/prior scale of context.
            r[1:]=0.
        elif variant=='kernel':
            r=r[:0]
        self.register_buffer('r',torch.as_tensor(r,dtype=torch.float32))
        self.register_buffer('d',torch.as_tensor(d,dtype=torch.float32))
        self.w=nn.Parameter(torch.zeros(len(r),1 if variant=='pooled_context' else d.shape[1]))

    def full(self):
        result=self.kernel.full()
        for c in range(len(self.r)):
            # In the pooled arm one scalar replaces the vector of named effects.
            result=result+(self.r[c]*self.w[c])@self.d.T
        return result

    def forward(self,pairs):
        return self.full()[pairs[0],pairs[1]]

    def penalty(self):
        return self.kernel.penalty(1.,1.)+.5*self.w.square().sum()


def prepare(out):
    out=Path(out)
    if out.exists():raise ValueError('Use a new output directory')
    base.load_protocol(base.OUT)
    out.mkdir(parents=True);(out/'source_snapshot').mkdir()
    names=('pathway_operator.py','non_tabular_models.py','non_tabular_experiment.py')
    p=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),tasks=list(base.TASKS),
           cases=['inner','42','2026','31415'],variants=list(VARIANTS),seed=20260914,
           epochs={'MiDrug':480,'LncDrug':210},lr=.03,
           epoch_policy='Inherited frozen kernel confirmation budgets, identical across all new variants; no new checkpoint selection.',
           primary='context operator, fixed 0.5 GNN logit plus 0.5 operator logit.',
           mechanism='Preserve each Reactome pathway coordinate in annotation and matched-cell expression interactions; compare annotation only, matched vs shuffled expression, pooled context, and the separable kernel baseline.',
           information='Same frozen RNA-gene/drug-gene/Reactome and DepMap expression inputs used previously. This changes retained structure, not measured biological evidence or the data source.',
           normalization='Annotation-only pathway degree normalization and per-channel expected squared norm across all entity pairs. No outcome-dependent gene/pathway selection.',
           missingness='No pathway coverage gives a zero operator increment; base node/product-kernel component remains for all candidates.',
           labels='Only provided sampled pairs enter BCE. No implicit all-matrix negative loss. Expression correlation is not a causal regulation or drug-effect direction.',
           controls='Shuffled context uses one fixed permutation of expression pathway identities, preserving every RNA context value and the drug/annotation inputs. Pooled context collapses pathway-specific coefficients to one per channel.',
           gate='MiDrug both global metrics improve over GNN and kernel-composition control; context must also beat annotation and shuffled context in both MiDrug metrics. LncDrug loss must remain <=0.001 for both metrics; inspect every confirmation partition.',
           innovation='Structured linear/RKHS extension under test. Do not count feature channels, regularization or controls as independent innovations, or claim a newly invented kernel principle.',
           resplits='Existing overlapping development partitions; not independent biological validation.',
           source_sha256={n:base.digest(base.ROOT/n) for n in names},input_sha256={},
           original_test_access=False,outer_query_labels_access=False)
    for n in names:shutil.copy2(base.ROOT/n,out/'source_snapshot'/n)
    gp=sparse.load_npz(EVIDENCE/'gene_pathway.npz')
    dg=sparse.load_npz(EVIDENCE/'drug_gene.npz')
    projection=pathway_projection(gp)
    d=unit_rows(dg@projection).toarray().astype(np.float32)
    dictionary=json.loads((EVIDENCE/'dictionary.json').read_text())
    if gp.shape[1]!=len(dictionary['pathways']) or gp.shape[0]!=len(dictionary['genes']):
        raise ValueError('Pathway/gene identity mismatch')
    for path in (EVIDENCE/'gene_pathway.npz',EVIDENCE/'drug_gene.npz',EVIDENCE/'dictionary.json',EVIDENCE/'expression_programs.npz'):
        p['input_sha256'][str(path)]=base.digest(path)
    for task in p['tasks']:
        rp=unit_rows(sparse.load_npz(EVIDENCE/(task+'_weighted.npz'))@projection).toarray()
        with np.load(EVIDENCE/'expression_programs.npz',allow_pickle=False) as z:
            global_expression=z[task+'_global_pathway'].copy()
            within=z[task+'_within_lineage_pathway'].copy()
        r=np.stack([rp,global_expression,within,np.abs(within)]).astype(np.float32)
        node_path=base.OUT/'inputs'/task/'arrays.npz'
        with np.load(node_path,allow_pickle=False) as z:
            node={k:z[k].copy() for k in ('ur','ud')}
        p['input_sha256'][str(node_path)]=base.digest(node_path)
        p['input_sha256'][str(EVIDENCE/(task+'_weighted.npz'))]=base.digest(EVIDENCE/(task+'_weighted.npz'))
        for case in p['cases']:
            src=node_path if case=='inner' else SPLITS/'inputs'/task/case/'arrays.npz'
            with np.load(src,allow_pickle=False) as z:
                a={k:z[k].copy() for k in ('pairs_train','y_train','pairs_val','y_val')}
            a.update(node,program_rna=r,program_drug=d)
            refs={str(s):base.GRAPH/('screen' if s==666 else 'confirm')/task/'base/c0'/('s'+str(s))/'scores.npz' for s in (666,0,1)} if case=='inner' else {'20260914':SPLITS/'runs'/task/case/'gnn'/'scores.npz'}
            for seed,path in refs.items():
                with np.load(path,allow_pickle=False) as z:
                    names2=('pairs_val','y_val','val') if case=='inner' else ('pairs','y','pred')
                    np.testing.assert_array_equal(a['pairs_val'],z[names2[0]])
                    np.testing.assert_array_equal(a['y_val'],z[names2[1]])
                    a['gnn_'+seed]=z[names2[2]].copy()
                p['input_sha256'][str(path)]=base.digest(path)
            dest=out/'inputs'/task/case/'arrays.npz';dest.parent.mkdir(parents=True)
            np.savez_compressed(dest,**a)
            for path in (src,dest):p['input_sha256'][str(path)]=base.digest(path)
    base.write_json(out/'protocol.json',p);print('FROZEN',out,flush=True)


def protocol(out):
    p=json.loads((Path(out)/'protocol.json').read_text())
    for n,h in p['source_sha256'].items():
        if base.digest(base.ROOT/n)!=h:raise ValueError('Source changed: '+n)
    for n,h in p['input_sha256'].items():
        if base.digest(n)!=h:raise ValueError('Input changed: '+n)
    return p


def fit(a,variant,epochs,lr,device):
    start=time.perf_counter()
    model=PathwayOperator(a,variant).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.)
    pairs=torch.as_tensor(a['pairs_train'],device=device)
    y=torch.as_tensor(a['y_train'],device=device)
    trace=[]
    for epoch in range(1,epochs+1):
        optimizer.zero_grad(set_to_none=True)
        loss=F.binary_cross_entropy_with_logits(model(pairs),y)+model.penalty()/len(y)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite operator objective')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
        if epoch%20==0 or epoch==epochs:trace.append(dict(epoch=epoch,regularized_training_loss=float(loss)))
    model.eval()
    with torch.no_grad():pred=model(torch.as_tensor(a['pairs_val'],device=device)).cpu().numpy()
    return model,pred,trace,time.perf_counter()-start


def verify_files(dest,r):
    for n,h in r['files'].items():
        if base.digest(dest/n)!=h:raise ValueError('Artifact changed: '+str(dest/n))


def run(out,task,cases,device):
    out=Path(out);p=protocol(out);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    for case in cases:
        with np.load(out/'inputs'/task/case/'arrays.npz',allow_pickle=False) as z:a={n:z[n].copy() for n in z.files}
        for variant in p['variants']:
            dest=out/'runs'/task/case/variant
            if (dest/'result.json').exists():
                verify_files(dest,json.loads((dest/'result.json').read_text()));continue
            dest.mkdir(parents=True,exist_ok=True)
            model,pred,trace,seconds=fit(a,variant,p['epochs'][task],p['lr'],device)
            torch.save(model.state_dict(),dest/'model.pt')
            restored=PathwayOperator(a,variant).to(device)
            restored.load_state_dict(torch.load(dest/'model.pt',map_location=device,weights_only=True));restored.eval()
            q=torch.as_tensor(a['pairs_val'],device=device)
            with torch.no_grad():
                replay=restored(q).cpu().numpy()
                reverse=restored(q.flip(1)).cpu().numpy()[::-1]
            np.testing.assert_allclose(replay,pred,atol=2e-6,rtol=2e-6)
            np.testing.assert_array_equal(replay,reverse)
            scores=dict(pred=pred,y=a['y_val'],pairs=a['pairs_val'])
            for n,g in a.items():
                if n.startswith('gnn_'):
                    scores[n]=g;scores['composed_'+n]=.5*g+.5*pred
            measured={n:base.metrics(scores['y'],v,scores['pairs']) for n,v in scores.items() if n not in ('y','pairs')}
            np.savez_compressed(dest/'scores.npz',**scores);base.write_json(dest/'trace.json',trace)
            base.write_json(dest/'result.json',dict(task=task,case=case,variant=variant,seconds=seconds,
                epochs=p['epochs'][task],parameters=sum(v.numel() for v in model.parameters()),metrics=measured,
                replay_max_abs=float(np.max(np.abs(replay-pred))),permutation_checked=True,
                files={n:base.digest(dest/n) for n in ('model.pt','scores.npz','trace.json')}))
            print('COMPLETE',task,case,variant,round(seconds,2),measured,flush=True)


def report(out,cases):
    out=Path(out);p=protocol(out);rows=[]
    for task in p['tasks']:
        for case in cases:
            for variant in p['variants']:
                dest=out/'runs'/task/case/variant
                r=json.loads((dest/'result.json').read_text());verify_files(dest,r)
                with np.load(dest/'scores.npz',allow_pickle=False) as z:
                    for n,old in r['metrics'].items():
                        m=base.metrics(z['y'],z[n],z['pairs'])
                        for k,v in m.items():
                            if abs(v-old[k])>1e-12:raise ValueError('Metric mismatch')
                        rows.append(dict(task=task,case=case,variant=variant,score=n,**m))
    tag='inner' if cases==['inner'] else 'resplits'
    df=pd.DataFrame(rows);df.to_csv(out/(tag+'_metrics.csv'),index=False)
    contrasts=[]
    for task in p['tasks']:
        for case in cases:
            for variant in p['variants']:
                part=df[(df.task==task)&(df.case==case)&(df.variant==variant)]
                new=part[part.score.str.startswith('composed_')];ref=part[part.score.str.startswith('gnn_')]
                row=dict(task=task,case=case,variant=variant)
                for m in ('auc','ap','macro_drug_ap'):
                    row[m]=float(new[m].mean());row['delta_'+m]=float(new[m].mean()-ref[m].mean())
                    row['standalone_'+m]=float(part[part.score=='pred'][m].iloc[0])
                contrasts.append(row)
    c=pd.DataFrame(contrasts);c.to_csv(out/(tag+'_contrasts.csv'),index=False)
    summary=c.groupby(['task','variant'])[['auc','ap','macro_drug_ap','delta_auc','delta_ap','delta_macro_drug_ap','standalone_auc','standalone_ap']].mean()
    summary.to_csv(out/(tag+'_summary.csv'));print(summary.to_string())
    primary=c[c.variant=='context'];mi=primary[primary.task=='MiDrug'];ln=primary[primary.task=='LncDrug']
    controls={}
    for variant in ('kernel','annotation','shuffled_context','pooled_context'):
        controls[variant]={m:float(mi[m].mean()-c[(c.task=='MiDrug')&(c.variant==variant)][m].mean()) for m in ('auc','ap')}
    gate=all(mi['delta_'+m].mean()>0 and ln['delta_'+m].min()>=-.001 for m in ('auc','ap'))
    mechanism=all(v>0 for name,entry in controls.items() for v in entry.values())
    assessment=dict(stage=tag,primary_two_task_performance=bool(gate),mechanism_controls=controls,
                    mechanism_supported=mechanism,shared_gate=bool(gate and mechanism),
                    frontier_innovations_confirmed=0,independent_biological_validation=False,
                    original_test_access=False,interpretation='Named-pathway RKHS extension; outcome and novelty are separate claims.')
    base.write_json(out/(tag+'_ASSESSMENT.json'),assessment)
    base.write_json(out/(tag+'_COMPLETION.json'),dict(runs=len(p['tasks'])*len(cases)*len(p['variants']),
        all_metrics_recomputed=True,all_artifact_hashes_verified=True,
        files={n:base.digest(out/n) for n in (tag+'_metrics.csv',tag+'_contrasts.csv',tag+'_summary.csv',tag+'_ASSESSMENT.json')}))
    print(json.dumps(assessment,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('command',choices=('prepare','run','report'))
    ap.add_argument('--out-dir',default=str(OUT));ap.add_argument('--task',choices=base.TASKS)
    ap.add_argument('--cases',nargs='+',default=['inner']);ap.add_argument('--device',default='cpu');args=ap.parse_args()
    if args.command=='prepare':prepare(args.out_dir)
    elif args.command=='run':run(args.out_dir,args.task,args.cases,args.device)
    else:report(args.out_dir,args.cases)


if __name__=='__main__':main()
