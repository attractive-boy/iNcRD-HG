"""Label-free target-level pathway matching and degree-preserving controls."""
from pathlib import Path

import numpy as np
from scipy import sparse

from improve import tabular_foundation as base
from improve.pathway_context_data import TIERS, WEIGHTS
from improve.tabicl_pathway_modules import (DIRECT_NAMES, ProgramPairs, binary,
                                           l2_normalize, assess as original_assess)


CANDIDATES = ('expanded', 'target_local')
VARIANTS = (*CANDIDATES, 'expanded_rewired', 'target_local_rewired')
STRENGTHS = (.1, .25, .5, 1.)
CONTROL_SEED = 20260913
SWAPS_PER_EDGE = 20
TOP_K = 3


def names(variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown functional evidence variant')
    last = (('target_program_mean', 'target_program_top3') if variant.startswith('target_local')
            else ('drug_gene_cosine', 'drug_program_cosine'))
    return list(DIRECT_NAMES) + [tier+'_'+name for tier in TIERS
        for name in ('rna_present', 'rna_log_target_count', *last)]


def recipes():
    return [dict(id='baseline',variant='baseline',alpha=0.,role='baseline')] + [
        dict(id=variant+'_'+format(alpha,'.2f').replace('.','p'),variant=variant,alpha=alpha,
             role='candidate' if variant in CANDIDATES else 'negative_control')
        for variant in VARIANTS for alpha in STRENGTHS]


def rewire_additions(original, expanded, seed=CONTROL_SEED, swaps_per_edge=SWAPS_PER_EDGE):
    """Bipartite double-edge swaps retain all old edges and both added degrees.

    This is a finite mixing control, not a claim of uniform random graph sampling.
    """
    original, expanded = binary(original), binary(expanded)
    if original.shape != expanded.shape or original.minimum(expanded).nnz != original.nnz:
        raise ValueError('Expanded matrix must contain all original edges')
    added = (expanded-original).tocoo()
    rows, cols = added.row.copy(), added.col.copy()
    fixed = [set(original.indices[original.indptr[i]:original.indptr[i+1]])
             for i in range(original.shape[0])]
    neighbors = [set() for _ in fixed]
    initial = set(zip(rows.tolist(), cols.tolist()))
    for row,col in initial:
        neighbors[row].add(col)
    rng=np.random.default_rng(seed)
    target=int(swaps_per_edge*len(rows));accepted=attempts=0
    while len(rows)>1 and accepted<target and attempts<max(1,target)*100:
        attempts+=1
        i,j=rng.integers(len(rows),size=2)
        a,b,x,y=int(rows[i]),int(rows[j]),int(cols[i]),int(cols[j])
        if a==b or x==y or y in neighbors[a] or x in neighbors[b] or y in fixed[a] or x in fixed[b]:
            continue
        neighbors[a].remove(x);neighbors[a].add(y)
        neighbors[b].remove(y);neighbors[b].add(x)
        cols[i],cols[j]=y,x
        accepted+=1
    randomized=sparse.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=original.shape)
    result=original+randomized
    np.testing.assert_array_equal(result.sum(0),expanded.sum(0))
    np.testing.assert_array_equal(result.sum(1),expanded.sum(1))
    if result.nnz != expanded.nnz or original.minimum(result).nnz != original.nnz:
        raise ValueError('Control altered old edges or edge counts')
    overlap=len(initial & set(zip(rows.tolist(),cols.tolist())))
    info=dict(seed=seed,added_edges=len(rows),target_swaps=target,accepted_swaps=accepted,
              attempts=attempts,unchanged_added_edges=overlap,
              fraction_added_edges_changed=1-overlap/len(rows) if len(rows) else 0.,
              original_edges_preserved=True,drug_degrees_preserved=True,gene_degrees_preserved=True,
              uniform_random_sample_claimed=False)
    return result.tocsr(), info


def verify_files(directory):
    directory=Path(directory)
    manifest=base.read_json(directory/'INPUTS.json')
    if manifest.get('labels_read'):
        raise ValueError('External evidence construction read RNA-drug labels')
    for name,expected in manifest['files'].items():
        if base.digest(directory/name)!=expected:
            raise ValueError('Frozen functional input changed: '+name)
    return manifest


class FunctionalPrograms:
    """All variants have 26 columns and use the same rule for both RNA types."""
    def __init__(self, gp, original, expanded, rewired, tiers):
        self.gp=binary(gp)
        self.original=binary(original)
        self.expanded=binary(expanded)
        self.rewired=binary(rewired)
        self.tiers=tiers
        for mat in (self.original,self.expanded,self.rewired):
            if mat.shape != self.original.shape or mat.shape[1]!=self.gp.shape[0]:
                raise ValueError('Drug/gene dictionary mismatch')
        for layers in tiers.values():
            if set(layers)!=set(TIERS):
                raise ValueError('Missing evidence tier')
            shape=layers['functional'].shape
            if shape[1]!=self.gp.shape[0] or any(m.shape!=shape for m in layers.values()):
                raise ValueError('RNA/gene dictionary mismatch')
            if any(not np.isfinite(m.data).all() or (m.data<0).any() for m in layers.values()):
                raise ValueError('Invalid RNA evidence')
        self.tables={}

    @classmethod
    def load(cls,reference,drug_inputs,rewired_path):
        reference,drug_inputs=Path(reference),Path(drug_inputs)
        verify_files(reference);manifest=verify_files(drug_inputs)
        if base.digest(reference/'INPUTS.json')!=manifest['reference_manifest_sha256']:
            raise ValueError('Expansion uses another reference dictionary')
        dictionary=base.read_json(reference/'dictionary.json')
        load=lambda name:sparse.load_npz(reference/(name+'.npz'))
        result=cls(load('gene_pathway'),load('drug_gene'),
                   sparse.load_npz(drug_inputs/'drug_gene_expanded.npz'),sparse.load_npz(rewired_path),
                   {task:{tier:load(task+'_'+tier) for tier in TIERS} for task in dictionary['rna_names']})
        if result.expanded.shape!=(len(dictionary['drug_names']),len(dictionary['genes'])):
            raise ValueError('Frozen drug dictionary mismatch')
        for task in result.tiers:
            if result.tiers[task]['functional'].shape[0]!=len(dictionary['rna_names'][task]):
                raise ValueError('Frozen RNA dictionary mismatch')
        return result

    def table(self,task,variant):
        if task not in self.tiers or variant not in VARIANTS:
            raise ValueError('Unknown task or variant')
        if (task,variant) in self.tables:
            return self.tables[task,variant]
        dg=self.rewired if variant.endswith('_rewired') else self.expanded
        layers=self.tiers[task]
        weighted=sparse.csr_matrix(layers['functional'].shape,dtype=np.float64)
        for tier in TIERS:
            weighted=weighted.maximum(layers[tier]*WEIGHTS[tier])
        kernel=ProgramPairs(self.gp,dg,{task:weighted},{task:sparse.csr_matrix(weighted.shape)})
        main,_=kernel.group(weighted)
        extra=[]
        for tier in TIERS:
            table,program=kernel.group(layers[tier])
            if variant.startswith('target_local'):
                # RNA program -> each individual drug target's pathway profile.
                # No direction, causal effect, or measured pathway activity is inferred.
                columns=table[:,:,[0,2]].copy()
                targets=np.unique(dg.indices)
                similarity=(program @ l2_normalize(kernel.projection[targets]).T).toarray()
                lookup=np.full(dg.shape[1],-1,dtype=int);lookup[targets]=np.arange(len(targets))
                local=np.zeros((*table.shape[:2],2),dtype=np.float32)
                for drug in range(dg.shape[0]):
                    genes=dg.indices[dg.indptr[drug]:dg.indptr[drug+1]]
                    if not len(genes):
                        continue
                    scores=similarity[:,lookup[genes]]
                    k=min(TOP_K,len(genes))
                    local[:,drug,0]=scores.mean(1)
                    local[:,drug,1]=np.partition(scores,len(genes)-k,axis=1)[:,-k:].mean(1)
                extra.append(np.concatenate([columns,local],axis=-1))
            else:
                extra.append(table[:,:,[0,2,5,11]])
        result=np.concatenate([main,*extra],axis=-1).astype(np.float32)
        if result.shape[-1]!=len(names(variant)) or not np.isfinite(result).all():
            raise ValueError('Invalid functional features')
        self.tables[task,variant]=result
        return result

    def transform(self,task,pairs,variant):
        pairs=np.asarray(pairs)
        if pairs.ndim!=2 or pairs.shape[0]!=2 or not np.issubdtype(pairs.dtype,np.integer):
            raise ValueError('Expected integer RNA/drug indices with shape (2,n)')
        table=self.table(task,variant)
        if (pairs<0).any() or (pairs[0]>=table.shape[0]).any() or (pairs[1]>=table.shape[1]).any():
            raise ValueError('Entity index outside frozen dictionary')
        return table[pairs[0],pairs[1]].copy()


def compose(baseline,graph_baseline,augmented,reference_alpha,strength):
    """Gradually replace only the accepted evidence branch; graph mass is fixed."""
    if not (0<=strength<=1 and 0<=reference_alpha<=1):
        raise ValueError('Invalid mixture coefficient')
    branch=(1-reference_alpha)*graph_baseline+reference_alpha*augmented
    return (1-strength)*baseline+strength*branch


def assess(rows,config,seeds):
    result=original_assess(rows,config,seeds)
    result['eligible']=(result['eligible'] and config['variant'] in CANDIDATES and
        all(result['metrics']['LncDrug'][metric]['nonnegative_seeds']==len(seeds) for metric in ('auc','ap')))
    result['lnc_per_seed_required']=True
    result['role']=config['role']
    return result
