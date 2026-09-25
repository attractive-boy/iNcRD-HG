"""Shared, label-free gene/program pair features and a two-task acceptance gate."""
import json
from pathlib import Path

import numpy as np
from scipy import sparse

from improve import tabular_foundation as base


DIRECT_NAMES = (
    'rna_gene_present', 'drug_gene_present', 'rna_gene_log_count', 'drug_gene_log_count',
    'shared_gene_log_count', 'gene_cosine', 'drug_gene_coverage', 'rna_gene_coverage',
    'rna_pathway_log_count', 'drug_pathway_log_count', 'shared_pathway_log_count',
    'pathway_program_cosine', 'drug_pathway_coverage', 'rna_pathway_coverage',
)
NEIGHBOR_NAMES = (
    'neighbor_gene_present', 'neighbor_gene_log_count', 'neighbor_gene_cosine',
    'neighbor_drug_gene_coverage', 'neighbor_pathway_log_count',
    'neighbor_program_cosine', 'neighbor_drug_pathway_coverage',
    'direct_neighbor_program_consistency',
)
VARIANTS = ('direct', 'cross_rna')
STRENGTHS = (.05, .1, .25, .5, 1.)


def l2_normalize(matrix):
    matrix = sparse.csr_matrix(matrix, dtype=np.float64)
    norm = np.sqrt(np.asarray(matrix.multiply(matrix).sum(1)).ravel())
    inverse = np.divide(1., norm, out=np.zeros_like(norm), where=norm > 0)
    return sparse.diags(inverse).dot(matrix).tocsr()


def binary(matrix):
    result = sparse.csr_matrix(matrix, dtype=np.float64).copy()
    result.eliminate_zeros()
    result.data[:] = 1.
    return result


def divide(num, denom):
    num, denom = np.broadcast_arrays(np.asarray(num, dtype=np.float64), np.asarray(denom, dtype=np.float64))
    return np.divide(num, denom, out=np.zeros_like(num), where=denom > 0)


class ProgramPairs:
    """Same definition for both RNA types; no RNA–drug labels accepted here."""
    def __init__(self, gene_pathway, drug_gene, rna_gene, neighbor_gene):
        self.gp = binary(gene_pathway)
        self.dg = binary(drug_gene)
        self.rg = {t: sparse.csr_matrix(x, dtype=np.float64) for t,x in rna_gene.items()}
        self.ng = {t: sparse.csr_matrix(x, dtype=np.float64) for t,x in neighbor_gene.items()}
        if not self.rg or set(self.rg) != set(self.ng):
            raise ValueError('Direct and neighborhood evidence must cover the same tasks')
        g = self.gp.shape[0]
        if self.dg.shape[1] != g:
            raise ValueError('Drug/gene dictionary mismatch')
        for task, matrix in self.rg.items():
            if matrix.shape[1] != g or self.ng[task].shape != matrix.shape:
                raise ValueError('RNA/gene dictionary mismatch')
        for matrix in [self.gp,self.dg,*self.rg.values(),*self.ng.values()]:
            if not np.isfinite(matrix.data).all() or (matrix.data < 0).any():
                raise ValueError('Evidence must be finite and nonnegative')
        gene_degree = np.asarray(self.gp.sum(1)).ravel()
        path_size = np.asarray(self.gp.sum(0)).ravel()
        self.projection = (sparse.diags(divide(1.,np.sqrt(gene_degree))) @ self.gp @
                           sparse.diags(divide(1.,np.sqrt(path_size)))).tocsr()
        self.drug_program = self.program(self.dg)
        self.tables = {}

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        manifest = base.read_json(directory/'INPUTS.json')
        for name, expected in manifest['files'].items():
            if base.digest(directory/name) != expected:
                raise ValueError('Frozen pathway evidence changed: '+name)
        load = lambda name: sparse.load_npz(directory/(name+'.npz'))
        dictionaries = base.read_json(directory/'dictionary.json')
        tasks = dictionaries['rna_names']
        result = cls(load('gene_pathway'), load('drug_gene'),
                     {t:load(t+'_gene') for t in tasks},
                     {t:load(t+'_indirect_gene') for t in tasks})
        if len(dictionaries['genes']) != result.gp.shape[0] or len(dictionaries['pathways']) != result.gp.shape[1]:
            raise ValueError('Annotation dictionary shape mismatch')
        for task in tasks:
            if len(dictionaries['rna_names'][task]) != result.rg[task].shape[0]:
                raise ValueError('RNA dictionary shape mismatch')
        if len(dictionaries['drug_names']) != result.dg.shape[0]:
            raise ValueError('Drug dictionary shape mismatch')
        return result

    def program(self, genes):
        # L2 normalization cancels per-RNA total mass. Ubiquitous genes and
        # large pathways are downweighted using annotation-only degrees.
        return l2_normalize(genes @ self.projection)

    def group(self, genes):
        r = binary(genes)
        nr, nd = r.shape[0], self.dg.shape[0]
        rc = np.asarray(r.sum(1)).ravel()[:,None]
        dc = np.asarray(self.dg.sum(1)).ravel()[None,:]
        common = (r @ self.dg.T).toarray()
        cosine = (l2_normalize(genes) @ l2_normalize(self.dg).T).toarray()
        rp = self.program(genes)
        rb, db = binary(rp), binary(self.drug_program)
        rpc, dpc = np.asarray(rb.sum(1)).ravel()[:,None], np.asarray(db.sum(1)).ravel()[None,:]
        shared = (rb @ db.T).toarray()
        pc = (rp @ self.drug_program.T).toarray()
        columns = [rc>0,dc>0,np.log1p(rc),np.log1p(dc),np.log1p(common),cosine,
                   divide(common,dc),divide(common,rc),np.log1p(rpc),np.log1p(dpc),
                   np.log1p(shared),pc,divide(shared,dpc),divide(shared,rpc)]
        table = np.stack([np.broadcast_to(x,(nr,nd)) for x in columns],axis=-1).astype(np.float32)
        if not np.isfinite(table).all():
            raise ValueError('Nonfinite pair features')
        return table, rp

    def table(self, task, variant):
        if variant not in VARIANTS or task not in self.rg:
            raise ValueError('Unregistered task or variant')
        key = task, variant
        if key in self.tables:
            return self.tables[key]
        direct, direct_program = self.group(self.rg[task])
        self.tables[task,'direct'] = direct
        if variant == 'direct':
            return direct
        neighbor, neighbor_program = self.group(self.ng[task])
        consistency = np.asarray(direct_program.multiply(neighbor_program).sum(1)).ravel()
        selected = neighbor[:,:,[0,2,5,6,8,11,12]]
        consistency = np.broadcast_to(consistency[:,None,None], (*direct.shape[:2],1))
        result = np.concatenate([direct,selected,consistency],axis=-1).astype(np.float32)
        self.tables[key] = result
        return result

    def transform(self, task, pairs, variant):
        pairs = np.asarray(pairs)
        if pairs.ndim != 2 or pairs.shape[0] != 2 or not np.issubdtype(pairs.dtype, np.integer):
            raise ValueError('Expected integer RNA/drug indices, shape (2,n)')
        table = self.table(task,variant)
        if (pairs < 0).any() or (pairs[0]>=table.shape[0]).any() or (pairs[1]>=table.shape[1]).any():
            raise ValueError('Entity index outside frozen dictionary')
        return table[pairs[0],pairs[1]].copy()


def names(variant):
    if variant not in VARIANTS:
        raise ValueError('Unknown variant')
    return list(DIRECT_NAMES + (NEIGHBOR_NAMES if variant == 'cross_rna' else ()))


def recipes():
    return [dict(id='baseline',variant='baseline',alpha=0.)] + [
        dict(id=f'{variant}_{round(100*a):03d}',variant=variant,alpha=a)
        for variant in VARIANTS for a in STRENGTHS]


def assess(rows, config, seeds):
    """No tolerance below zero: LncDrug AUC and AP must both be preserved."""
    detail = {}
    for task in ('MiDrug','LncDrug'):
        b = {r['seed']:r for r in rows if r['task']==task and r['recipe']=='baseline'}
        c = {r['seed']:r for r in rows if r['task']==task and r['recipe']==config['id']}
        if set(b) != set(seeds) or set(c) != set(seeds):
            raise ValueError('Incomplete paired seed results')
        detail[task] = {}
        for metric in ('auc','ap','macro_drug_ap'):
            bv = np.array([b[s][metric] for s in seeds],dtype=float)
            cv = np.array([c[s][metric] for s in seeds],dtype=float)
            if metric != 'macro_drug_ap' and not np.isfinite(np.r_[bv,cv]).all():
                raise ValueError('Nonfinite primary metric')
            detail[task][metric] = dict(mean=float(cv.mean()),baseline_mean=float(bv.mean()),
                                        delta=float(cv.mean()-bv.mean()),positive_seeds=int((cv>bv).sum()),
                                        nonnegative_seeds=int((cv>=bv).sum()))
    eligible = (config['variant'] != 'baseline' and
                all(detail['LncDrug'][m]['delta'] >= 0 for m in ('auc','ap')) and
                all(detail['MiDrug'][m]['delta'] > 0 and detail['MiDrug'][m]['positive_seeds'] >= 2
                    for m in ('auc','ap')))
    return dict(recipe=config,eligible=eligible,metrics=detail)
