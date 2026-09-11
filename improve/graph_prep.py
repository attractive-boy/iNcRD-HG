# Rebuild the frozen folds with a configurable similarity graph.
#
# `heter_data_prepare.py` hard-codes ONE threshold for all three similarity
# relations:
#
#     lncRNA_edge_ken = homo_sim_net('LncSimMat_ken.csv', 0.3)   # line 51
#     miRNA_edge_ken  = homo_sim_net('MiSimMat_ken.csv',  0.3)   # line 77
#     drug_edge       = homo_sim_net('drugSimMat.csv',    0.3)    # line 98
#
# and `find_best_params.py` sweeps lr / epochs / channels / aggr -- so graph
# construction is the one part of the method that was never tuned at all.  At
# 0.3 the three relations come out at mean degree 5.6 / 25.8 / 2.9, an order of
# magnitude apart, because one cutoff is being applied to three similarity
# matrices with different distributions.  DrugDrug is the starved one, and the
# drug side is where 99.7% of this metric's ranking pairs live.
#
# This matters for S_val specifically: the validation fold is scored by message
# passing over the TRAINING graph, and these similarity edges are in it.
# Changing them changes the validation prediction without touching a single
# label.
#
# The split must not move, or the numbers stop being comparable to the paper.
# RandomLinkSplit only ever partitions the (ncRNA, edge_type, drug) relation and
# samples its negatives over the ncRNA x drug node space; the similarity edges
# feed neither.  `--verify` asserts that -- it reloads the released folds and
# requires edge_label_index and edge_label to match element-for-element.
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData

SEED = 666
RELATIONS = [['lncRNA', 'LncDrug'], ['miRNA', 'MiDrug']]


def seed_all():
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True


def sim_edges(path, thr=None, knn=None):
    """Sparsify a similarity matrix.

    `thr` is the released rule: keep |sim| > thr, which gives every node a
    degree proportional to how similar its row happens to be to everything
    else.  `knn` keeps each node's k most similar partners instead, so degree
    is uniform across nodes and the three relations are comparable to each
    other -- which the single global cutoff does not give.
    """
    m = np.abs(pd.read_csv(path, header=None).values)
    if knn is not None:
        n = m.shape[0]
        mm = m.copy()
        # The matrices contain NaN (LncSimMat_ken has two per row); the
        # threshold path filters them (NaN > t is False) but argpartition
        # would let them win a top-k slot, and a NaN weight would poison the
        # message passing.  argpartition sorts -mm ascending, so NaN must
        # become -inf here (i.e. +inf in the negated space) to sort to the
        # bottom and never be selected.
        mm[np.isnan(mm)] = -np.inf
        np.fill_diagonal(mm, -np.inf)
        idx = np.argpartition(-mm, kth=min(knn, n - 1) - 1, axis=1)[:, :knn]
        rows = np.repeat(np.arange(n), idx.shape[1])
        cols = idx.ravel()
        return np.vstack((rows, cols))
    rows, cols = np.where(m > thr)
    return np.vstack((rows, cols))


def signed_edges(path, thr=None, knn=None):
    """Like `sim_edges` but on the SIGNED matrix, returning (pos, neg) pairs.

    The released pipeline takes `np.abs` of every similarity matrix before
    thresholding, so an anti-correlated pair (sim < -thr) enters the graph as a
    positive-similarity edge and is message-passed identically to a strongly
    co-expressed pair.  `signed_edges` keeps the sign: threshold mode returns
    (sim > thr) and (sim < -thr) edge pairs; knn mode selects each node's k
    most similar partners by |sim| and then splits them by sign.  The sign is
    recoverable afterwards from the matrix, so the caller can weight the two
    relations differently.
    """
    m = pd.read_csv(path, header=None).values
    if knn is not None:
        n = m.shape[0]
        am = np.abs(m)
        am[np.isnan(am)] = -np.inf
        np.fill_diagonal(am, -np.inf)
        idx = np.argpartition(-am, kth=min(knn, n - 1) - 1, axis=1)[:, :knn]
        rows = np.repeat(np.arange(n), idx.shape[1])
        cols = idx.ravel()
        vals = m[rows, cols]
        pos = np.vstack((rows[vals > 0], cols[vals > 0]))
        neg = np.vstack((rows[vals < 0], cols[vals < 0]))
        return pos, neg
    rows, cols = np.where(m > thr)
    pos = np.vstack((rows, cols))
    rows, cols = np.where(m < -thr)
    neg = np.vstack((rows, cols))
    return pos, neg


def build(net_dir, lnc, mi, drug, knn=False, signed=False):
    """Reproduce construct_heterogeneous_network with configurable sparsity.

    `signed=True` replaces each similarity relation with `_pos` and `_neg`
    relations (from `signed_edges`), so the anti-correlation that the released
    abs() discards survives as its own edge type.
    """
    kw = (lambda v: dict(knn=int(v))) if knn else (lambda v: dict(thr=float(v)))

    lncRNA_names = pd.read_csv(f'{net_dir}/lncRNA.csv')
    miRNA_names = pd.read_csv(f'{net_dir}/miRNA.csv')
    drug_names = pd.read_csv(f'{net_dir}/Drug.csv').iloc[:, 1]
    lnc_map = {n: i for i, n in lncRNA_names['x'].items()}
    mi_map = {n: i for i, n in miRNA_names['x'].items()}
    drug_map = {n: i for i, n in drug_names.items()}

    ld = pd.read_csv(f'{net_dir}/LncDrug_edge.csv').iloc[:, [0, 4]]
    ld['ncRNA_Name'] = ld['ncRNA_Name'].map(lnc_map)
    ld['CID'] = ld['CID'].map(drug_map)
    md = pd.read_csv(f'{net_dir}/MiDrug_edge.csv').iloc[:, [0, 4]]
    md['ncRNA_Name'] = md['ncRNA_Name'].map(mi_map)
    md['CID'] = md['CID'].map(drug_map)
    lm = pd.read_csv(f'{net_dir}/LncMiRNA_interaction.csv')
    lm['lncRNA'] = lm['lncRNA'].map(lnc_map)
    lm['miRNA'] = lm['miRNA'].map(mi_map)

    # Feature construction is verbatim, including the 1e-8 that is on the
    # lncRNA denominator and not the other two.
    lnc_f = torch.tensor(
        pd.read_csv(f'{net_dir}/LncExp.csv', index_col=0).T.values,
        dtype=torch.float)
    mi_df = pd.read_csv(f'{net_dir}/MiExp.csv', index_col=0).T
    mi_df = mi_df[sorted(mi_df.columns, key=lambda c: int(c.split('-')[-1]))]
    mi_f = torch.tensor(mi_df.values, dtype=torch.float)
    dr_f = torch.tensor(
        pd.read_csv(f'{net_dir}/drug_features-chemBertA.csv',
                    index_col=0).values, dtype=torch.float)

    data = HeteroData()
    data['lncRNA'].node_id = torch.arange(len(lncRNA_names))
    data['miRNA'].node_id = torch.arange(len(miRNA_names))
    data['drug'].node_id = torch.arange(len(drug_names))
    data['lncRNA'].x = (lnc_f - lnc_f.min(1, keepdim=True).values) / (
        lnc_f.max(1, keepdim=True).values - lnc_f.min(1, keepdim=True).values
        + 1e-8)
    data['miRNA'].x = (mi_f - mi_f.min(1, keepdim=True).values) / (
        mi_f.max(1, keepdim=True).values - mi_f.min(1, keepdim=True).values)
    data['drug'].x = (dr_f - dr_f.min(1, keepdim=True).values) / (
        dr_f.max(1, keepdim=True).values - dr_f.min(1, keepdim=True).values)

    if signed:
        for nt, rel, fname in (('lncRNA', 'LncLnc', 'LncSimMat_ken.csv'),
                               ('miRNA', 'MiMi', 'MiSimMat_ken.csv'),
                               ('drug', 'DrugDrug', 'drugSimMat.csv')):
            pos, neg = signed_edges(f'{net_dir}/{fname}', **kw(
                {'lncRNA': lnc, 'miRNA': mi, 'drug': drug}[nt]))
            data[nt, f'{rel}_pos', nt].edge_index = torch.tensor(
                pos, dtype=torch.long)
            data[nt, f'{rel}_neg', nt].edge_index = torch.tensor(
                neg, dtype=torch.long)
    else:
        data['lncRNA', 'LncLnc', 'lncRNA'].edge_index = torch.tensor(
            sim_edges(f'{net_dir}/LncSimMat_ken.csv', **kw(lnc)), dtype=torch.long)
        data['miRNA', 'MiMi', 'miRNA'].edge_index = torch.tensor(
            sim_edges(f'{net_dir}/MiSimMat_ken.csv', **kw(mi)), dtype=torch.long)
        data['drug', 'DrugDrug', 'drug'].edge_index = torch.tensor(
            sim_edges(f'{net_dir}/drugSimMat.csv', **kw(drug)), dtype=torch.long)
    data['lncRNA', 'LncDrug', 'drug'].edge_index = torch.tensor(
        ld.T.values, dtype=torch.long)
    data['miRNA', 'MiDrug', 'drug'].edge_index = torch.tensor(
        md.T.values, dtype=torch.long)
    data['lncRNA', 'LncMi', 'miRNA'].edge_index = torch.tensor(
        lm.T.values, dtype=torch.long)
    return T.AddSelfLoops()(T.ToUndirected()(data))


def split_all(data, device):
    """The released loop, in order.  LncDrug is transformed first and both
    relations draw from the same generator, so the order fixes MiDrug's split
    as well and must not be changed."""
    data = data.to(device=device)
    out = {}
    for rna, et in RELATIONS:
        tr = T.RandomLinkSplit(
            num_val=0.2, num_test=0.2, neg_sampling_ratio=2.0,
            is_undirected=True, add_negative_train_samples=True,
            edge_types=(rna, et, 'drug'),
            rev_edge_types=('drug', 'rev_' + et, rna))
        out[et] = tr(data)
    return out


def verify(splits, ref_dir):
    """The split must be element-for-element what the release shipped."""
    for rna, et in RELATIONS:
        key = (rna, et, 'drug')
        for name, d in zip(('train', 'val', 'test'), splits[et]):
            ref = torch.load(f'{ref_dir}/{et}_{name}_data.pth',
                             map_location='cpu')
            a, b = d[key].cpu(), ref[key]
            assert torch.equal(a.edge_label_index, b.edge_label_index), \
                f'{et}/{name}: edge_label_index moved'
            assert torch.equal(a.edge_label, b.edge_label), \
                f'{et}/{name}: edge_label moved'
            assert torch.equal(a.edge_index, b.edge_index), \
                f'{et}/{name}: target-relation message edges moved'
        print(f'  {et}: split identical to the release')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--net-dir', required=True, help='2.construct_network')
    p.add_argument('--out-dir', required=True, help='writes <out>/3.heter_data')
    p.add_argument('--lnc', default=0.3)
    p.add_argument('--mi', default=0.3)
    p.add_argument('--drug', default=0.3)
    p.add_argument('--knn', action='store_true',
                   help='treat --lnc/--mi/--drug as per-node k, not thresholds')
    p.add_argument('--verify-against', default=None,
                   help='a 3.heter_data to assert the split matches')
    args = p.parse_args()

    seed_all()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = build(args.net_dir, args.lnc, args.mi, args.drug, knn=args.knn)
    for r in [('lncRNA', 'LncLnc', 'lncRNA'), ('miRNA', 'MiMi', 'miRNA'),
              ('drug', 'DrugDrug', 'drug')]:
        print(f'  {r[1]:9s} {data[r].edge_index.size(1):7d} edges '
              f'(mean deg {data[r].edge_index.size(1)/data[r[0]].num_nodes:.1f})')
    splits = split_all(data, device)
    if args.verify_against:
        verify(splits, args.verify_against)

    out = f'{args.out_dir}/3.heter_data'
    os.makedirs(out, exist_ok=True)
    for et, (tr, va, te) in splits.items():
        for name, d in (('train', tr), ('val', va), ('test', te)):
            torch.save(d, f'{out}/{et}_{name}_data.pth')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
