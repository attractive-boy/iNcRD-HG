"""Run SubKNet's own model (RW_layer graph kernel + DenseGCN global branch) on the
MPHGNN miRNA-drug benchmark, to test whether the official network reproduces.

Uses SubKNet's code path (subgraph_extraction.links2subgraphs, model.Model,
prediction.preds, utils.comp_feat) unchanged; only the data is the MPHGNN
benchmark, formatted as <dataset>/sims_m.txt, sims_d.txt, inters.txt.
"""
import os, sys, argparse
import numpy as np
import torch

CODE = '/tmp/opencode/subknet_repo/code'
os.chdir(CODE)
sys.path.insert(0, CODE)

ap = argparse.ArgumentParser()
ap.add_argument('--seeds', type=int, nargs='+', default=[666, 0, 1])
ap.add_argument('--epochs', type=int, default=25)
ap.add_argument('--folds', type=int, default=5)
ap.add_argument('--lr', type=float, default=1e-2)
ap.add_argument('--batch_size', type=int, default=128)
ap.add_argument('--device', default='cuda:1')
a = ap.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = a.device.split(':')[-1]
sys.argv = ['subknet', '--dataset', 'MiDrugMPH', '--types', 'hete', '--epochs', str(a.epochs),
            '--lr', str(a.lr), '--batch_size', str(a.batch_size), '--fixs', '0', '--max_step', '2',
            '--gcn_layer', '1', '--dropout_rate', '0.2', '--lambda1', '0.5', '--lambda2', '0.5', '--lambda3', '0.5']

from parameters import args_parser
from utils import comp_feat, comput_scores
from subgraph_extraction import links2subgraphs
from model import Model
import prediction as _prediction
_prediction.ProgressBar.update = lambda self, *a, **k: None  # silence per-batch bar
from prediction import preds
from sklearn.metrics import roc_auc_score, average_precision_score

args = args_parser()
DEV = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

D = torch.load('/tmp/opencode/mphgnn_repo/MiDrug_data_end.pth', weights_only=False).cpu()
NM, ND = D['miRNA'].num_nodes, D['drug'].num_nodes
bench = D['miRNA', 'MiDrug', 'drug'].edge_index.numpy()
tp = D['miRNA', 'MiDrug', 'drug'].edge_label_index.numpy()
ty = D['miRNA', 'MiDrug', 'drug'].edge_label.numpy()
xmi, xdr = D['miRNA'].x.numpy().astype(np.float64), D['drug'].x.numpy().astype(np.float64)


def cos_sim(x):
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    return x @ x.T


os.makedirs('../data/MiDrugMPH', exist_ok=True)
np.savetxt('../data/MiDrugMPH/sims_m.txt', cos_sim(xmi), delimiter='\t')
np.savetxt('../data/MiDrugMPH/sims_d.txt', cos_sim(xdr), delimiter='\t')
MD_full = np.zeros((NM, ND))
MD_full[bench[0], bench[1]] = 1
np.savetxt('../data/MiDrugMPH/inters.txt', MD_full, delimiter='\t')
print('data written', flush=True)

pos_set = set(map(tuple, bench.T.tolist()))
test_pos = tp[:, ty == 1]
test_neg = tp[:, ty == 0]


def make_edges(pairs, labels):
    return [(int(pairs[0, i]), int(pairs[1, i]), int(labels[i])) for i in range(pairs.shape[1])]


test_edges = make_edges(tp, ty)

results = []
full_metrics_all = []
for seed in a.seeds:
    rng = np.random.default_rng(seed)
    negs = []
    while len(negs) < bench.shape[1]:
        i, j = int(rng.integers(0, NM)), int(rng.integers(0, ND))
        if (i, j) not in pos_set:
            negs.append((i, j))
    train_val = [(int(bench[0, k]), int(bench[1, k]), 1) for k in range(bench.shape[1])] + \
                [(i, j, 0) for i, j in negs]
    order = np.random.default_rng(seed + 1).permutation(len(train_val))
    train_val = [train_val[k] for k in order]

    folds = np.array_split(np.arange(len(train_val)), a.folds)
    test_aucs, test_aprs = [], []
    for f, val_idx in enumerate(folds):
        train_idx = np.setdiff1d(np.arange(len(train_val)), val_idx)
        MD = np.copy(MD_full)
        # mask val positives and test positives out of the observed graph
        for k in val_idx:
            e = train_val[k]
            if e[2] == 1:
                MD[e[0], e[1]] = 0
        for e in test_edges:
            if e[2] == 1:
                MD[e[0], e[1]] = 0
        MD2 = MD
        featM, featD, featM_sim, featD_sim = comp_feat(MD2, 'MiDrugMPH')
        train_pos = np.where(MD2 == 1)
        tz = np.array([[train_val[k][0], train_val[k][1]] for k in train_idx if train_val[k][2] == 0])
        vp = np.array([[train_val[k][0], train_val[k][1]] for k in val_idx if train_val[k][2] == 1])
        vz = np.array([[train_val[k][0], train_val[k][1]] for k in val_idx if train_val[k][2] == 0])
        train_zero = (tz[:, 0], tz[:, 1])
        val_pos = (vp[:, 0], vp[:, 1])
        val_zero = (vz[:, 0], vz[:, 1])
        test_pos_e = (test_pos[0], test_pos[1])
        test_neg_e = (test_neg[0], test_neg[1])
        tr, va, te, max_n = links2subgraphs(MD2, train_pos, train_zero, val_pos, val_zero, test_pos_e, test_neg_e,
                                            h=1, max_nodes_per_hop=None, featM=featM, featD=featD)
        size_sub = int(np.mean([x[0].shape[0] for x in tr]))
        model = Model(featM_sim.shape[1], featD_sim.shape[1], tr[0][1].shape[1], 2, args,
                      max_step=args.max_step, dropout_rate=args.dropout_rate, size_graph_filter=[size_sub])
        model = model.cuda()
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=5e-4)
        crit = torch.nn.CrossEntropyLoss()
        yy_tr = [x[3] for x in tr]; adj_tr = [x[0] for x in tr]
        f_tr = [x[1] for x in tr]; n_tr = [x[2] for x in tr]
        yy_va = [x[3] for x in va]; adj_va = [x[0] for x in va]
        f_va = [x[1] for x in va]; n_va = [x[2] for x in va]
        best = (-1, None); counter = 0
        for ep in range(a.epochs):
            model.train()
            _, _, _, cv_tr = preds(MD2, adj_tr, f_tr, n_tr, yy_tr, featM_sim, featD_sim, args.batch_size, ep, model,
                                   optimizer=opt, criterion=crit)
            model.eval()
            _, _, _, cv = preds(MD2, adj_va, f_va, n_va, yy_va, featM_sim, featD_sim, args.batch_size, ep,
                                model, optimizer=None, criterion=crit)
            if cv[1] > best[0]:
                best = (cv[1], {k: v.detach().clone() for k, v in model.state_dict().items()})
                counter = 0
            else:
                counter += 1
                if counter >= 20:
                    break
        model.load_state_dict(best[1]); model.eval()
        yy_te = [x[3] for x in te]
        adj_te = [x[0] for x in te]; f_te = [x[1] for x in te]; n_te = [x[2] for x in te]
        _, out_te, _, cv_te = preds(MD2, adj_te, f_te, n_te, yy_te, featM_sim, featD_sim, a.batch_size, 0, model,
                                    optimizer=None, criterion=crit)
        test_aprs.append(cv_te[0]); test_aucs.append(cv_te[1])
        # full threshold metrics at best-F1 (class-1 probability)
        _sc = np.asarray(out_te)[:, 0]; _lb = np.asarray(yy_te)
        from sklearn.metrics import (precision_recall_curve, confusion_matrix,
                                     matthews_corrcoef)
        _pr, _rc, _th = precision_recall_curve(_lb, _sc)
        _f1 = np.where((_pr + _rc) > 0, 2 * _pr * _rc / (_pr + _rc + 1e-12), 0)
        _t = _th[int(np.argmax(_f1[:-1]))] if len(_th) else 0.5
        _pd = (_sc >= _t).astype(int)
        _tn, _fp, _fn, _tp = confusion_matrix(_lb, _pd, labels=[0, 1]).ravel()
        full_metrics_all.append((( _tp + _tn) / len(_lb), _tp / (_tp + _fp + 1e-12),
                                 _tp / (_tp + _fn + 1e-12),
                                 2 * _tp / (2 * _tp + _fp + _fn + 1e-12),
                                 float(matthews_corrcoef(_lb, _pd))))
        print(f'seed {seed} fold {f}: val AUC {best[0]:.4f} test AUC/AUPR {test_aucs[-1]:.4f}/{test_aprs[-1]:.4f}', flush=True)
    results.append((np.mean(test_aucs), np.mean(test_aprs)))
    print(f'seed {seed}: AUC {results[-1][0]:.4f} AUPR {results[-1][1]:.4f}', flush=True)

r = np.array(results)
print('\n===== SubKNet official model on MPHGNN benchmark (5-fold, independent test) =====')
print('AUC %.4f±%.4f  AUPR %.4f±%.4f' % (r[:, 0].mean(), r[:, 0].std(), r[:, 1].mean(), r[:, 1].std()))
fm = np.array(full_metrics_all); print('SubKNet self-run threshold metrics (bestF1): ACC %.4f Prec %.4f Rec %.4f F1 %.4f MCC %.4f' % (fm[:,0].mean(), fm[:,1].mean(), fm[:,2].mean(), fm[:,3].mean(), fm[:,4].mean()))
print('SubKNet reported  AUC 0.9717  AUPR 0.9341')
print('MPHGNN reported   AUC 0.9780  AUPR 0.9370')
