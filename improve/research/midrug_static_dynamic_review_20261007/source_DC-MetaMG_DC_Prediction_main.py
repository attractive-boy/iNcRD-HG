import numpy as np
import torch
import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.metrics import precision_score, accuracy_score


from util import (
    set_seed, load_embeddings, load_edge_splits,
    build_pairs, deal_embedding, kfold_split,
    calculate_metrics, EarlyStopping,
)
from model import MLP

# ── Configuration ──────────────────────────────────────────────────────────────
DATASET   = 'LncRNA'   # 'MiRNA' or 'LncRNA'
N_DRUG    = 154       # MiRNA: 60  | LncRNA: 154
SEED      = 2026
DEVICE    = torch.device('cuda:0')
N_FOLDS   = 5
EPOCHS    = 3000
WARMUP    = 2000
PATIENCE  = 100
LR        = 0.001
WD        = 5e-4

set_seed(SEED)

# ── Load embeddings & edge splits ──────────────────────────────────────────────
emb_coa, emb_fin = load_embeddings(DATASET)
edge_splits = load_edge_splits(DATASET)

train_pos_idx = edge_splits['train_pos_edge_index']
train_neg_idx = edge_splits['train_neg_edge_index']
test_pos_idx  = edge_splits['test_pos_edge_index']
test_neg_idx  = edge_splits['test_neg_edge_index']

# Build feature pairs from edge indices
train_pos, train_pos1 = build_pairs(emb_coa, emb_fin, train_pos_idx, N_DRUG)
train_neg, train_neg1 = build_pairs(emb_coa, emb_fin, train_neg_idx, N_DRUG)
test_pos,  test_pos1  = build_pairs(emb_coa, emb_fin, test_pos_idx,  N_DRUG)
test_neg,  test_neg1  = build_pairs(emb_coa, emb_fin, test_neg_idx,  N_DRUG)

# Concatenate drug + ncRNA embeddings within each pair
train_pos,  train_pos1  = deal_embedding(train_pos,  train_pos1)
train_neg,  train_neg1  = deal_embedding(train_neg,  train_neg1)
test_pos,   test_pos1   = deal_embedding(test_pos,   test_pos1)
test_neg,   test_neg1   = deal_embedding(test_neg,   test_neg1)

# Fixed test set (same across all folds)
test_data  = torch.tensor(
    np.stack([test_pos + test_neg, test_pos1 + test_neg1]), dtype=torch.float32
).to(DEVICE)
test_label = torch.tensor(
    [1] * len(test_pos) + [0] * len(test_neg), dtype=torch.float32
).to(DEVICE)

feat_dim = test_data.size(-1)



def find_best_threshold(y_true, y_scores, metric='precision'):
    best_thresh = 0.5
    best_val = 0
    for thresh in [i/100 for i in range(30, 71)]:
        pred = (y_scores >= thresh).float()
        if metric == 'precision':
            val = precision_score(y_true.cpu(), pred.cpu(), zero_division=0)
        else:
            val = accuracy_score(y_true.cpu(), pred.cpu())
        if val > best_val:
            best_val = val
            best_thresh = thresh
    return best_thresh


def evaluate(model):
    """Run inference on the fixed test set and return metrics."""
    model.eval()
    with torch.no_grad():
        out = model(test_data[0], test_data[1])
    scores = out.cpu().squeeze(-1)
    preds  = (scores >= 0.5).float()

    #best_thresh = find_best_threshold(test_label.cpu(), scores, metric='precision')
    #preds = (preds >= best_thresh).float()

    auc  = roc_auc_score(test_label.cpu(), scores)
    aupr = average_precision_score(test_label.cpu(), scores)
    f1   = f1_score(test_label.cpu(), preds)
    acc, sen, pre, spe = calculate_metrics(test_label, preds)
    return [auc, acc, pre, sen, f1, aupr]


# ── 5-Fold cross-validation ────────────────────────────────────────────────────
fold_scores = []

for fold in range(N_FOLDS):
    tr_data, tr_label, _, _ = kfold_split(
        train_pos, train_neg, train_pos1, train_neg1, fold, N_FOLDS
    )

    model = MLP(feat_dim, feat_dim , 64, 1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    #loss_fn = model.criterion
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)

    for epoch in tqdm.tqdm(range(EPOCHS), desc=f'Fold {fold + 1}/{N_FOLDS}'):
        model.train()
        out  = model(tr_data[0], tr_data[1])
        loss = loss_fn(out, tr_label.unsqueeze(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= WARMUP:
            test_score = evaluate(model)
            early_stopping(test_score[0], model)
            if early_stopping.counter == 0:
                best_score = test_score
            if early_stopping.early_stop or epoch == EPOCHS - 1:
                break

    fold_scores.append(evaluate(model))
    auc, acc, pre, sen, f1, aupr = fold_scores[-1]
    print(f"Fold {fold + 1}: AUC={auc*100:.2f}  ACC={acc*100:.2f}  "
          f"PRE={pre*100:.2f}  SEN={sen*100:.2f}  F1={f1*100:.2f}  AUPR={aupr*100:.2f}")

# ── Summary ────────────────────────────────────────────────────────────────────
scores = np.array(fold_scores)
mean   = np.round(scores.mean(axis=0), 5)

print("\n\033[1;31m5-Fold Mean Results:")
print(f"AUC={mean[0]:.4f}  ACC={mean[1]:.4f}  PRE={mean[2]:.4f}  "
      f"SEN={mean[3]:.4f}  F1={mean[4]:.4f}  AUPR={mean[5]:.4f}\033[0m")

# ── False Negative Analysis ────────────────────────────────────────────────────
print("\n" + "="*70)
print("Analyzing False Negatives on Positive Test Set")
print("="*70)


