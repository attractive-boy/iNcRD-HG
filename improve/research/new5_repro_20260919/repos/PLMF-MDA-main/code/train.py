import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, roc_curve, auc
from parms_setting import settings
from data_preprocess import load_data, collate_fn
from layers import DrugMiRNAInteractionModel
from torch.utils.data import DataLoader
import numpy as np


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Loading data...")
    train_loader, test_loader = load_data(args)

    train_loader = DataLoader(train_loader.dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate_fn, drop_last=True)
    test_loader = DataLoader(test_loader.dataset, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate_fn, drop_last=False)

    model = DrugMiRNAInteractionModel(args).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        train_labels = []
        train_probs = []

        for batch in train_loader:

            drug_graphs = batch['drug_graphs']
            drug_smiles = batch['drug_smiles']
            # print("mirna_encoded: ", batch['mirna_encoded'])
            mirna_encoded = batch['mirna_encoded'].to(device)
            mirna_seqs = batch['mirna_seqs']
            labels = batch['labels'].to(device)

            outputs = model(drug_graphs, drug_smiles, mirna_encoded, mirna_seqs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            train_labels.extend(labels.cpu().numpy())
            train_probs.extend(outputs.cpu().detach().cpu().numpy())

        avg_train_loss = running_loss / len(train_loader.dataset)
        train_labels = np.array(train_labels)
        train_probs = np.array(train_probs)
        train_preds = (train_probs > 0.5).astype(int)
        train_acc = accuracy_score(train_labels, train_preds)
        fpr, tpr, _ = roc_curve(train_labels, train_probs)
        train_auc = auc(fpr, tpr)
        precision, recall, _ = precision_recall_curve(train_labels, train_probs)
        train_aupr = auc(recall, precision)
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} "
              f"Acc: {train_acc:.4f} AUC: {train_auc:.4f} AUPR: {train_aupr:.4f}")

    test_loss, test_acc, test_auc, test_aupr = evaluate(model, test_loader, criterion, device)
    print(f"Final Test Result | Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} AUC: {test_auc:.4f} AUPR: {test_aupr:.4f}")

    print("Training completed.")


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            drug_graphs = batch['drug_graphs']
            drug_smiles = batch['drug_smiles']
            mirna_encoded = batch['mirna_encoded'].to(device)
            mirna_seqs = batch['mirna_seqs']
            labels = batch['labels'].to(device)

            outputs = model(drug_graphs, drug_smiles, mirna_encoded, mirna_seqs)
            loss_main = criterion(outputs, labels)
            total_loss += loss_main.item()

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = (np.array(all_probs) > 0.5).astype(int)

    acc = accuracy_score(all_labels, all_preds)
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auc_score = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    aupr_score = auc(recall, precision)

    avg_loss = total_loss / len(loader)

    return avg_loss, acc, auc_score, aupr_score

if __name__ == '__main__':
    args = settings()
    train(args)