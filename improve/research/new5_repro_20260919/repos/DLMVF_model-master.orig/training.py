
import torch.nn as nn
from numpy import interp

from dataprocess import kfoldprepare
from model import DLMVF
from utils import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc
from torch_geometric.data import DataLoader
from sklearn.model_selection import KFold
import logging



def train(model, device, train_loader, data_o, optimizer, epoch):

    print('epoch:', epoch)
    model.train()
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        data_o = data_o.to(device)
        output= model(data, data_o)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device)).requires_grad_(True)
        total_loss = loss
        total_loss.backward()
        optimizer.step()

        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * train_loader.batch_size,
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))

def predicting(model, device, loader, data_o):
    model.eval()
    total_probs = []
    total_preds = []
    total_labels = []
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output= model(data, data_o)
            probs= output.cpu().numpy()
            preds = (output >= 0.5).float().cpu().numpy()

            total_probs.extend(probs)
            total_preds.extend(preds)
            total_labels.extend(data.y.view(-1, 1).cpu().numpy())

    total_probs = np.array(total_probs).flatten()
    total_preds = np.array(total_preds).flatten()
    total_labels = np.array(total_labels).flatten()

    accuracy = accuracy_score(total_labels, total_preds)
    precision = precision_score(total_labels, total_preds)
    recall = recall_score(total_labels, total_preds)
    f1 = f1_score(total_labels, total_preds)

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    roc_auc = roc_auc_score(total_labels, total_probs)

    precision_vals, recall_vals, _ = precision_recall_curve(total_labels, total_probs)

    sorted_indices = np.argsort(recall_vals)
    recall_vals = recall_vals[sorted_indices]
    precision_vals = precision_vals[sorted_indices]

    pr_auc = auc(recall_vals, precision_vals)
    fpr, tpr, _ = roc_curve(total_labels, total_probs)
    # Print evaluation metrics
    print(f"ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")

    return accuracy, precision, recall, f1, roc_auc, pr_auc, fpr, tpr, recall_vals, precision_vals




if __name__ == '__main__':
    modeling = DLMVF

    cuda_name = "cuda:7"
    device = torch.device(cuda_name)

    TRAIN_BATCH_SIZE = 64
    TEST_BATCH_SIZE = 64
    LR = 0.0005
    LOG_INTERVAL = 40
    NUM_EPOCHS = 15
    NUM_RUNS = 1
    print('Learning rate: ', LR)
    print('Epochs: ', NUM_EPOCHS)

    model = modeling().to(device)
    print(f"Model is on device: {next(model.parameters()).device}")

    # Prepare dataset and KFold cross-validation setup
    train_data, test_data, data_o = kfoldprepare()
    # Metrics for cross-validation
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)
    mean_precision = np.zeros_like(mean_recall)

    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    roc_aucs = []
    pr_aucs = []

    # Main loop for 5-fold cross-validation
    for fold in range(NUM_RUNS):
        print(f"\nFold {fold + 1}/5")

        # Split data into training and testing for the current fold

        train_loader = DataLoader(train_data[fold], batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_data[fold], batch_size=TEST_BATCH_SIZE, shuffle=False, drop_last=True)

        # Initialize model, loss function, and optimizer
        model = modeling().to(device)

        loss_fn = nn.BCELoss()  # for classification
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        # Train model for each epoch
        for epoch in range(NUM_EPOCHS):
            train(model, device, train_loader, data_o, optimizer, epoch + 1)
            predicting(model, device, test_loader, data_o)

        # Predict on the test set for the current fold
        accuracy, precision, recall, f1, roc_auc, pr_auc, fpr, tpr, recall_vals, precision_vals = predicting(model,
                                                                                                             device,
                                                                                                             test_loader,
                                                                                                             data_o)

        # Aggregate metrics for average calculations later
        sorted_indices = np.argsort(recall_vals)
        recall_vals_sorted = recall_vals[sorted_indices]
        precision_vals_sorted = precision_vals[sorted_indices]

        interpolated_precision = interp(mean_recall, recall_vals_sorted, precision_vals_sorted)
        mean_precision += interpolated_precision  # Sum the interpolated precision values for averaging later
        logging.info(f"Fold {fold + 1} Metrics:")
        logging.info(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
        logging.info(f"ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        roc_aucs.append(roc_auc)
        pr_aucs.append(pr_auc)
        sorted_indices_roc = np.argsort(fpr)
        fpr_sorted = fpr[sorted_indices_roc]
        tpr_sorted = tpr[sorted_indices_roc]
        mean_tpr += interp(mean_fpr, fpr_sorted, tpr_sorted)  # one dimensional interpolation

    mean_tpr[0] = 0.0

    # Calculate average metrics after 5-fold cross-validation
    avg_accuracy = np.mean(accuracies)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1_scores)
    avg_roc_auc = np.mean(roc_aucs)
    avg_pr_auc = np.mean(pr_aucs)
    mean_precision /= NUM_RUNS
    mean_tpr /= NUM_RUNS
    mean_tpr[-1] = 1.0

    mean_fpr_tpr = np.vstack((mean_fpr, mean_tpr)).T
    np.savetxt('DLMVF_mean_fpr_tpr.csv', mean_fpr_tpr, delimiter=',', header='mean_fpr,mean_tpr', comments='')
    mean_recall_precision = np.vstack((mean_recall, mean_precision)).T
    np.savetxt('DLMVF_mean_recall_precision.csv', mean_recall_precision, delimiter=',',
               header='mean_recall,mean_precision',
               comments='')

    print("\nAverage Metrics after 5-Fold Cross-Validation:")
    print(f"Accuracy: {avg_accuracy:.4f}")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")
    print(f"F1 Score: {avg_f1:.4f}")
    print(f"ROC AUC: {avg_roc_auc:.4f}")
    print(f"PR AUC: {avg_pr_auc:.4f}")

    from sklearn.metrics import auc
    csv_roc_auc = auc(mean_fpr, mean_tpr)
    csv_pr_auc = auc(mean_recall, mean_precision)

    print(f"\n=== AUC Verification ===")
    print(f"Predicted ROC AUC: {avg_roc_auc:.4f}")
    print(f"CSV ROC AUC: {csv_roc_auc:.4f}")
    print(f"Predicted PR AUC: {avg_pr_auc:.4f}")
    print(f"CSV PR AUC: {csv_pr_auc:.4f}")
    print(f"Difference ROC: {abs(avg_roc_auc - csv_roc_auc):.4f}")
    print(f"Difference PR: {abs(avg_pr_auc - csv_pr_auc):.4f}")

