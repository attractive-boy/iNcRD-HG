import sys
import torch.nn as nn
from numpy import interp
from cnn_gcnmulti import GCNNetmuti
from utils import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc

from torch_geometric.data import DataLoader



def train(model, device, train_loader, optimizer, epoch):
    print('Training on {} samples...'.format(len(train_loader.dataset)))
    model.train()
    for batch_idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        output = model(data)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device)).requires_grad_(True)
        loss.backward()
        optimizer.step()

        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * train_loader.batch_size,
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.item()))


def predicting(model, device, loader):
    model.eval()
    total_probs = []
    total_preds = []
    total_labels = []
    print('Make prediction for {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            probs = output.cpu().numpy()
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

    # Evaluation metrics
    roc_auc = roc_auc_score(total_labels, total_probs)
    precision_vals, recall_vals, _ = precision_recall_curve(total_labels, total_probs)
    sorted_indices = np.argsort(recall_vals)
    recall_vals = recall_vals[sorted_indices]
    precision_vals = precision_vals[sorted_indices]

    pr_auc = auc(recall_vals, precision_vals)
    print(f"ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")

    return accuracy, precision, recall, f1, roc_auc, pr_auc


def accuracy(true_labels, preds):
    return np.mean(true_labels == preds)


modeling = GCNNetmuti

model_st = modeling.__name__

cuda_name = "cuda:0"
if len(sys.argv) > 3:
    cuda_name = "cuda:" + str(int(sys.argv[1]))
print('cuda_name:', cuda_name)

TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 64
LR = 0.0005
LOG_INTERVAL = 45
NUM_EPOCHS = 30
NUM_RUNS = 5

print('Learning rate: ', LR)
print('Epochs: ', NUM_EPOCHS)


print('\nrunning on ', model_st + '_')
processed_data_file_train = 'data/processed/' + 'train1.pt'
processed_data_file_test = 'data/processed/' + 'test1.pt'
if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
    print('please run process_data.py to prepare data in pytorch format!')
else:
    #train
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    roc_aucs = []
    pr_aucs = []
    for run in range(NUM_RUNS):
        train_data = TestbedDataset(root='data', dataset='train'+str(run))
        test_data = TestbedDataset(root='data', dataset='test'+str(run))
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, drop_last=True)
        print(f"\nRun {run + 1}/{NUM_RUNS}")
        model = modeling().to(device)
        loss_fn = nn.BCELoss()  # for classification
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        for epoch in range(NUM_EPOCHS):
            train(model, device, train_loader, optimizer, epoch + 1)

        accuracy, precision, recall, f1, roc_auc, pr_auc= predicting(model, device, test_loader)
        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        roc_aucs.append(roc_auc)
        pr_aucs.append(pr_auc)

    # average
    avg_accuracy = np.mean(accuracies)
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
    avg_f1 = np.mean(f1_scores)
    avg_roc_auc = np.mean(roc_aucs)
    avg_pr_auc = np.mean(pr_aucs)

    print("\nAverage Metrics after 5 runs:")
    print(f"Accuracy: {avg_accuracy:.4f}")
    print(f"Precision: {avg_precision:.4f}")
    print(f"Recall: {avg_recall:.4f}")
    print(f"F1 Score: {avg_f1:.4f}")
    print(f"ROC AUC: {avg_roc_auc:.4f}")
    print(f"PR AUC: {avg_pr_auc:.4f}")
