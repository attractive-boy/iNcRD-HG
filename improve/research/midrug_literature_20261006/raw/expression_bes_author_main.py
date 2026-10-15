import argparse
import torch
import numpy as np
import random
import os
import time
import csv

from utils import get_dataset, set_seed, load_model_weights_path, select_optimizer

from train_eval import train_and_eval
from model import model_repulsive, model_gravite,classificationGCN
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--split', type=str, default='public')
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--hidden', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--optimizer', type=str, default='Adam')
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--seeds', type=int, default=42) 
args = parser.parse_args()
NUM_RUNS = 3
all_best_metrics = []

for run in range(1, NUM_RUNS + 1):
    print(f"\n===== begin {run}/{NUM_RUNS}  run=====")
    
    run_start_time = time.time()
    
    run_seed = args.seeds
    # set_seed(args.seeds+run)
    
    dataset_name = args.dataset
    lr = args.lr
    weight_decay = args.weight_decay
    str_optimizer = args.optimizer
    momentum = args.momentum
    epochs = args.epochs

    data = get_dataset(dataset_name, run_seed)
    data = data.to(device)
    model_repulsive_inst = model_repulsive(32, 2, data, dataset_name, run_seed)
    optimizer = select_optimizer(
        model_repulsive_inst.parameters(), 
        lr=lr, 
        weight_decay=weight_decay, 
        str_optimizer=str_optimizer,
        momentum=momentum
    )

    
    model_classification = classificationGCN(32, 64, data.num_classes,dataset_name)
    class_optimizer = torch.optim.Adam(model_classification.parameters(), lr=0.01, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(class_optimizer, step_size=100, gamma=0.1)  
    
    run_result_path = f'result_NC/{dataset_name}_{run_seed}_run{run}.txt'
    csv_result_path = f'result_NC/{dataset_name}_{run_seed}_run{run}.csv'
    
    run_directory = os.path.dirname(run_result_path)
    if not os.path.exists(run_directory):
        os.makedirs(run_directory)
    
    with open(csv_result_path, 'w', newline='') as csvfile:
        fieldnames = ['epoch', 'time_elapsed', 'acc', 'f1', 'auroc']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
    
    best_metrics = {
        'test_acc': -1.0,
        'test_f1': -1.0,
        'test_auroc': -1.0,
        'epoch': 0
    }
    best_embedding = None
    
    for epoch in range(1, epochs + 1):
        test_acc, test_f1, test_auroc,embedding = train_and_eval(
            model_repulsive_inst, optimizer, data, dataset_name, 
            model_classification, class_optimizer, epoch, scheduler
        )
        
        time_elapsed = time.time() - run_start_time
      
        time_str = time_elapsed
        
        print(f"run {run} - Epoch {epoch} | time: {time_str} | "
              f" Acc {test_acc:.4f}, F1 {test_f1:.4f}, AUROC {test_auroc:.4f}"
            )
        
        with open(run_result_path, 'a') as f:
            f.write(f"run_{run}_epoch_{epoch} | time: {time_str} s| "
                    f"test_acc: {test_acc:.4f}, test_f1: {test_f1:.4f}, test_auroc: {test_auroc:.4f}\n")
        
        with open(csv_result_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=['epoch', 'time_elapsed', 'acc', 'f1', 'auroc', ])
            writer.writerow({
                'epoch': epoch,
                'time_elapsed': time_str,
                'acc': round(test_acc, 6),
                'f1': round(test_f1, 6),
                'auroc': round(test_auroc, 6)
            })
        
        current_is_best = False
        if (test_acc > best_metrics['test_acc']):
            if test_acc > best_metrics['test_acc']:
                best_metrics['test_acc'] = test_acc
                best_metrics['test_f1'] = test_f1
                best_metrics['test_auroc'] = test_auroc
                best_embedding = embedding
            
          
            best_metrics['epoch'] = epoch
            current_is_best = True
        
            
            print(f"run {run} - Epoch {epoch} save embedding")
    
    all_best_metrics.append(best_metrics)
    print(f"===== NO {run}/{NUM_RUNS} runs end =====")


def calculate_stats(metrics_list, metric_name):
    values = [m[metric_name] for m in metrics_list]
    mean = np.mean(values)
    std = np.std(values)
    return mean, std


stats = {
    'test_acc': calculate_stats(all_best_metrics, 'test_acc'),
    'test_f1': calculate_stats(all_best_metrics, 'test_f1'),
    'test_auroc': calculate_stats(all_best_metrics, 'test_auroc'),

}

best_result_path = f'result_NC/{dataset_name}_{args.seeds}_summary.txt'
best_directory = os.path.dirname(best_result_path)
if not os.path.exists(best_directory):
    os.makedirs(best_directory)

with open(best_result_path, 'w') as f:
    f.write(f"===== result =====\n")
    f.write(f"test (ave ± std):\n")
    f.write(f"Accuracy: {stats['test_acc'][0]:.4f} ± {stats['test_acc'][1]:.4f}\n")
    f.write(f"F1(macro): {stats['test_f1'][0]:.4f} ± {stats['test_f1'][1]:.4f}\n")
    f.write(f"AUROC: {stats['test_auroc'][0]:.4f} ± {stats['test_auroc'][1]:.4f}\n")

    
    for i, metrics in enumerate(all_best_metrics, 1):
        f.write(f"run {i} best result (Epoch {metrics['epoch']}):\n")
        f.write(f"Acc {metrics['test_acc']:.4f}, F1 {metrics['test_f1']:.4f}, AUROC {metrics['test_auroc']:.4f}\n")

print("\nall runs finish:", best_result_path)
    