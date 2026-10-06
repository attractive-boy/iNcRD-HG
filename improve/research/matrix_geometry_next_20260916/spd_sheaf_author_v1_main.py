
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python main.py --config ./config/bbbp.yaml
"""
# Standard library imports
import argparse
import os
import sys
import time

# Third-party imports
import numpy as np
import torch
torch.set_default_dtype(torch.float64)

import torch.nn as nn
import torch.nn.functional as F
import yaml
from logzero import logger
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to sys.path to allow for absolute imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from SPDSheaf.model.SPDsheaf import SPDGNNModel

from utils.helper import set_seed, setup_device, print_config, print_final_statistics, print_all_params
from utils.logging_utils import setup_logging
from utils.data import CustomDataset, create_data, collate_fn

# Muon optimizer
# from muon import SingleDeviceMuonWithAuxAdam
from pytorch_optimizer import Muon

##########################
## Prepare Data Loaders ##
##########################
def setup_dataset_config(args):
    """Setup dataset configuration and return relevant parameters"""
    datafile = args.select_dataset
    batch_size = args.batch_size
    train_ratio = args.train_ratio
    vali_ratio = args.vali_ratio
    test_ratio = args.test_ratio

    target_map = {'tox21': 12, 'muv': 17, 'sider': 27, 'clintox': 2,
                  'bace': 1, 'bbbp': 1, 'hiv': 1}
    target_dim = target_map.get(datafile, 1)

    encoder_atom = args.encoder_atom
    encoder_bond = args.encoder_bond
    encode_dim = args.encode_dim

    return datafile, batch_size, train_ratio, vali_ratio, test_ratio, target_dim, encoder_atom, encoder_bond, encode_dim

def prepare_data_loaders(datafile, batch_size, train_ratio, vali_ratio, test_ratio,
                         encoder_atom, encoder_bond, args=None):
    """Create/load data and prepare data loaders"""

    # Create data if not exists
    create_data(datafile, encoder_atom,
                encoder_bond, batch_size,
                train_ratio, vali_ratio,
                test_ratio)

    # Load datasets
    state = torch.load('data/processed/' + datafile + '.pth')

    loaded_train_dataset = CustomDataset(state['train_label'], state['train_graph_list'])
    loaded_valid_dataset = CustomDataset(state['valid_label'], state['valid_graph_list'])
    loaded_test_dataset = CustomDataset(state['test_label'], state['test_graph_list'])

    loaded_train_loader = DataLoader(
        loaded_train_dataset, batch_size=batch_size, shuffle=state['shuffle'],
        num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate_fn
    )

    if vali_ratio == 0.0:
        loaded_valid_loader = loaded_train_loader  # Use train as validation
    else:
        loaded_valid_loader = DataLoader(
            loaded_valid_dataset, batch_size=batch_size, shuffle=state['shuffle'],
            num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate_fn
        )

    loaded_test_loader = DataLoader(
        loaded_test_dataset, batch_size=batch_size, shuffle=state['shuffle'],
        num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate_fn
    )

    print('\nDataset loaded!')
    print(f"Training set: {len(loaded_train_dataset)}")
    print(f"Validation set: {len(loaded_valid_dataset)}")
    print(f"Testing set: {len(loaded_test_dataset)}\n")

    return loaded_train_loader, loaded_valid_loader, loaded_test_loader, loaded_train_dataset


#####################
## Setup Functions ##
#####################
def setup_loss_function(args):
    """Setup loss function based on configuration"""
    if args.loss_select == 'bce':
        loss_fn = nn.BCELoss(reduction='none')
    elif args.loss_select == 'l1':
        loss_fn = nn.L1Loss(reduction='sum')
    elif args.loss_select == 'l2':
        loss_fn = nn.MSELoss(reduction='none')
    elif args.loss_select == 'bcewl':
        loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    else:
        loss_fn = nn.BCELoss(reduction='none')
    return loss_fn


def setup_optimizer_and_scheduler(model, model_args, args):
    """Setup optimizer and scheduler based on configuration"""
    if model_args.Muon:
        # Group parameters for Muon and Adam
        muon_params, adam_params = [], []
        for name, p in model.named_parameters():
            if p.ndim >= 2:# and 'transforms' in name:
                # Apply Muon to the main hidden transformation layers
                muon_params.append(p)
            else:
                # Apply Adam to everything else (classifier, batchnorm biases, etc.)
                adam_params.append(p)

        logger.info(f"Optimizing {sum(p.numel() for p in muon_params):,} parameters with Muon.")
        logger.info(f"Optimizing {sum(p.numel() for p in adam_params):,} parameters with Adam.")

        # Define parameter groups for the optimizer
        param_groups = [
            dict(params=muon_params, use_muon=True,
                lr=args.Muon_LR, weight_decay=args.weighted_decay),  # Recommended Muon LR
            dict(params=adam_params, use_muon=False,
                lr=args.Adam_LR, betas=(0.9, 0.95), weight_decay=args.weighted_decay), # Adam params from your config
        ]

        # Optimizer and scheduler
        # The Muon paper suggests it may not need a scheduler.
        # It's best to disable it for the first run.
        optimizer = Muon(param_groups)
        scheduler = None

    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.Adam_LR)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    return optimizer, scheduler


###################
## Training Loop ##
###################
def train(model, device, train_loader, valid_loader, optimizer, loss_fn, epoch):
    """Training loop for one epoch"""
    model.train()
    total_loss_train = 0.0
    num_train_samples = 0

    # Show progress bar only in DEBUG mode
    show_progress = args.log_level == 'DEBUG'

    iterator = train_loader
    if tqdm is not None and show_progress:
        iterator = tqdm(train_loader, desc=f"Train Epoch {epoch}", leave=False, dynamic_ncols=True)

    for batch_idx, data in enumerate(iterator):
        optimizer.zero_grad()

        y = data[0].to(device, non_blocking=True)
        graph_list = data[1].to(device, non_blocking=True)
        graph_list = update_node_features(graph_list)
        node_features = graph_list.ndata['feat']

        # Forward pass
        output = model(graph_list, node_features)

        # Debug: check for NaNs/Infs in forward output
        if args.debug:
            assert torch.isfinite(output).all(), 'NaN/Inf in output (train)'

        # Prepare labels and predictions
        arr_label = torch.empty(0, device=device)
        arr_pred = torch.empty(0, device=device)
        for j in range(y.shape[1]):
            c_label, c_pred = y[:, j], output[:, j]
            zero = torch.zeros_like(c_label)
            c_label = torch.where(c_label == -1, zero, c_label)

            arr_label = torch.cat((arr_label, c_label), 0)
            arr_pred = torch.cat((arr_pred, c_pred), 0)

        arr_pred = arr_pred.float()
        arr_label = arr_label.float()

        loss = loss_fn(arr_pred, arr_label)
        train_loss = torch.sum(loss)
        total_loss_train += train_loss
        num_train_samples += arr_label.size(0)
        if tqdm is not None and show_progress:
            try:
                iterator.set_postfix({"batch_loss": float(train_loss.detach().item())})
            except Exception:
                pass

        train_loss.backward()


        # Optional gradient clipping (standard training branch)
        if bool(args.enable_grad_clip):
            clip_norm = float(args.grad_clip_norm)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

        # Debug: check gradient norm and explode (without modifying gradients again)
        if args.debug:
            threshold = float(args.grad_explode_threshold)
            with torch.no_grad():
                grads = [p.grad.detach() for p in model.parameters() if p.grad is not None]
                if len(grads) == 0:
                    total_norm = torch.tensor(0.0, device=device)
                else:
                    total_norm = torch.norm(torch.stack([g.norm(2) for g in grads]), 2)
            if not torch.isfinite(total_norm) or float(total_norm) >= threshold:
                raise RuntimeError(f'Gradient exploded or non-finite: {total_norm}')

        optimizer.step()

    # Validation
    model.eval()
    total_loss_val = 0.0
    num_val_samples = 0
    with torch.no_grad():
        val_iter = valid_loader
        if tqdm is not None and show_progress:
            val_iter = tqdm(valid_loader, desc=f"Valid Epoch {epoch}", leave=False, dynamic_ncols=True)

        for batch_idx, valid_data in enumerate(val_iter):
            y = valid_data[0].to(device, non_blocking=True)
            graph_list = valid_data[1].to(device, non_blocking=True)
            graph_list = update_node_features(graph_list)
            node_features = graph_list.ndata['feat']
            output = model(graph_list, node_features)
            if args.debug:
                assert torch.isfinite(output).all(), 'NaN/Inf in output (valid)'

            arr_label = torch.empty(0, device=device)
            arr_pred = torch.empty(0, device=device)
            for j in range(y.shape[1]):
                c_label, c_pred = y[:, j], output[:, j]
                zero = torch.zeros_like(c_label)
                c_label = torch.where(c_label == -1, zero, c_label)

                arr_label = torch.cat((arr_label, c_label), 0)
                arr_pred = torch.cat((arr_pred, c_pred), 0)

            arr_pred = arr_pred.float()
            arr_label = arr_label.float()

            loss = loss_fn(arr_pred, arr_label)
            valid_loss = torch.sum(loss)
            total_loss_val += valid_loss
            num_val_samples += arr_label.size(0)
            if tqdm is not None and show_progress:
                try:
                    val_iter.set_postfix({"valid_loss": float(valid_loss.detach().item())})
                except Exception:
                    pass

        # Return average losses
        avg_train_loss = total_loss_train / num_train_samples if num_train_samples > 0 else 0.0
        avg_val_loss = total_loss_val / num_val_samples if num_val_samples > 0 else 0.0
        return total_loss_train, total_loss_val, avg_train_loss, avg_val_loss


def message_func(edges):
    return {'feat': edges.data['feat']}


def reduce_func(nodes):
    num_edges = nodes.mailbox['feat'].size(1)
    agg_feats = torch.sum(nodes.mailbox['feat'], dim=1) / num_edges
    return {'agg_feats': agg_feats}


def update_node_features(g):
    """Update node features by aggregating neighbor features"""
    g.send_and_recv(g.edges(), message_func, reduce_func)
    g.ndata['feat'] = torch.cat((g.ndata['feat'], g.ndata['agg_feats']), dim=1)
    return g


#########################
## Prediction Function ##
#########################
def predicting(model, device, data_loader, epoch=0):
    """Prediction and evaluation"""
    model.eval()

    total_preds = torch.empty(0, device=device)
    total_labels = torch.empty(0, device=device)
    per_task_preds = None
    per_task_labels = None

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            y = data[0].to(device, non_blocking=True)
            graph_list = data[1].to(device, non_blocking=True)
            graph_list = update_node_features(graph_list)
            node_features = graph_list.ndata['feat']
            output = model(graph_list, node_features)
            if args.debug:
                assert torch.isfinite(output).all(), 'NaN/Inf in output (predict)'

            arr_label = torch.empty(0, device=device)
            arr_pred = torch.empty(0, device=device)
            num_tasks = y.shape[1]
            # Initialize per-task containers on first batch
            if per_task_preds is None:
                per_task_preds = [list() for _ in range(num_tasks)]
                per_task_labels = [list() for _ in range(num_tasks)]

            for j in range(num_tasks):
                c_label, c_pred = y[:, j], output[:, j]
                zero = torch.zeros_like(c_label)
                c_label = torch.where(c_label == -1, zero, c_label)

                arr_label = torch.cat((arr_label, c_label), 0)
                arr_pred = torch.cat((arr_pred, c_pred), 0)
                # Accumulate per-task arrays across batches
                per_task_labels[j].append(c_label)
                per_task_preds[j].append(c_pred)

            total_preds = torch.cat((total_preds, arr_pred), 0)
            total_labels = torch.cat((total_labels, arr_label), 0)

    AUC = roc_auc_score(total_labels.detach().cpu().numpy().flatten(), total_preds.detach().cpu().numpy().flatten())
    # Compute per-task AUCs
    per_task_auc = []
    if per_task_preds is not None and per_task_labels is not None:
        for j in range(len(per_task_preds)):
            yj = torch.cat(per_task_labels[j], 0).detach().cpu().numpy().flatten()
            pj = torch.cat(per_task_preds[j], 0).detach().cpu().numpy().flatten()
            try:
                auc_j = roc_auc_score(yj, pj)
            except Exception:
                auc_j = float('nan')
            per_task_auc.append(auc_j)
        # Print concise per-task AUC summary
        logger.info(f"Epoch {epoch} | Per-task AUCs: {[None if (isinstance(a, float) and (np.isnan(a))) else round(a, 5) for a in per_task_auc]}")
    return AUC



def train_single_iteration(i, iter, model_args, args, device, loaded_train_dataset,
                          loaded_train_loader, loaded_valid_loader, loaded_test_loader,
                          datafile, loss_fn):
    """Execute a single training iteration"""
    print(f"\n{'='*50}")
    print(f"Training iteration {i+1}/{iter}")
    print(f"{'='*50}\n")

    # Initialize SPD Sheaf model
    sample_graph = loaded_train_dataset[0][1]
    num_nodes = sample_graph.num_nodes()
    num_edges = sample_graph.num_edges()

    model = SPDGNNModel(model_args, num_nodes, num_edges)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}\n")

    # Setup optimizer and scheduler
    optimizer, scheduler = setup_optimizer_and_scheduler(model, model_args, args)

    best_auc = 0
    logger.info(f"Gradient Clipping Enabled: {args.enable_grad_clip}")

    # Track losses for plotting
    train_losses = []
    val_losses = []
    test_aucs = []

    # Training epochs
    for epoch in range(args.NUM_EPOCHS):
        train_loss, val_loss, avg_train_loss, avg_val_loss = train(
            model, device,
            loaded_train_loader, loaded_valid_loader,
            optimizer, loss_fn, epoch + 1
        )
        # Log training and validation loss
        logger.info(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.5f} | Valid Loss: {avg_val_loss:.5f}")

        # Store losses
        train_losses.append(avg_train_loss.detach().cpu().item() if isinstance(avg_train_loss, torch.Tensor) else avg_train_loss)
        val_losses.append(avg_val_loss.detach().cpu().item() if isinstance(avg_val_loss, torch.Tensor) else avg_val_loss)

        # Evaluate
        AUC = predicting(model, device, loaded_test_loader, epoch + 1)
        test_aucs.append(AUC)

        if AUC > best_auc:
            best_auc = AUC
            logger.info(f'[NEW BEST] AUC: {best_auc:.5f}')

            # Save best model
            save_dir = f'./checkpoint/{args.select_dataset}'
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), f'{save_dir}/best_model.pth')

        if ((epoch+1) % 10 == 0 and not model_args.Muon):
            logger.info(f"Epoch {epoch+1} | Best AUC: {best_auc:.5f} | LR: {scheduler.get_last_lr()}")
        elif (epoch+1) % 10 == 0 and model_args.Muon:
            logger.info(f"Epoch {epoch+1} | Best AUC: {best_auc:.5f}")

        if scheduler is not None:
            scheduler.step()

    # Plotting
    try:
        # Create a figure with subplots: Loss, AUC, and Text
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 2, width_ratios=[3, 1])

        ax_loss = fig.add_subplot(gs[0, 0])
        ax_auc = fig.add_subplot(gs[1, 0])
        ax_text = fig.add_subplot(gs[:, 1])

        # Plot losses
        ax_loss.plot(train_losses, label='Train Loss')
        ax_loss.plot(val_losses, label='Validation Loss')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title(f'Training and Validation Loss - {datafile}')
        ax_loss.legend()
        ax_loss.grid(True)

        # Plot AUC
        ax_auc.plot(test_aucs, label='Test AUC', color='green')
        ax_auc.set_xlabel('Epoch')
        ax_auc.set_ylabel('AUC')
        ax_auc.set_title(f'Test ROC-AUC - {datafile}')
        ax_auc.legend()
        ax_auc.grid(True)

        # Add config parameters to the right side
        ax_text.axis('off')
        config_str = "Configuration:\n\n"
        # Filter out some internal/large objects if needed, or just print all simple types
        for key, value in vars(args).items():
            if isinstance(value, (int, float, str, bool, list)):
                # Truncate long lists or strings if necessary
                val_str = str(value)
                if len(val_str) > 50:
                    val_str = val_str[:47] + "..."
                config_str += f"{key}: {val_str}\n"

        ax_text.text(0, 1, config_str, transform=ax_text.transAxes,
                     fontsize=8, verticalalignment='top', fontfamily='monospace')

        # Create directory if it doesn't exist
        plot_dir = f'plots/{datafile}'
        os.makedirs(plot_dir, exist_ok=True)

        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        plot_path = f'{plot_dir}/loss_auc_plot_iter{i+1}_{timestamp}.png'
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Loss and AUC plot saved to {plot_path}")
    except Exception as e:
        logger.error(f"Failed to save plot: {e}")

    logger.info(f"\nIteration {i+1} completed. Best AUC: {best_auc:.5f}")
    return best_auc


class ModelArgs:
    def __init__(self, config_args):
        # For ABIDE dataset, use num_node_features from config file
        # For molecular datasets, use encode_dim[0] + encode_dim[1]
        if hasattr(config_args, 'num_node_features') and config_args.num_node_features is not None:
            self.num_node_features = config_args.num_node_features
        else:
            self.num_node_features = encode_dim[0] + encode_dim[1]
        self.hidden_dims = config_args.hidden_dims
        self.feat_dim = config_args.feat_dim
        self.sheaf = config_args.sheaf
        self.num_layers = config_args.num_layers
        self.dropout = config_args.dropout
        self.vec2sym = config_args.vec2sym
        self.transform = config_args.spd_transform
        self.nonlinear = config_args.spd_nonlinear
        self.orthogonal_map = config_args.orthogonal_map
        self.classifier = config_args.classifier
        # Inline MLP head after pooling
        self.use_mlp_head = config_args.use_mlp_head
        self.mlp_head_type = config_args.mlp_head_type
        # MLP Classifier head parameters
        self.mlp_hidden_dim = config_args.mlp_hidden_dim
        self.mlp_layers = config_args.mlp_layers
        self.mlp_dropout = config_args.mlp_dropout
        # Factorized Bilinear Pooling (dual-SPD MLP head)
        self.fbp_rank = config_args.fbp_rank
        self.interaction_type = config_args.interaction_type # Interaction type selection
        self.bidirectional_interaction = False  # Bidirectional interaction, default to False
        # Cross-attention parameters
        self.ca_d_model = config_args.ca_d_model
        self.ca_num_heads = config_args.ca_num_heads
        # Gradient clipping
        self.enable_grad_clip = config_args.enable_grad_clip
        self.grad_clip_norm = config_args.grad_clip_norm
        self.coord_method = config_args.coord_method
        self.epsilon = config_args.epsilon
        self.tau = config_args.tau
        self.output_dim = target_dim
        self.num_classes = target_dim
        self.coor_pooling = config_args.coor_pooling
        self.coor_pooling_power = config_args.coor_pooling_power
        self.feat_pooling = config_args.feat_pooling
        self.Muon = config_args.Muon
        self.Muon_LR = config_args.Muon_LR
        self.weighted_decay = config_args.weighted_decay
        self.n_centroids = target_dim
        self.sheaf_act = config_args.sheaf_act

        self.sheaf_learner_type = config_args.sheaf_learner_type


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="SPD Sheaf GNN Training")
    parser.add_argument("--config", type=str, default='./config/bbbp.yaml',
                        help="Path to config file")
    # Optional explicit log-level override (e.g., --log-level DEBUG)
    parser.add_argument("--log-level", dest="log_level", type=str, default=None,
                        help="Logging level override: DEBUG or INFO")

    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as config_file:
            config = yaml.safe_load(config_file)
        for key, value in config.items():
            # Do not override explicit CLI args (e.g., --log-level)
            if getattr(args, key, None) is None:
                setattr(args, key, value)
            else:
                # If CLI provides a value, keep it; otherwise assign
                # This preserves CLI > YAML precedence.
                pass

    return args

if __name__ == '__main__':
    # ===== Print Process ID =====
    logger.info(f"Process ID: {os.getpid()}")

    # ===== Initialization Phase =====
    args = parse_arguments()

    # Configure logging early (supports logzero + loguru)
    setup_logging(level=getattr(args, 'log_level', None),
                  debug=getattr(args, 'debug', None),
                  log_file=getattr(args, 'log_file', None))

    device = setup_device()
    print_config(args)
    set_seed(args.seed)

    # ===== Configuration Phase =====
    datafile, batch_size, train_ratio, vali_ratio, test_ratio, target_dim, encoder_atom, encoder_bond, encode_dim = setup_dataset_config(args)

    # ===== Data Preparation Phase =====
    loaded_train_loader, loaded_valid_loader, loaded_test_loader, loaded_train_dataset = prepare_data_loaders(
        datafile, batch_size, train_ratio, vali_ratio, test_ratio, encoder_atom, encoder_bond, args
    )

    # ===== Training Setup Phase =====
    loss_fn = setup_loss_function(args)
    model_args = ModelArgs(args)

    All_AUC = []
    start_time = time.time()

    # ===== Training Execution Phase =====
    for i in range(args.iter):
        best_auc = train_single_iteration(
            i, args.iter, model_args, args, device,
            loaded_train_dataset, loaded_train_loader,
            loaded_valid_loader, loaded_test_loader,
            datafile, loss_fn
        )
        All_AUC.append(best_auc)

    # ===== Results Phase =====
    end_time = time.time()
    elapsed_time = end_time - start_time
    print_final_statistics(All_AUC, elapsed_time)
    print_all_params(args, model_args)
