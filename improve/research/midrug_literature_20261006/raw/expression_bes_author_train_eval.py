import os
import time
import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
import torch.nn.functional as F  

def train_and_eval(model, optimizer, data, dataset_name, model_classification, 
                   class_optimizer, epoch, scheduler, tau=1.0, delta=5.0, alpha=1.0):

    device = data.x.device if hasattr(data.x, 'device') else torch.device("cpu")
    labels = data.y.to(device)
    train_mask = data.train_mask.to(device)
    num_classes = torch.unique(labels).size(0)
    
    model.to(device)
    model_classification.to(device)
    

    model.eval()
    with torch.no_grad():
        Phi_X = model(data)
        
    train_Phi = Phi_X[train_mask]
    train_labels = labels[train_mask]
    train_indices = train_mask.nonzero().squeeze()

    class_centroids = []
    for c in range(num_classes):
        c_mask = (train_labels == c)
        if c_mask.sum() > 0:
            class_centroids.append(train_Phi[c_mask].mean(dim=0))
        else:
            class_centroids.append(torch.zeros(Phi_X.size(1), device=device))
    class_centroids = torch.stack(class_centroids) # [num_classes, D]

    centered_Phi = train_Phi - train_Phi.mean(dim=0, keepdim=True)
    Sigma = (centered_Phi.t() @ centered_Phi) / (train_Phi.size(0) - 1 + 1e-8)
    Sigma = Sigma + 1e-4 * torch.eye(Sigma.size(0), device=device)
    Sigma_inv = torch.inverse(Sigma)

    B_indices_list = []
    for i in range(train_indices.size(0)):
        idx = train_indices[i].item()
        m = train_labels[i].item() 
        Phi_xi = train_Phi[i]
        
        in_boundary = False
        for n in range(num_classes):
            if n == m: continue
            mu_diff = class_centroids[m] - class_centroids[n]
            slab_val = torch.abs(torch.dot(mu_diff @ Sigma_inv, Phi_xi))
            if slab_val <= delta:
                in_boundary = True
                break
        if in_boundary:
            B_indices_list.append(idx)
            
    if len(B_indices_list) > 0:
        B_indices = torch.tensor(B_indices_list, dtype=torch.long, device=device)
    else:
        B_indices = torch.tensor([], dtype=torch.long, device=device)

    if B_indices.numel() > 1:
        B_Phi = Phi_X[B_indices]
        B_labels = labels[B_indices]
        dist_B = torch.cdist(B_Phi, B_Phi)
        k_nn = min(5, B_indices.numel() - 1)
        _, knn_idx = torch.topk(dist_B, k=k_nn + 1, dim=1, largest=False)
        knn_idx = knn_idx[:, 1:] 
        
        knn_labels = B_labels[knn_idx]
        mismatch = (knn_labels != B_labels.unsqueeze(1)).float()
        S_v = mismatch.mean(dim=1)
        
        boundary_nodes = B_indices[S_v > 0.5]
    else:
        boundary_nodes = torch.tensor([], dtype=torch.long, device=device)

    if boundary_nodes.numel() > 0:
        model.train()
        optimizer.zero_grad()
        
        Z_hat = model(data)
        
        beta_size = min(256, boundary_nodes.size(0))
        batch_idx = boundary_nodes[torch.randperm(boundary_nodes.size(0))[:beta_size]]
        
        Z_batch = Z_hat[batch_idx]
        y_batch = labels[batch_idx]
        detached_centroids = class_centroids.detach()
        
        dist_to_pos = torch.norm(Z_batch - detached_centroids[y_batch], dim=1)
        dist_to_all_neg = torch.cdist(Z_batch, detached_centroids)
        
        neg_mask = torch.ones((Z_batch.size(0), num_classes), dtype=torch.bool, device=device)
        neg_mask[torch.arange(Z_batch.size(0)), y_batch] = False
        
        dist_to_all_neg_masked = dist_to_all_neg.clone()
        dist_to_all_neg_masked[~neg_mask] = float('inf')
        min_dist_to_neg, _ = torch.min(dist_to_all_neg_masked, dim=1)
        
        sim_pos = - torch.clamp(dist_to_pos - min_dist_to_neg, min=0.0) ** 2
        sim_neg = - dist_to_all_neg ** 2 
        
        numerator = torch.exp(sim_pos / tau)
        
        exp_sim_neg = torch.exp(sim_neg / tau) * neg_mask.float()
        
        denominator = numerator + exp_sim_neg.sum(dim=1)
        
        gravity_loss = - torch.log(numerator / (denominator + 1e-8)).mean()
        
        gravity_loss.backward()

        eta = optimizer.param_groups[0]['lr']
        original_params = []
        
        for p in model.parameters():
            original_params.append(p.clone().detach())
            if p.grad is not None:
                p.data.add_(p.grad, alpha=-eta)
                
        model.eval()
        with torch.no_grad():
            Z_virtual = model(data)
            
        delta_B = torch.norm(Z_virtual - Z_hat.detach(), dim=1).sum().item()
        
        model.train()
        scale_factor = alpha / (delta_B + 1e-8)
        
        for p, orig_p in zip(model.parameters(), original_params):
            p.data.copy_(orig_p) 
            if p.grad is not None:
                p.grad.mul_(scale_factor)
                
        optimizer.step()


    model.eval() 
    with torch.no_grad():
        h_for_clf = model(data)
    
    model_classification.train()
    class_optimizer.zero_grad()
    log_probs, _ = model_classification(h_for_clf, data.edge_index)
    
    cls_loss = torch.nn.CrossEntropyLoss()(log_probs[train_mask], labels[train_mask])
    cls_loss.backward()
    class_optimizer.step()

    model_classification.eval()
    with torch.no_grad():
        out_eval, embedding_to_return = model_classification(h_for_clf, data.edge_index)
        test_mask = data.test_mask
        y_true = labels[test_mask].cpu().numpy()
        y_prob = torch.softmax(out_eval[test_mask], dim=1).cpu().numpy()
        y_pred = out_eval[test_mask].argmax(dim=1).cpu().numpy()

        acc = (y_pred == y_true).mean()
        f1 = f1_score(y_true, y_pred, average='macro')
        auroc = roc_auc_score(y_true, y_prob, multi_class='ovr')

    if epoch < 200 and scheduler is not None: 
        scheduler.step()
    return acc, f1, auroc, embedding_to_return