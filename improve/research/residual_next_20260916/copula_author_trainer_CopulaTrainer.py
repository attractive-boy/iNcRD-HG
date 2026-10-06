import copy
from typing import Dict

import torch
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix

# My Own modules
import utils
from tracker import runtime_tracker


# ----------------------------------------------------------------------------------------------- #


class CopulaTrainer:
    def __init__(self, args, model, optimizer):
        self.args = args
        self.model = model
        self.optimizer = optimizer
        
    @runtime_tracker
    @staticmethod
    def train(model, optimizer, G: Data):
        model.train()
        optimizer.zero_grad()
        output = model(G)
        loss = model.loss_fn(*output, G)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    @runtime_tracker
    @staticmethod
    def evaluate(model, G: Data, action: str):
        model.eval()
        with torch.no_grad():
            metrics = sign_predict_with_conditional_sampling(model, G, action)
        return metrics
    
    def __call__(self, G, stats_trn_file, stats_val_file) -> dict:

        # Trend (epoch-loss, epoch-metric plot) containers
        trn_trend = []
        val_trend = []

        # Resource tracker
        trn_stats_list = []
        val_stats_list = []

        # Time tracker
        trn_time_list = []
        val_time_list = []

        # GPU memory tracker
        trn_mem_list = []
        val_mem_list = []

        # Early stopping objects
        curr_patience = 0
        max_patience = self.args.train.patience
        best_epoch = 0
        best_epoch_state = {'loss': float('inf'), 'auc': 0.0, 'macro_f1': 0.0}
        
        criterions: list[str] = self.args.train.criterions
        best_model_state = self.model.state_dict()

        # BEGIN TRAINING
        for epoch in range(self.args.train.epochs):
            loss, trn_stats = self.train(self.model, self.optimizer, G)
            trn_trend.append((epoch, loss))
            trn_stats_list.append(trn_stats)
            trn_time_list.append(trn_stats.time)
            trn_mem_list.append(trn_stats.nvidia_smi_used_cuda)

            print(f"Epoch {epoch:4d} | {utils.formalize(trn_stats, format='int')}", file=stats_trn_file, flush=True)

            # Validation
            if epoch % self.args.train.check_period == 0:
                val_metrics, val_stats = self.evaluate(self.model, G, action='validation')
                epoch_state: dict = {'loss': loss} | val_metrics

                val_trend.append((epoch, val_metrics))
                val_stats_list.append(val_stats)
                val_time_list.append(val_stats.time)
                val_mem_list.append(val_stats.nvidia_smi_used_cuda)
                print(f"Epoch {epoch:4d} | {utils.formalize(val_stats, format='int')}", file=stats_val_file, flush=True)
    
                message = utils.formalize(epoch_state)

                if utils.improvement_check(criterions, epoch_state, best_epoch_state):
                    message += " ----> New Best"
                    curr_patience = 0
                    best_epoch = epoch
                    best_epoch_state = epoch_state.copy()
                    # WARNING: YOU MUST DEEP COPY THE MODEL STATE DICT
                    best_model_state = copy.deepcopy(self.model.state_dict())  
                else:
                    curr_patience += 1
                
                print(f"Epoch {epoch:4d} | {message}", flush=True)

                if curr_patience >= max_patience:
                    break
        # END TRAINING

        self.model.load_state_dict(best_model_state)
        tst_metrics, _ = self.evaluate(self.model, G, action='test')

        results = {
            'trn_trend': trn_trend,
            'val_trend': val_trend,
            'trn_stats_list': trn_stats_list,
            'val_stats_list': val_stats_list,
            'best_epoch': best_epoch,
            'best_epoch_state': best_epoch_state,
            'best_model_state': best_model_state,
            'tst_metrics': tst_metrics,
            'trn_time_list': trn_time_list,
            'val_time_list': val_time_list,
            'trn_mem_list': trn_mem_list,
            'val_mem_list': val_mem_list
        }

        return results
    

# ----------------------------------------------------------------------------------------------- #


def sign_predict_with_conditional_sampling(model, G: Data, action: str) -> Dict[str, float]:
    
    trn_mask = G.uni_trn_mask
    
    if action == 'validation':
        eval_mask = G.uni_val_mask
    elif action == 'test':
        eval_mask = G.uni_tst_mask
    else:
        raise ValueError(f"Unknown action: {action}. Use 'validation' or 'test'.")

    eval_y = G.uni_edge_weight[eval_mask].clamp(min=0, max=1)
    pred_score = model.predict(G, uni_trn_mask=trn_mask, uni_eval_mask=eval_mask)
    pred_label = (pred_score >= 0.5)

    eval_y = eval_y.cpu()
    pred_score = pred_score.cpu()
    pred_label = pred_label.cpu()            

    auc = roc_auc_score(eval_y, pred_score)
    macro_f1 = f1_score(eval_y, pred_label, average='macro')

    if action == 'test':
        tn, fp, fn, tp = confusion_matrix(eval_y, pred_label).ravel().tolist()
        utils.print_confusion_matrix(tn, fp, fn, tp)
        utils.print_detail_marco_f1(tn, fp, fn, tp)

    metrics = { 'auc': auc, 'macro_f1': macro_f1 }
    return metrics
