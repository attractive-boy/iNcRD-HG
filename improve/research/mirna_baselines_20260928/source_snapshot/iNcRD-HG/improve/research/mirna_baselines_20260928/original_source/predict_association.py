# %%
import torch
from torch_geometric.nn import SAGEConv, to_hetero,GeneralConv
from torch.nn import Linear
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score,average_precision_score
import numpy as np
import random
import pandas as pd
from torch_geometric.explain import Explainer
from torch_geometric.explain.algorithm import CaptumExplainer
import captum.attr as attr



SEED = 666
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

# %%
class Model(torch.nn.Module):
    def __init__(self, data,hidden_channels,aggr_str):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr=aggr_str)
        self.decoder = EdgeDecoder(hidden_channels)

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        z_dict = self.encoder(x_dict, edge_index_dict)
        pred,latent_feature = self.decoder(z_dict, edge_label_index)
        return pred,latent_feature

# %%
class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GeneralConv((-1, -1), hidden_channels)
        self.conv2 = GeneralConv((-1, -1), out_channels)


    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.conv2(x, edge_index)
        return x

# %%
class EdgeDecoder(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = Linear(2 * hidden_channels, 2 * hidden_channels)
        self.lin2 = Linear(2 * hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_label_index):
        row, col = edge_label_index
        z = torch.cat([z_dict[RNA_type][row], z_dict['drug'][col]], dim=-1)
        z = self.lin1(z)
        z = F.leaky_relu(z, negative_slope=0.1)
        z = self.lin2(z)
        latent_feature = z.detach().cpu().numpy()
        z = F.leaky_relu(z, negative_slope=0.1)
        z = self.lin3(z)
        z = z.view(-1)

        return z,latent_feature

def train(train_data):
    model.train()
    optimizer.zero_grad()
    pred,latent_feature = model(train_data.x_dict, train_data.edge_index_dict,
                 train_data[RNA_type, edge_type, 'drug'].edge_label_index)
    target = train_data[RNA_type,edge_type, 'drug'].edge_label
    if torch.isnan(pred).any():
        print("NaN found in pred during training")
    loss = F.binary_cross_entropy_with_logits(pred, target)
    loss.backward()
    optimizer.step()
    return float(loss)

# %%
@torch.no_grad()
def test(data):
    model.eval()
    pred,latent_feature = model(data.x_dict, data.edge_index_dict,
                 data[RNA_type, edge_type, 'drug'].edge_label_index)
    target = data[RNA_type, edge_type, 'drug'].edge_label.float()
    rmse = F.mse_loss(pred, target).sqrt()
    return float(rmse), pred.cpu().numpy(), target.cpu().numpy(),latent_feature

# Use CaptumExplainer to extract importance of nodes/edges in the heterogeneous graph for predicting test edges
def explain_graph(test_data,edge_type):
    # Initialize CaptumExplainer
    explainer = Explainer(
        model=model,
        algorithm=CaptumExplainer(attribution_method=attr.IntegratedGradients),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='binary_classification',
            task_level='edge',
            return_type='probs'
        )
    )
    # Retrieve indices and labels of the test edges
    edge_label_index = test_data[edge_type].edge_label_index
    edge_label = test_data[edge_type].edge_label

    node_importance = {node_type: [] for node_type in test_data.node_types}
    edge_importance = {edge_type: [] for edge_type in test_data.edge_types}
    # Explain specific test edge, such as index=3
    #         explanation = explainer(
    #               x=test_data.x_dict,
    #               edge_index=test_data.edge_index_dict,
    #               edge_label_index = edge_label_index,
    #               index = 3
    #             )
    for edge_idx in range(edge_label_index.size(1)):
        # Generate explanation
        explanation = explainer(
            x=test_data.x_dict,
            edge_index=test_data.edge_index_dict,
            edge_label_index=edge_label_index[:, edge_idx],
        )
        # Extract node masks and edge masks
        for node_type, mask in explanation.node_mask_dict.items():
            node_importance[node_type].append(mask.cpu().detach().numpy())
        for edge_type_one, mask in explanation.edge_mask_dict.items():
            edge_importance[edge_type_one].append(mask.cpu().detach().numpy())
    for node_type, importance_list in node_importance.items():
        # Retain importance of each feature of each node for predicting test edges
        # importance_matrix = np.vstack(importance_list)
        # df = pd.DataFrame(importance_matrix)
        # Retain overall importance of each node for predicting test edges
        for list_num in range(len(importance_list)):
            importance_list[list_num] = np.mean(importance_list[list_num], axis=1, keepdims=True)
        importance_matrix = np.concatenate(importance_list, axis=1)
        df = pd.DataFrame(importance_matrix)
        df.to_csv(f"./5.graph_important/val_{node_type}_importance_for_test_{edge_type}.csv", index=False)
    for edge_type_one, importance_list in edge_importance.items():
        importance_matrix = np.vstack(importance_list).T
        df = pd.DataFrame(importance_matrix)
        df.to_csv(f"./5.graph_important/val_{edge_type_one}_importance_for_test_{edge_type}.csv", index=False)

# %%
if __name__ == '__main__':
    # %%
    device = torch.device('cuda')
    # %%
    RNA_type_all = [['lncRNA', 'LncDrug'], ['miRNA', 'MiDrug']]
    for RNA_type, edge_type in RNA_type_all:
        print(RNA_type)
        print(edge_type)
        method_name = 'our'
        # load data
        train_data = torch.load('./3.heter_data/' + edge_type + '_train_data.pth')
        val_data = torch.load('./3.heter_data/' + edge_type + '_val_data.pth')
        test_data = torch.load('./3.heter_data/' + edge_type + '_test_data.pth')
        # %%
        if RNA_type == 'lncRNA':
           learning_rate = 0.0005
           epoch_num = 101
           channels = 64
           aggr =  'min'
        elif RNA_type == 'miRNA':
            learning_rate = 0.0001
            epoch_num = 301
            channels = 256
            aggr = 'mean'
        # %%
        model = Model(train_data, channels,aggr).to(device)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.99)

        # %%
        for epoch in range(1, epoch_num):
            loss = train(train_data)
            train_rmse, train_pred, train_true,latent_feature_train = test(train_data)
            #test_rmse, test_pred, test_true, latent_feature = test(val_data)
            test_rmse, test_pred, test_true,latent_feature = test(test_data)
            test_AUC = roc_auc_score(test_true, test_pred)
            test_AUPR = average_precision_score(test_true, test_pred)
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_rmse:.4f}, Val: {test_rmse:.4f}, Val:{test_AUC:.4f}, Val:{test_AUPR:.4f}')
            res_name = f'./4.results/{method_name}_{edge_type}_score.csv'
            df = pd.DataFrame({
                'y_scores': test_pred,
                'y_true': test_true
            })
            # If you do not need to save prediction results, annotate following command
            #df.to_csv(res_name, index=False)
            #np.save(f'./8.feature_visualization/latent_feature_{edge_type}_val.npy',latent_feature)
            #np.save(f'./8.feature_visualization/latent_feature_{edge_type}_test.npy', latent_feature)
        #torch.save(model, f'./4.results/model_{method_name}_{edge_type}.pth')
        # Use CaptumExplainer to extract importance of nodes/edges for predicting test edges. 
        # If importance experiment does not need to be calculated, annotate the following command
        #explain_graph(test_data, edge_type)
        #explain_graph(val_data, edge_type)
# %%
