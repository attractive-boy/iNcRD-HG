
import torch
import numpy as np
import random
from abc import ABC, abstractmethod
import os 
import math  

def select_optimizer(parameters, str_optimizer, lr, weight_decay, momentum):
    if str_optimizer == 'Adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif str_optimizer == 'SGD':
        return torch.optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif str_optimizer == 'AdamW':
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    elif str_optimizer == 'Adamax':
        return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
    elif str_optimizer == 'NAdam':
        return torch.optim.NAdam(parameters, lr=lr, weight_decay=weight_decay)
    elif str_optimizer == 'RMSprop':
        return torch.optim.RMSprop(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif str_optimizer == 'Rprop':
        return torch.optim.Rprop(parameters, lr=lr)
    elif str_optimizer == 'Adagrad':
        return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
    elif str_optimizer == 'Adadelta':
        return torch.optim.Adadelta(parameters, lr=lr, weight_decay=weight_decay)
    elif str_optimizer == 'ASGD':
        return torch.optim.ASGD(parameters, lr=lr, weight_decay=weight_decay)
    elif str_optimizer == 'SparseAdam':
        return torch.optim.SparseAdam(parameters, lr=lr)
    else:
        raise ValueError(f"Optimizer '{str_optimizer}' not recognized")

def load_model_weights_path(dataset_name, seeds):

    if dataset_name == 'WikiCS' and seeds == 2318:
        path = 'models/WikiCS/2318/WikiCS_2318.pth'
    elif dataset_name == 'WikiCS' and seeds == 42:
        path = 'models/WikiCS/42/WikiCS.pth'
    elif dataset_name == 'Cora':
        path = 'models/Cora/Cora.pth'
    elif dataset_name == 'CiteSeer':
        path = 'models/CiteSeer/CiteSeer.pth'
    elif dataset_name == 'PubMed':
        path = 'models/PubMed/PubMed.pth'
    return path

def load_conv_weights_path(dataset_name, seeds):

    if dataset_name == 'WikiCS' and seeds == 2318:
        path1 = 'models/WikiCS/2318/Cheb_WikiCS_2318.pth'
        path2 = 'models/WikiCS/2318/GATv2_WikiCS_2318.pth'
    elif dataset_name == 'WikiCS' and seeds == 42:
        path1 = 'models/WikiCS/42/Cheb_WikiCS.pth'
        path2 = 'models/WikiCS/42/GATv2_WikiCS.pth'
    elif dataset_name == 'Cora':
        path1 = 'models/Cora/Cheb_Cora.pth'
        path2 = 'models/Cora/GCN_Cora.pth'
    elif dataset_name == 'CiteSeer':
        path1 = 'models/CiteSeer/SGC_CiteSeer.pth'
        path2 = 'models/CiteSeer/SSGC_CiteSeer.pth'
    elif dataset_name == 'PubMed':
        path1 = 'models/PubMed/Cheb_PubMed.pth'
        path2 = 'models/PubMed/Graph_PubMed.pth'
    return path1,path2

def get_dataset(dataset_name,seeds):
    if dataset_name == 'WikiCS' and seeds == 2318:
        dataset = torch.load('data/WikiCS_data_2318.pt')
    elif dataset_name == 'WikiCS' and seeds == 42:
        dataset = torch.load('data/WikiCS_data_42.pt')
    elif dataset_name == 'Cora':
        dataset = torch.load('data/Cora_data.pt')
    elif dataset_name == 'CiteSeer':
        dataset = torch.load('data/CiteSeer_data.pt')
    elif dataset_name == 'PubMed':
        dataset = torch.load('data/PubMed_data.pt')
    else:
        print('This dataset does not exist')
    
    return dataset



def set_seed(seed):
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)  
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False    


def compute_distance_squared(point_feature, cls_centroid):
    distance_squared = torch.norm(point_feature - cls_centroid, p=2) ** 2
    return distance_squared

