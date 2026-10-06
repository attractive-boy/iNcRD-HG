from __future__ import division
from __future__ import print_function

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
from warnings import simplefilter
import numpy as np
import torch
from train import train, test
import random
import seaborn
simplefilter(action='ignore', category=FutureWarning)

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--nhid', type=int, default=64, help='the dimension of hidden layer')
parser.add_argument('--epoch', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--k', type=int, default=10,
                    help='k of kneighbors_graph.')
parser.add_argument('--gamma', type=float, default=0.8)

parser.add_argument('--rep_num', type=int, default=1,
                    help='Number of rep.')
parser.add_argument('--train_ratio', type=float, default=0.1,
                    help='train_ratio')
parser.add_argument('--valid_ratio', type=float, default=0,
                    help='valid_ratio')
# parser.add_argument('--test_ratio', type=float, default=0.9,
#                     help='test_ratio')
parser.add_argument('--data_split_mode', type=str, default='Ratio',
                    help='data_split_mode, [Ratio, Num]')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')


parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default="HW",
                    help='Dataset to use.')

parser.add_argument('--max_iter', type=int, default=200)
parser.add_argument('--threshold', type=float, default=1e-6)
parser.add_argument('--momentum', type=float, default=0.8)
parser.add_argument('--device', type=str, default="0", help='gpu')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
# print(args)

device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
torch.backends.cudnn.enabled = True



res = str(args) + '\n'
acc = []
f1 = []
for i in range(args.rep_num):
    print('rep_num:', i + 1)
    model, features, labels, adj, idx_test = train(args, device)
    # Testing
    acc_test, f1_score = test(args, model, features, labels, adj, idx_test)
    acc.append(acc_test)
    f1.append(f1_score)
    del model
res += 'ACC: ' + str(round(np.array(acc).mean(), 2)) + ' std:' + str(round(np.array(acc).std(), 2)) + ' ' + 'F1: ' + str(round(np.array(f1).mean(), 2)) + ' std:' + str(round(np.array(f1).std(), 2)) + '\n\n'
print("Optimization Finished!")

print("accuracy_mean= {:.4f}".format(np.array(acc).mean()), "accuracy_std= {:.4f}".format(np.array(acc).std()))
print("f1_mean= {:.4f}".format(np.array(f1).mean()), "f1_std= {:.4f}".format(np.array(f1).std()))

