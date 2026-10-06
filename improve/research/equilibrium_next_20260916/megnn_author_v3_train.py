from __future__ import division
from __future__ import print_function

import torch
import time
from DataLoader import LoadMatData, generate_permutation
import torch.nn.functional as F
import torch.optim as optim
from utils import accuracy, f1_test, draw_plt
from models import MEGNN
from tqdm import tqdm
from sklearn.preprocessing import normalize


def train(args, device):

    adj, features, labels, nfeats, num_view, num_class = LoadMatData(args.dataset, args.k, 'E:\dataset\multiview datasets\\')
    idx_train, idx_test = generate_permutation(labels, args)
    model = MEGNN(adj[0].shape[0], num_view, num_class, nfeats, args.nhid, args.threshold, args.max_iter,
                        args.gamma)
    total_para = sum(x.numel() for x in model.parameters())
    print("Total number of paramerters in networks is {}  ".format(total_para / 1e6))
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)

    for i in range(num_view):
        # exec("features_{}= torch.from_numpy(features[{}]/1.0).float().to(device)".format(i, i))
        exec("features[{}]= normalize(features[{}])".format(i, i))
        exec("features[{}]= torch.Tensor(features[{}] / 1.0).to(device)".format(i, i))
        exec("adj[{}]=adj[{}].to_dense().float().to(device)".format(i, i))

    if args.cuda:
        model.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()  # [:10]
        idx_test = idx_test.cuda()
    t1 = time.time()
    with tqdm(total=args.epoch) as pbar:
        pbar.set_description('Training:')
        for i in range(args.epoch):
            t = time.time()
            model.train()
            optimizer.zero_grad()
            output = model(features, adj)
            output = F.log_softmax(output, dim=1)
            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            acc_train = accuracy(output[idx_train], labels[idx_train])
            f1_train = f1_test(output[idx_train], labels[idx_train])
            loss_train.backward()
            # clip_gradient(model, clip_norm=0.5)
            optimizer.step()


            loss_test = F.nll_loss(output[idx_test], labels[idx_test])
            acc_test = accuracy(output[idx_test], labels[idx_test])
            f1 = f1_test(output[idx_test], labels[idx_test])

            outstr = 'Epoch: {:04d} '.format(i + 1) + \
                     'loss_train: {:.4f} '.format(loss_train.item()) + \
                     'acc_train: {:.4f} '.format(acc_train.item()) + \
                     'loss_test: {:.4f} '.format(loss_test.item()) + \
                     'acc_test: {:.4f} '.format(acc_test.item()) + \
                     'f1_test: {:.4f} '.format(f1.item()) + \
                     'time: {:.4f}s'.format(time.time() - t)
            pbar.set_postfix_str(outstr)
            # print(outstr)
            pbar.update(1)
    # return model, loss_val.item(), acc_val.item(), loss_test.item(), acc_test.item()
    # draw_plt(output[idx_test], labels[idx_test], args.dataset)
    print('total_time:', time.time() - t1)
    return model, features, labels, adj, idx_test


def test(args, model, features, labels, adj, idx_test):
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    f1 = f1_test(output[idx_test], labels[idx_test])
    print("Dataset:", args.dataset, "gamma:", args.gamma)
    print("Test set results:",
          "accuracy= {:.2f}".format(100 * acc_test.item()),
          "f1= {:.2f}".format(100 * f1.item()))
    return 100 * acc_test.item(), 100 * f1.item()
