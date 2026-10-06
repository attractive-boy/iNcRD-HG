"""Reproduce AGCLNDA (official code) on the ncDR split.

The official DataHandler reads ./Datasets/mydata1/{train0,test0}; we write the
ncDR train/test matrices there (train = positives only), then run the official
Coach (LightGCN + VGAE/denoising view generators + graph contrastive) and add
our own full-metric evaluation on the shared test set.
"""
import os
import pickle
import random
import sys
from copy import deepcopy

import numpy as np
import scipy.sparse as sp
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
AGC = "/data1/licheng/incrd-hg/baselines/AGCLNDA"
sys.path.insert(0, AGC)
sys.path.insert(0, "/data1/licheng/incrd-hg/incrd-run")
os.chdir(AGC)
_argv = sys.argv[:]; sys.argv = ["x"]
from Params import args  # noqa: E402
sys.argv = _argv
from DataHandler import DataHandler  # noqa: E402
from Model import AGCLNDA, DenoisingNet, vgae, vgae_decoder, vgae_encoder  # noqa: E402
from Utils.Utils import calcRegLoss, pairPredict  # noqa: E402

sys.path.insert(0, HERE)
import common  # noqa: E402

DEV = "cuda:0"


def seed_it(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def write_ncdr(seed):
    pairs = common.load_ncdr()[0]
    msg, tpos = common.build_split(pairs, seed)
    P = common.ncdr_pos_set(pairs)
    tneg = common.sample_neg(P, tpos.shape[1], seed + 7)
    trn = sp.coo_matrix((np.ones(msg.shape[1]), (msg[0], msg[1])), shape=(common.NM, common.ND))
    tst = sp.coo_matrix((np.zeros(0), (np.zeros(0, int), np.zeros(0, int))),
                        shape=(common.NM, common.ND))
    os.makedirs("./Datasets/mydata1/", exist_ok=True)
    pickle.dump(trn, open("./Datasets/mydata1/train0", "wb"))
    pickle.dump(tst, open("./Datasets/mydata1/test0", "wb"))
    return tpos, tneg


def main():
    ms = []
    for seed in common.SEEDS:
        seed_it(seed)
        args.latdim = 512; args.epoch = 30; args.gnn_layer = 3
        tpos, tneg = write_ncdr(seed)
        handler = DataHandler(); handler.LoadData()

        model = AGCLNDA().to(DEV)
        gen1 = vgae(vgae_encoder().to(DEV), vgae_decoder(args.latdim).to(DEV)).to(DEV)
        gen2 = DenoisingNet(model.getGCN(), model.getEmbeds()).to(DEV)
        gen2.set_fea_adj(args.nc + args.drug, deepcopy(handler.torchBiAdj).to(DEV))
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        opt1 = torch.optim.Adam(gen1.parameters(), lr=args.lr)
        opt2 = torch.optim.Adam(filter(lambda p: p.requires_grad, gen2.parameters()),
                                lr=args.lr, eps=args.eps)

        loader = handler.trnLoader
        for ep in range(args.epoch):
            temp = max(0.05, args.init_temperature * pow(args.temperature_decay, ep))
            loader.dataset.negSampling()
            for tem in loader:
                data = deepcopy(handler.torchBiAdj).to(DEV)
                with torch.no_grad():
                    idxs = handler.torchBiAdj._indices()
                    data1 = gen1.generate(handler.torchBiAdj, idxs, handler.torchBiAdj)
                opt.zero_grad(); opt1.zero_grad(); opt2.zero_grad()
                ancs, poss, negs = [x.long().to(DEV) for x in tem]
                out1 = model.forward_graphcl(data1)
                out2 = model.forward_graphcl_(gen2)
                (model.loss_graphcl(out1, out2, ancs, poss).mean() * args.ssl_reg).backward()
                opt.step(); opt.zero_grad()
                _out1 = model.forward_graphcl(data1)
                _out2 = model.forward_graphcl_(gen2)
                loss_ib = (model.loss_graphcl(_out1, out1.detach(), ancs, poss)
                           + model.loss_graphcl(_out2, out2.detach(), ancs, poss))
                (loss_ib.mean() * args.ib_reg).backward()
                opt.step(); opt.zero_grad()
                ncE, dE = model.forward_gcn(data)
                bpr = -(pairPredict(ncE[ancs], dE[poss], dE[negs])).sigmoid().log().sum() / args.batch
                (bpr + calcRegLoss(model) * args.reg).backward()
                l1 = gen1(deepcopy(handler.torchBiAdj).to(DEV), ancs, poss, negs)
                l2 = gen2(ancs, poss, negs, temp)
                (l1 + l2).backward()
                opt.step(); opt1.step(); opt2.step()
            if ep % 10 == 0:
                print(f"  seed {seed} ep {ep} done", flush=True)

        model.eval()
        with torch.no_grad():
            ncE, dE = model.forward_gcn(handler.torchBiAdj)
            pp = (ncE[torch.tensor(tpos[0], device=DEV)]
                  * dE[torch.tensor(tpos[1], device=DEV)]).sum(-1).cpu().numpy()
            pn = (ncE[torch.tensor(tneg[0], device=DEV)]
                  * dE[torch.tensor(tneg[1], device=DEV)]).sum(-1).cpu().numpy()
        m = common.full_metrics(pp, pn)
        ms.append(m)
        print(f"seed {seed}: AUC {m['AUC']:.4f} AUPR {m['AUPR']:.4f} ACC {m['ACC']:.4f} "
              f"F1 {m['F1']:.4f} MCC {m['MCC']:.4f}", flush=True)

    print("\nAGCLNDA (ncDR same-protocol) means:")
    for k in ("AUC", "AUPR", "ACC", "Precision", "Recall", "Specificity", "F1", "MCC"):
        v = np.array([m[k] for m in ms]); print(f"  {k:12s} {v.mean():.4f} +- {v.std():.4f}")


if __name__ == "__main__":
    main()
