import os
import numpy as np
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_


class Trainer(torch.nn.Module):
    """Trains MetaMG and exports embeddings + the train/test edge split
    used by the DC_Prediction stage.
    """

    def __init__(self, args, model, splits: dict):
        """
        Args:
            args: argument namespace
            model: MetaMG model instance
            splits: dict with 'train' and 'test' PyG Data objects (from get_data)
        """
        super(Trainer, self).__init__()
        self.model = model
        self.args = args
        self.splits = splits
        self.num_m_step = args.m_step
        self.device = torch.device(args.device)
        self.clip_grad_norm = {
            "max_norm": args.clip_max_norm,
            "norm_type": args.clip_norm_type,
        }
        assert self.num_m_step is not None

    def train(self, train_data):
        """Run the training loop, then export embeddings and edge splits."""
        self.model.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=1e-3, weight_decay=5e-4
        )

        for epoch_idx in range(self.args.start_epoch, self.args.epochs):
            loss = self._train_epoch(train_data)
            if (epoch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch_idx + 1}/{self.args.epochs}] Loss: {loss}")

        self._save_embeddings()
        self._save_edge_splits()

    def _train_epoch(self, train_data):
        self.model.train()
        self.model.optimizer.zero_grad()

        losses = self.model.calculate_loss(train_data)
        loss = sum(losses) if isinstance(losses, tuple) else losses

        assert not torch.isnan(loss), "Loss is NaN"

        loss.backward()
        if self.clip_grad_norm:
            clip_grad_norm_(self.model.parameters(), **self.clip_grad_norm)
        self.model.optimizer.step()

        if isinstance(losses, tuple):
            return sum(l.item() for l in losses)
        return loss.item()

    def _save_embeddings(self):
        """Save the four learned embeddings used as DC_Prediction features."""
        out_dir = os.path.join('source', self.args.dataset)
        os.makedirs(out_dir, exist_ok=True)

        coarse_drug, coarse_rna, fine_drug, fine_rna = self.model.predict()

        np.savetxt(os.path.join(out_dir, 'Drug_GT.txt'), X=coarse_drug, newline='\n', encoding='UTF-8')
        np.savetxt(os.path.join(out_dir, 'NcRNA_GS.txt'), X=coarse_rna, newline='\n', encoding='UTF-8')
        np.savetxt(os.path.join(out_dir, 'Drug_MG.txt'), X=fine_drug, newline='\n', encoding='UTF-8')
        np.savetxt(os.path.join(out_dir, 'NcRNA_SQ.txt'), X=fine_rna, newline='\n', encoding='UTF-8')
        print(f"Embeddings saved to {out_dir}/")

    def _save_edge_splits(self):
        """Persist the positive/negative train/test edge indices so that
        DC_Prediction can reproduce the exact same split without data leakage.
        """
        out_dir = os.path.join('source', self.args.dataset)
        os.makedirs(out_dir, exist_ok=True)

        train_data = self.splits['train']
        test_data = self.splits.get('test', None)

        edge_splits = {
            'train_pos_edge_index': train_data.pos_edge_label_index.cpu(),
            'train_neg_edge_index': train_data.neg_edge_label_index.cpu(),
        }
        if test_data is not None:
            edge_splits['test_pos_edge_index'] = test_data.pos_edge_label_index.cpu()
            edge_splits['test_neg_edge_index'] = test_data.neg_edge_label_index.cpu()

        save_path = os.path.join(out_dir, 'edge_splits.pt')
        torch.save(edge_splits, save_path)
        print(f"Edge splits saved to {save_path}")
