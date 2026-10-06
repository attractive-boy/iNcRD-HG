import torch
from torch.utils.data import Dataset
import pandas as pd


class DrugTargetDataset(Dataset):
    """Dataset for drug-target pairs with pre-computed embeddings.

    CSV must have columns: SMILES, Protein
    Embedding files are dicts: {key_string -> torch.Tensor}
    """

    def __init__(self, csv_path, drug_embeds_path, target_embeds_path):
        self.data = pd.read_csv(csv_path)

        self.drug_embeds = torch.load(drug_embeds_path, weights_only=True)
        self.target_embeds = torch.load(target_embeds_path, weights_only=True)

        # Keep only pairs where both embeddings exist
        valid = [
            idx for idx, row in self.data.iterrows()
            if row['SMILES'] in self.drug_embeds and row['Protein'] in self.target_embeds
        ]
        self.data = self.data.iloc[valid].reset_index(drop=True)
        print(f"Loaded {len(self.data)} valid drug-target pairs")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        return {
            'drug': self.drug_embeds[row['SMILES']],
            'target': self.target_embeds[row['Protein']],
            'smile': row['SMILES'],
            'protein': row['Protein'],
        }
