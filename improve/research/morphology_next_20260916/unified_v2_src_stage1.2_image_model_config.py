import torch

CONFIG = {
    # Data paths (relative to repo root)
    'train_csv': 'Stage1.2_image/data/train_pairs.csv',
    'test_csv': 'Stage1.2_image/data/test_pairs.csv',
    'drug_embeds_path': 'Stage1.2_image/embedding/All_compound_image.pt',
    'target_embeds_path': 'Stage1.2_image/embedding/All_CRISPR_image.pt',

    # Model hyperparameters
    'hidden_dim': 1024,
    'output_dim': 512,
    'dropout': 0.1,

    # Training hyperparameters
    'batch_size': 256,
    'learning_rate': 1e-4,
    'num_epochs': 30,
    'temperature': 0.10,
    'weight_decay': 1e-5,

    # Output
    'save_dir': 'Stage1.2_image/model/checkpoints',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}
