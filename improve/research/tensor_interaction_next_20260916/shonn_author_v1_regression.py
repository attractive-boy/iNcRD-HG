import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from lib.layers import SpectralTriadic
from lib.utils import get_run_dir
from lib.utils import save_history, save_final_test_evaluation, save_model, save_regression_reconstruction
from lib.train_and_evaluate_regression import train_and_evaluate_regression, reconstruction

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Setup save directory
run_dir = get_run_dir("regression")

# Choice of criterion
criterion = nn.MSELoss()


class SpectralTriadicMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            SpectralTriadic(2, 20),
            SpectralTriadic(20, 1)
        )

    def forward(self, x):
        return self.model(x)


def target_function(xy):
    x = xy[:, 0]
    y = xy[:, 1]
    z = x**4 + y**4 + 2 * x**2 * y**2 - 2 * x**2
    return z.unsqueeze(1)


def make_regression_dataset(num_samples, low=-2.0, high=2.0):
    inputs = torch.empty(num_samples, 2).uniform_(low, high)
    targets = target_function(inputs)
    return TensorDataset(inputs, targets)


# Dataset: x, y sampled in [-2, 2]
full_dataset = make_regression_dataset(num_samples=40000, low=-2.0, high=2.0)

# Split: 70% train, 15% val, 15% test
train_size = int(0.70 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size]
)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

model = SpectralTriadicMLP().to(device)
model_name = "spectral_triadic_mlp"

learning_rate = 1e-3
epochs = 100
warmup_epochs = 10
patience = 10

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

scheduler_warmup = optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
)

scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=patience, min_lr=1e-6
)

history, final_test_evaluation, model = train_and_evaluate_regression(
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    epochs,
    device,
    scheduler_warmup=scheduler_warmup,
    scheduler_plateau=scheduler_plateau,
    verbose=True,
)

# Save results
save_history(run_dir, model_name, history)
save_final_test_evaluation(run_dir, model_name, final_test_evaluation)
save_model(run_dir, model_name, model)

# Build reconstruction data and save
inputs, true_profile, reconstructions = reconstruction(test_loader, model, device)
save_regression_reconstruction(run_dir, model_name, inputs, true_profile, reconstructions)

