import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use only the second GPU (index 1)

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from lib.layers import SpectralTriadic, DirectSpaceTriadic
from lib.utils import get_run_dir
from lib.utils import save_history, save_final_test_evaluation, save_model
from lib.utils import compute_dataset_mean_std
from lib.train_and_evaluate_classification import train_and_evaluate

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Setup save directory
run_dir = get_run_dir("mnist")

# Choice of criterion
criterion = nn.CrossEntropyLoss()

# MODELS
class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # MNIST images are 28x28 = 784
        self.layer = nn.Linear(784, 10)

    def forward(self, x):
        x = self.flatten(x)
        return self.layer(x)

class TriadicPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # MNIST images are 28x28 = 784
        self.layer = SpectralTriadic(784, 10)

    def forward(self, x):
        x = self.flatten(x)
        return self.layer(x)
    
class DirectSpaceTriadicPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        # MNIST images are 28x28 = 784
        self.layer = DirectSpaceTriadic(784, 10)

    def forward(self, x):
        x = self.flatten(x)
        return self.layer(x)

# Data loading
stats_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
mean, std = compute_dataset_mean_std(stats_dataset)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Split train into train and validation (e.g., 50000 train, 10000 val)
train_size = 50000
val_size = 10000
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

models = [
    (Perceptron().to(device), "direct_linear"),
    (TriadicPerceptron().to(device), "spectral_triadic"),
    (DirectSpaceTriadicPerceptron().to(device), "direct_space_triadic")
]

# Train each model
epochs = 50
learning_rate = 1e-2
warmup_epochs = 10
patience = 5
for model, name in models:
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    scheduler_warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    
    scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=patience, min_lr=1e-6
    )

    history, final_test_evaluation, model = train_and_evaluate(
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
        verbose=True
        )
    
    # Save results
    save_history(run_dir, name, history)
    save_final_test_evaluation(run_dir, name, final_test_evaluation)
    save_model(run_dir, name, model)