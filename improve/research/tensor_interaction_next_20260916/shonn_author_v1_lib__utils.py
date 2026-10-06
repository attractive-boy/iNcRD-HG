import os
import json
import torch
from torch.utils.data import DataLoader

def get_run_dir(base_dir):
    """
    This script is made to handle multiple training runs.
    If not present it creates the base directory (e.g. "mnist"), and then
    creates the subdirectory "run_0", in which to save the results.
    If the base directory already exists, it looks for the next available "run_i" subdirectory and creates it.
    Returns the path to the new run directory.
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    i = 0
    while True:
        run_dir = os.path.join(base_dir, f"run_{i}")
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)
            return run_dir
        i += 1

def save_history(run_dir, model_name, history):
    """
    Save the training history to a JSON file in the specified run directory.
    """
    os.makedirs(os.path.join(run_dir, model_name), exist_ok=True)
    with open(os.path.join(run_dir, model_name, "history.json"), "w") as f:
        json.dump(history, f)

def save_final_test_evaluation(run_dir, model_name, test_evaluation):
    """
    Save the final test evaluation results to a JSON file in the specified run directory.
    """
    os.makedirs(os.path.join(run_dir, model_name), exist_ok=True)
    with open(os.path.join(run_dir, model_name, "final_test_evaluation.json"), "w") as f:
        json.dump(test_evaluation, f)

def save_model(run_dir, model_name, model):
    """
    Save the model's state dictionary to a file in the specified run directory.
    """
    os.makedirs(os.path.join(run_dir, model_name), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(run_dir, model_name, "model_final.pth"))

def load_all_histories_of_a_model(base_dir, model_name):
    """
    Load all history.json files for the specified model across all runs in the base directory.
    Returns a list of history dictionaries.
    """

    histories = []
    for entry in sorted(os.listdir(base_dir)):
        run_dir = os.path.join(base_dir, entry)
        # Ignore files such as summary plots
        if not os.path.isdir(run_dir):
            continue
        history_path = os.path.join(run_dir, model_name, "history.json")
        with open(history_path, "r") as f:
            history = json.load(f)
            histories.append(history)
    return histories


def compute_dataset_mean_std(dataset, batch_size=512):
    """
    Compute per-channel normalization statistics (mean, std).

    Expected dataset sample format: (image, label), where image is a tensor
    shaped (C, H, W) with values in [0, 1] (typically after ToTensor).

    Method:
    1) Sum pixel values per channel across the full dataset.
    2) Sum squared pixel values per channel across the full dataset.
    3) Use E[X] and E[X^2] to compute variance:
       var = E[X^2] - (E[X])^2
       std = sqrt(var)

    Returns:
    - mean: tuple[float, ...] (length = number of channels)
    - std:  tuple[float, ...] (length = number of channels)
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    channel_sum = None
    channel_sum_sq = None
    num_pixels = 0

    for images, _ in loader:
        # images shape: (B, C, H, W)
        batch_pixels = images.size(0) * images.size(2) * images.size(3)
        num_pixels += batch_pixels

        # Aggregate first and second moments per channel.
        sum_per_channel = images.sum(dim=(0, 2, 3))
        sum_sq_per_channel = (images * images).sum(dim=(0, 2, 3))

        if channel_sum is None:
            channel_sum = sum_per_channel
            channel_sum_sq = sum_sq_per_channel
        else:
            channel_sum += sum_per_channel
            channel_sum_sq += sum_sq_per_channel

    # E[X] and E[X^2] over all pixels per channel.
    mean = channel_sum / num_pixels
    variance = channel_sum_sq / num_pixels - mean * mean
    std = torch.sqrt(variance)

    return tuple(mean.tolist()), tuple(std.tolist())

def save_regression_reconstruction(run_dir, model_name, inputs, targets, reconstructions):
    """
    Save the inputs, targets, and reconstructions for regression tasks to a JSON file.
    """
    os.makedirs(os.path.join(run_dir, model_name), exist_ok=True)
    data = {
        "inputs": inputs.tolist(),
        "targets": targets.tolist(),
        "reconstructions": reconstructions.tolist()
    }
    with open(os.path.join(run_dir, model_name, "regression_reconstruction.json"), "w") as f:
        json.dump(data, f)
    