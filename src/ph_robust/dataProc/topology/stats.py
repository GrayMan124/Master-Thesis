import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import cv2
import gudhi as gd
import gudhi.representations
from tqdm import tqdm
import json


def calculate_dataset_stats(dataset_loader):
    """
    Iterates through the dataset to calculate Mean and Std for Topology channel.
    """
    print("--- Calculating Topology Statistics (This takes a moment) ---")
    cnt = 0
    fst_moment = torch.empty(3)  # Placeholder
    snd_moment = torch.empty(3)
    max_t = 0

    # We only need to check the first batch to get channel count
    for i, (images, topo_features) in enumerate(tqdm(dataset_loader)):
        # topo_features shape: (Batch, Channels, H, W)
        b, c, h, w = topo_features.shape
        nb_pixels = b * h * w

        if i == 0:
            fst_moment = torch.zeros(c)
            snd_moment = torch.zeros(c)

        # Calculate sum and sum_of_squares across (Batch, H, W)
        # leaving Channels dimension intact
        sum_ = torch.sum(topo_features, dim=[0, 2, 3])
        sum_sq_ = torch.sum(topo_features**2, dim=[0, 2, 3])

        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_sq_) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean = fst_moment
    std = torch.sqrt(snd_moment - fst_moment**2)

    print(f"Calculated Mean: {mean}")
    print(f"Calculated Std: {std}")
    return mean.tolist(), std.tolist()


def calculate_accurate_stats_two_pass(dataset):
    """
    Robust Two-Pass algorithm to calculate Mean and Std.
    Essential for high-variance data like Persistent Images.
    """
    # Create a simple loader just for stats
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8)

    print("--- [Pass 1/2] Calculating Global Mean ---")
    pixel_sum = None
    total_pixels = 0
    n_channels = None
    max_t = 0
    # PASS 1: Mean
    for i, (images, topo_features) in enumerate(tqdm(loader)):
        # topo_features: (B, C, H, W)
        if n_channels is None:
            n_channels = topo_features.shape[1]
            pixel_sum = torch.zeros(
                n_channels, dtype=torch.float64
            )  # Float64 for safety

        # Sum over Batch(0), Height(2), Width(3) -> Result (C,)
        pixel_sum += torch.sum(topo_features, dim=[0, 2, 3]).double()
        if torch.max(topo_features) > max_t:
            max_t = torch.max(topo_features)

        # Count pixels (B * H * W)
        total_pixels += (
            topo_features.shape[0] * topo_features.shape[2] * topo_features.shape[3]
        )

    global_mean = pixel_sum / total_pixels
    print(f"Global Mean calculated: {global_mean.tolist()}")

    # PASS 2: Std
    print("--- [Pass 2/2] Calculating Global Std ---")
    sum_squared_diff = torch.zeros(n_channels, dtype=torch.float64)

    for i, (images, topo_features) in enumerate(tqdm(loader)):
        topo_features = topo_features.double()
        # Reshape mean for broadcasting: (1, C, 1, 1)
        mean_view = global_mean.view(1, n_channels, 1, 1)

        # (x - mean)^2
        diff = (topo_features - mean_view) ** 2
        sum_squared_diff += torch.sum(diff, dim=[0, 2, 3])

    global_var = sum_squared_diff / total_pixels
    global_std = torch.sqrt(global_var)

    print(f"Global Std calculated: {global_std.tolist()}")

    return (
        global_mean.float().tolist(),
        global_std.float().tolist(),
        [max_t.float().tolist()],
    )


def save_stats(mean, std, max_t, path):
    json_dict = {"mean": mean, "std": std, "max": max_t}
    print(json_dict)
    with open(path / "topo_stats.json", "w") as f:
        json.dump({"mean": mean, "std": std, "max": max_t}, f)


def load_stats(path):
    with open(path / "topo_stats.json", "r") as f:
        data = json.load(f)
    return data["mean"], data["std"], data["max"]
