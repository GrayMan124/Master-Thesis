from torch.utils.data import DataLoader
from datasets import AugmentAndCalculateFeatures
from torch.utils.data import Subset
import numpy as np


def get_topo_DS(dir_path, dataset, args):
    train_transform = AugmentAndCalculateFeatures(train=True, args=args)
    val_transform = AugmentAndCalculateFeatures(train=False, args=args)

    train_set_full = dataset(root=dir_path, transform=train_transform, download=False)
    val_set_full = dataset(root=dir_path, transform=val_transform, download=False)

    dataset_len = len(train_set_full)
    indicies = list(range(dataset_len))

    train_size = int(dataset_len * (1 - args.val_size))
    # val_size = dataset_len - train_size

    np.random.seed(42)
    np.random.shuffle(indicies)
    train_idx, val_idx = indicies[:train_size], indicies[train_size:]

    train_subset = Subset(train_set_full, train_idx)
    val_subset = Subset(val_set_full, val_idx)

    print(f"Training samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader
