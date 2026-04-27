import os
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

from ph_robust.data_processing.processing import process_data
from ph_robust.data_processing.datasets import PrecomputedDataset


def build_dataloaders(cfg):
    data_path = cfg.data.path
    if cfg.topo.max_norm:
        data_path = data_path + "mn"
    versions = [f"train_v{i}" for i in range(cfg.data.num_versions)]

    if not os.path.isdir(os.path.join(data_path, versions[-1])):
        print("----- Processed Data not found ------")
        ds = load_dataset("zh-plus/tiny-imagenet")
        process_data(
            data_set=ds,
            data_path=data_path,
            num_versions=cfg.data.num_versions,
            args=cfg,
        )
    else:
        print("----- Using Cached Dataset ----- ")

    resize = transforms.Compose([transforms.Resize((224, 224), antialias=True)])
    train_ds = PrecomputedDataset(data_path, version_folders=versions, transform=resize)
    val_ds = PrecomputedDataset(data_path, version_folders=["val"], transform=resize)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
