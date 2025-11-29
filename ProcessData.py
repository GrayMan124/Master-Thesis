import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["HF_HOME"] = "/mnt/sam/pi_data/"
import torch
import numpy as np 
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import Caltech256 
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from config import args

from datasets import load_dataset

from topoTransform import process_PI, AugmentAndCalculateFeatures
 
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(args.seed)

    
def apply_train_transformation(batch):
    # print(batch['image'])
    # images = [transform_train(img.convert('RGB')) for img in batch['image']]
    images = transform_train(batch['image'].convert('RGB'))
    batch['pixel_values'] = images 

    return batch


def apply_val_transformation(batch):
    images = [val_transforms(img.convert('RGB')) for img in batch['image']]
    batch['pixel_values'] = images 

    return batch
def custom_collate(batch_list):
    inputs = torch.stack([item['pixel_values'] for item in batch_list])
    
    labels = torch.tensor([item['label'] for item in batch_list])
    
    return inputs, labels

if __name__ == '__main__':
    args.model = "ResNet" 

    ds = Caltech256(root = './data/', download= False)

    train_ratio = 0.8
    val_ratio = 1.0 - train_ratio
    total_size = len(ds) # Both datasets have the same length

    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size

    print(f"Total images: {total_size}")
    print(f"Splitting into {train_size} training and {val_size} validation images.")

    # Create a generator with a fixed seed for reproducible splits
    
    split_gen = torch.Generator().manual_seed(119)
    # # Split the dataset with *training transforms*
    train_subset,  val_subset = random_split(
        ds, 
        [train_size, val_size],
        generator = split_gen 
    )
    

    
    print(len(train_subset))
    data_dir = './data/caltech256_processed/'

    save_path = Path(data_dir)

    # --- PROCESS VALIDATION (Do this once) ---
    print("Processing Validation Set...")
    val_save_path = save_path / "val"
    val_save_path.mkdir(parents=True, exist_ok=True)

    processing_train = AugmentAndCalculateFeatures(train=True)
    processing_val = AugmentAndCalculateFeatures(train=False)

    for idx in tqdm(range(len(val_subset))):
        img_pil, label = val_subset.__getitem__(idx) # Get raw PIL image

        # Apply non-augmented transform (Resize + TDA)
        tensor_data, topo_data = processing_val(img_pil)

        # Save tuple (Image, Topo, Label)
        torch.save((tensor_data, topo_data, label), val_save_path / f"{idx}.pt")
    for v in range(5): 
        print(f"Processing version: {v}") 
        version_path = save_path / f"train_v{v}"
        version_path.mkdir(parents=True, exist_ok=True)
        
        for idx in tqdm(range(len(train_subset))):
            img_pil, label = train_subset.__getitem__(idx)
            
            # Apply augmented transform (RandomCrop + TDA)
            # Because global RNG changes, this will be different every time
            tensor_data, topo_data = processing_train(img_pil)
            
            torch.save((tensor_data, topo_data, label), version_path / f"{idx}.pt")

