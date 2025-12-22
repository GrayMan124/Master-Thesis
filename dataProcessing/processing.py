import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["HF_HOME"] = "/mnt/sam/pi_data/"
import torch
from torch.utils.data import  random_split
from pathlib import Path
from tqdm import tqdm
from .topology.topologicalProcessing import AugmentAndCalculateFeatures
 
torch.autograd.set_detect_anomaly(True)

    
class PrecomputedDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, version_folders=None):
        """
        data_dir: Path to 'data_cache'
        version_folders: List of subfolders to include. 
                         For training: ['train_v0', 'train_v1', ...]
                         For val: ['val']
        """
        self.files = []
        for v_folder in version_folders:
            path = Path(data_dir) / v_folder
            self.files.extend(list(path.glob("*.pt")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_tensor, topo, label = torch.load(self.files[idx])
        return (img_tensor, topo), label

# def apply_train_transformation(batch):
#     # print(batch['image'])
#     # images = [transform_train(img.convert('RGB')) for img in batch['image']]
#     images = transform_train(batch['image'].convert('RGB'))
#     batch['pixel_values'] = images 
#
#     return batch
#
#
# def apply_val_transformation(batch):
#     images = [val_transforms(img.convert('RGB')) for img in batch['image']]
#     batch['pixel_values'] = images 
#
#     return batch
#
# def custom_collate(batch_list):
#     inputs = torch.stack([item['pixel_values'] for item in batch_list])
#
#     labels = torch.tensor([item['label'] for item in batch_list])
#
#     return inputs, labels

def get_train_val_split(data_set, val_size):
    train_ratio = 1 - val_size 
    total_size = len(data_set) # Both datasets have the same length
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size

    print(f"Total images: {total_size}")
    print(f"Splitting into {train_size} training and {val_size} validation images.")

    split_gen = torch.Generator().manual_seed(119)
    train_subset,  val_subset = random_split(
        data_set, 
        [train_size, val_size],
        generator = split_gen 
    )
    return train_subset, val_subset

def process_data(data_set, data_path, num_versions,  args):
    
    train_subset, val_subset = get_train_val_split(data_set = data_set, val_size = args.val_size) #TODO: change this 0.8 to args.train_split
    save_path = Path(data_path)

    # --- PROCESS VALIDATION (Do this once) ---
    print("Processing Validation Set...")
    val_save_path = save_path / "val"
    val_save_path.mkdir(parents=True, exist_ok=True)

    processing_train = AugmentAndCalculateFeatures(train=True)
    processing_val = AugmentAndCalculateFeatures(train=False)

    for idx in tqdm(range(len(val_subset))):
        img_pil = val_subset[idx]['image']
        label = val_subset[idx]['label']
        tensor_data, topo_data = processing_val(img_pil)
        torch.save((tensor_data, topo_data, label), val_save_path / f"{idx}.pt")

    for v in range(num_versions): 
        print(f"Processing version: {v}") 
        version_path = save_path / f"train_v{v}"
        version_path.mkdir(parents=True, exist_ok=True)
        
        for idx in tqdm(range(len(train_subset))):
            img_pil = train_subset[idx]['image']
            label = train_subset[idx]['label']
            tensor_data, topo_data = processing_train(img_pil)
            torch.save((tensor_data, topo_data, label), version_path / f"{idx}.pt")

