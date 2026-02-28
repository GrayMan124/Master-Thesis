import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import numpy as np
from torch.utils.data import  random_split
from pathlib import Path
from tqdm import tqdm
from .topology.topologicalProcessing import AugmentAndCalculateFeatures, calculate_accurate_stats_two_pass, load_stats, save_stats
 
torch.autograd.set_detect_anomaly(True)

    
class PrecomputedDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, version_folders=None, transform=None):
        """
        data_dir: Path to 'data_cache'
        version_folders: List of subfolders to include. 
                         For training: ['train_v0', 'train_v1', ...]
                         For val: ['val']
        """
        self.version_folders = version_folders
        self.data_dir = Path(data_dir)
        self.files = []
        self.transform = transform
        first_folder = self.data_dir / version_folders[0]
        self.file_names = [p.name for p in first_folder.glob("*.pt") ]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        filename = self.file_names[idx]

        selected_version = np.random.choice(self.version_folders)
        full_path = self.data_dir / selected_version / filename
        img_tensor, topo, label = torch.load(full_path)

        if self.transform:
            img_tensor = self.transform(img_tensor)

        return (img_tensor, topo), label


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
    data_set = data_set['train'] 
    train_subset, val_subset = get_train_val_split(data_set = data_set, val_size = args.val_size) 
    save_path = Path(data_path)

    print("----- Calculating Topo statistcs ------- ")
    raw_stats_stransform = AugmentAndCalculateFeatures(
        args = args,
        train= False,
        pi_mean = None,
        pi_std=None
    )
    
    class WrapperDataset(torch.utils.data.Dataset):
        def __init__(self, subset, transform):
            self.subset = subset
            self.transform = transform
        def __len__(self):
            return len(self.subset)
        def __getitem__(self, index):
            item = self.subset[index]
            img,topo = self.transform(item['image'])
            return img, topo 

    stats_ds = WrapperDataset(train_subset, raw_stats_stransform)

    pi_mean, pi_std, max_t = calculate_accurate_stats_two_pass(stats_ds)

    save_stats(pi_mean, pi_std, max_t, save_path)
    print("Processing Validation Set...")
    val_save_path = save_path / "val"
    val_save_path.mkdir(parents=True, exist_ok=True)
    
    if args.maxNorm:
        processing_train = AugmentAndCalculateFeatures(train=True, args = args, pi_mean = [max_t], pi_std = [0])
        processing_val = AugmentAndCalculateFeatures(train=False, args = args, pi_mean = [max_t], pi_std = [0])
    else:
        processing_train = AugmentAndCalculateFeatures(train=True, args = args, pi_mean=pi_mean, pi_std=pi_std)
        processing_val = AugmentAndCalculateFeatures(train=False, args = args, pi_mean = pi_mean, pi_std = pi_std)

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


def process_test(data_set, data_path, args):
    # data_set = data_set['test'] 
    save_path = Path(data_path)
    try: 
        pi_mean, pi_std = load_stats(save_path)
        print(f"Loaded Training stats -- Mean: {pi_mean}, Std: {pi_std}")
    except:
        raise FileNotFoundError("Could not find topo_stats.json")

    save_path = Path(data_path)

    print("Processing Test Set...")
    test_save_path = save_path / "test"
    test_save_path.mkdir(parents=True, exist_ok=True)

    processing_test = AugmentAndCalculateFeatures(train=False, args = args, pi_mean= pi_mean, pi_std = pi_std)

    for idx in tqdm(range(len(data_set))):
        img_pil = data_set[idx]['image']
        label = data_set[idx]['label']
        tensor_data, topo_data = processing_test(img_pil)
        torch.save((tensor_data, topo_data, label), test_save_path/ f"{idx}.pt")

