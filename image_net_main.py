import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["HF_HOME"] = "/mnt/sam/pi_data/"
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
#imports from the files
from utils import count_parameters, train_model
# from utils import test_model, train_model, count_parameters, MyDataset
# from models.ResNet import *
# from data_processing import *
# from models.TopoResNet import *
# from models.ResNetPIBlock import *
# from models.ResNetTopo2Dim import *
# from models.ResNetTopoBlock import *
# from models.TopoResNet import *
# from models.TopoResNetPI import * 
# from models.ResNet import *

from models.ResNet50 import ResNet_50

from config import args

from datasets import load_dataset
 
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(args.seed)


transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])
    ])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    

    all_var = os.environ

    hf_cache = os.getenv("HF_HOME")
    print(f"My Hugging Face cache is at: {hf_cache}")
    

    ds = load_dataset("ILSVRC/imagenet-1k",streaming = True)

    train_stream = ds['train'].map(apply_train_transformation)
    val_stream = ds['validation'].map(apply_val_transformation)

    train_stream_shuffled = train_stream.shuffle(buffer_size= 1000, seed = 42)


    batch_size = 32
    epochs = 50

    train_loader = DataLoader(train_stream_shuffled, batch_size = batch_size, collate_fn = custom_collate, num_workers = 4)

    val_loader = DataLoader(val_stream, batch_size = batch_size, collate_fn = custom_collate, num_workers = 4)

    data_loaders = {'train':train_loader, 'val':val_loader}

    model = ResNet_50(3,1000)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)#, verbose=True)

    model, _ = train_model(model, data_loaders, criterion, optimizer, epochs)
