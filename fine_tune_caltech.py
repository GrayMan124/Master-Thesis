import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["HF_HOME"] = "/mnt/sam/pi_data/"
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50 
from utils import count_parameters, train_model
from torchvision.datasets import Caltech256 

from models.ResNet50 import ResNet_50

from config import args

from datasets import load_dataset
 
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    


    
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda img: img.convert('RGB')),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456,0.406], std = [0.229, 0.224, 0.225])
        ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    ds_train = Caltech256(root = './data/',transform = transform_train, download= False)
    ds_val = Caltech256(root = './data/',transform = val_transforms, download= False)

    
    #Prepare model 
    model = resnet50(weights = "IMAGENET1K_V2")
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 257)

    model = model.to(device)    

    train_ratio = 0.8
    val_ratio = 1.0 - train_ratio
    total_size = len(ds_train) # Both datasets have the same length

    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size

    print(f"Total images: {total_size}")
    print(f"Splitting into {train_size} training and {val_size} validation images.")

    # # Create a generator with a fixed seed for reproducible splits
    generator = torch.Generator().manual_seed(42)

    # Split the dataset with *training transforms*
    train_subset, _ = random_split(
        ds_train, 
        [train_size, val_size],
        generator=generator
    )
    #
    # # Split the dataset with *validation transforms*
    # # We must reset the generator seed to get the same split
    generator.manual_seed(42)
    _, val_subset = random_split(
        ds_val,
        [train_size, val_size],
        generator=generator
    )
    #
    #
    # # Define batch size
    BATCH_SIZE = 32
    #
    # # Create the training DataLoader
    train_loader = DataLoader(
        dataset=train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,  # Shuffle the training data
        num_workers=4
    )
    #
    # # Create the validation DataLoader
    val_loader = DataLoader(
        dataset=val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False, # No need to shuffle validation data
        num_workers=4
    )
    epochs = 50
    #
    data_loaders = {'train':train_loader, 'val':val_loader}
    #
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)#, verbose=True)

    model, _ = train_model(model, data_loaders, criterion, optimizer, epochs)
