import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["HF_HOME"] = "/mnt/sam/pi_data/"
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from utils import count_parameters, train_model
from torchvision.models import resnet50
from huggingface_hub import login


from models.ResNet50 import ResNet_50

from config import args

from datasets import load_dataset
 
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

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
    
def train_transforms_batched(examples):
    examples['pixel_values'] = [transform_train(img.convert("RGB")) for img in examples['image']]
    return examples

def val_transforms_batched(examples):
    examples['pixel_values'] = [val_transforms(img.convert("RGB")) for img in examples['image']]
    return examples

def fast_collate(batch):
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    return pixel_values, labels

if __name__ == '__main__':
    args.model = "ResNet" 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    
    with open('../HF_API_code.txt','r') as file:
        API_CODE = file.read()

    API_CODE = API_CODE.replace('\n','')

    HUGGINGFACE_TOKEN = API_CODE
    login(token=HUGGINGFACE_TOKEN)


    all_var = os.environ

    hf_cache = os.getenv("HF_HOME")
    print(f"My Hugging Face cache is at: {hf_cache}")
    
    ds = load_dataset("zh-plus/tiny-imagenet", streaming= False)
    
    ds['train'] = ds['train'].with_transform(train_transforms_batched)
    ds['valid'] = ds['valid'].with_transform(val_transforms_batched)

    train_dataset = ds['train'].shuffle(seed = 42)
    val_dataset = ds['valid']


    batch_size = 32
    epochs = 50

    train_loader = DataLoader(train_dataset, batch_size = batch_size, collate_fn = fast_collate, num_workers = 4, pin_memory = True, shuffle = True)

    val_loader = DataLoader(val_dataset, batch_size = batch_size, collate_fn = fast_collate, num_workers = 4, pin_memory = True)

    data_loaders = {'train':train_loader, 'val':val_loader}

    model = ResNet_50(3,200)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)#, verbose=True)

    model, _ = train_model(model, data_loaders, criterion, optimizer, epochs)
