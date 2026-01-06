#Update, version from: 22-12-2025 
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50 
from torchvision import transforms 
from datasets import load_dataset

from config.config import args
from dataProcessing.processing import process_data, PrecomputedDataset
from utils import train_model, seed_all
from models.PI_finetune import PIFineTuneModel
from models.FineTuneResNet import ResNetFineTune


tensor_board_path = 'runs/' + args.name + args.model + '_' + args.tv + '_' + str(args.lr) + '_' + str(args.res) + '_' + str(args.seed) + '_' + str(args.topodim)   

if __name__ == '__main__':
    print("------ Running Fine Tuning with arguments------")
    print(args)
    
    seed_all(args.seed) 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # cache_dir = './data/caltech256_processed/'
    # cache_dir = './data/tinyImageNet/'
    cache_dir = '/mnt/sam/pi_data/processed_data/tinyImageNet/'
    versions = [f'train_v{i}' for i in range(3)]

    if not os.path.isdir(os.path.join(cache_dir,versions[-1])):
        print("----- Processed Data not found ------")
        ds = load_dataset("zh-plus/tiny-imagenet")
        process_data(data_set= ds, data_path= cache_dir, num_versions= 3 , args=args)
    else:
        print("----- Using Cached Dataset ----- ")
    
    resize_transform = transforms.Compose([
        transforms.Resize((224,224), antialias = True)
                                           ])
    train_ds = PrecomputedDataset(cache_dir, version_folders=versions, transform=resize_transform)
    val_ds = PrecomputedDataset(cache_dir, version_folders=['val'], transform=resize_transform)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)
    

    base_model = resnet50(weights = "IMAGENET1K_V2")
    
    # model = PIFineTuneModel(base_model = base_model, image_channels = 3, num_classes = 200, device= device, args= args)
    # model.to(device)

    model = ResNetFineTune(base_model = base_model, image_channels = 3, num_classes = 200, device= device, args= args)
    model.to(device)
    model = torch.compile(model, mode="reduce-overhead")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4, fused=True)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    # model, _ = train_model(model = model,
    #                        dataloaders = {"train": train_loader, "val": val_loader}, 
    #                        criterion = criterion,
    #                        optimizer = optimizer,
    #                        args = args,
    #                        tensor_board_path = tensor_board_path)
    resume_path = None

    model, _ = train_model(model = model,
                           dataloaders = {"train": train_loader, "val": val_loader}, 
                           criterion = criterion,
                           args = args,
                           tensor_board_path = tensor_board_path,
                           resume_path=resume_path)
