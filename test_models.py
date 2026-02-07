import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
import wandb
from torchvision.models import resnet50 
from torchvision import transforms 
from datasets import load_dataset

from config.config import args
from dataProcessing.processing import PrecomputedDataset, process_test
from utils import test_model, train_model, seed_all
from models.PI_finetune import PIFineTuneModel
from models.FineTuneResNet import ResNetFineTune


if __name__ == '__main__':
    print("------ Running Fine Tuning with arguments------")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    print(args)
    
    seed_all(args.seed) 

    data_path = args.data_path

    if not os.path.isdir(os.path.join(data_path,'test')):
        print("----- Processed TEST Data not found ------")
        ds = load_dataset("zh-plus/tiny-imagenet")
        process_test(data_set= ds, data_path= data_path,  args=args)
    else:
        print("----- Using Cached Dataset ----- ")
    
    resize_transform = transforms.Compose([
        transforms.Resize((224,224), antialias = True)
                                           ])
    test_ds = PrecomputedDataset(data_path, version_folders=['test'], transform=resize_transform)
    test_loader = DataLoader(test_ds, batch_size= args.batch_size, shuffle=False, num_workers= args.num_workers, pin_memory=True)
    

    base_model = resnet50(weights = "IMAGENET1K_V2")
    if args.modelFT == 'PI_IMG': 
        model = PIFineTuneModel(base_model = base_model, image_channels = 3, num_classes = 200, device= device, args= args)
        model.to(device)
    elif args.modelFT == "ResNet50":
        model = ResNetFineTune(base_model = base_model, image_channels = 3, num_classes = 200, device= device, args= args)
        model.to(device)
    else:
        raise Exception(f"Unrecognized modelFT argument: {args.modelFT}")
   
    model.load_state_dict(torch.load(os.path.join(args.modelPath,f"{args.name}.pkl"),weights_only=True))
    criterion = nn.CrossEntropyLoss()
    wandb.init(
        project = "ph-robust-img",
        id = args.run_id,
        resume = "must"
    )

    loss, top1, top5 = test_model(model = model, dataloader = test_loader , criterion = criterion)
    wandb.run.summary["test/top1"] = top1
    wandb.run.summary["test/top5"] = top5
    wandb.run.summary["test/loss"] = loss 
    wandb.finish()
