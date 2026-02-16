import functools
import torch
import numpy as np
from torch.utils.data import DataLoader 

from config.config import args
from dataProcessing.processing import process_data, PrecomputedDataset
from dataProcessing.topology.topologicalProcessing import process_PI 
from models.FineTuneResNet import ResNetFineTune
from models.PI_finetune import PIFineTuneModel
from benchmarks.benchamrk_utils import ModelWrapper, run_auto_attack,CleanImageDatasetLoader 
from utils import seed_all
import os
from tqdm import tqdm 

from torchvision.models import resnet50 
from torchvision import transforms 
from datasets import load_dataset

from dataProcessing.processing import PrecomputedDataset, process_test

if __name__ == '__main__':
    print("------ Running Adversarial attacks test------")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')
    print(args)
    
    seed_all(args.seed) 

    data_path = args.data_path
    print(f"Looking for data in {data_path}")
    if not os.path.isdir(os.path.join(data_path,'test')):
        print("----- Processed TEST Data not found ------")
        ds = load_dataset("zh-plus/tiny-imagenet",split='valid')
        process_test(data_set= ds, data_path= data_path,  args=args)
    else:
        print("----- Using Cached Dataset ----- ")
    
    resize_transform = transforms.Compose([
        transforms.Resize((224,224), antialias = True)
                                           ])
    test_ds = PrecomputedDataset(data_path, version_folders=['test'], transform=resize_transform)
    test_loader = DataLoader(test_ds, batch_size= args.batch_size, shuffle=False, num_workers= args.num_workers, pin_memory=True)
    
    all_data = []
    all_labels = []
    
    for (x,_), label in tqdm(test_loader):
        all_data.append(x)
        all_labels.append(label)

    x_test = torch.cat(all_data,dim=0)
    y_test = torch.cat(all_labels, dim = 0)
    base_model = resnet50(weights = "IMAGENET1K_V2")
    if args.modelFT == 'PI_IMG': 
        model = PIFineTuneModel(base_model = base_model, image_channels = 3, num_classes = 200, device= device, args= args)
        model.to(device)
    elif args.modelFT == "ResNet50":
        model = ResNetFineTune(base_model = base_model, image_channels = 3, num_classes = 200, device= device, args= args)
        model.to(device)
    else:
        raise Exception(f"Unrecognized modelFT argument: {args.modelFT}")
    # model.compile() 
    state_dict = torch.load(os.path.join(args.modelPath, f"{args.name}.pkl"), weights_only=True)

    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    # criterion = nn.CrossEntropyLoss()
    # run = wandb.init(
    #     project = "ph-robust-img",
    #     id = args.run_id,
    #     resume = "must"
    # )
    topo_func_partial = functools.partial(process_PI, args=args)
    wrapped_model = ModelWrapper(
            model = model,
            topo_func=topo_func_partial,
            device = device
    )

    run_auto_attack(model=wrapped_model, x_test= None, y_test=None, log_path='./')

