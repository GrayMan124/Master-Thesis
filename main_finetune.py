# Update, version from: 22-12-2025
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50
from torchvision import transforms
from datasets import load_dataset

from config.config import args
from dataProcessing.processing import process_data, PrecomputedDataset
from utils import train_model, seed_all, count_parameters
from models.PI_finetune import PIFineTuneModel
from models.FineTuneResNet import ResNetFineTune
from models.ReNet50_Topo import PH_ResNet50
from models.ResNet50_AttTopo import ResNet_AttnTopo

torch.autograd.set_detect_anomaly(False)

if __name__ == "__main__":
    print("------ Running Fine Tuning with arguments------")
    print(args)

    seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    data_path = args.data_path
    if args.maxNorm:
        data_path = data_path + "mn"
    versions = [f"train_v{i}" for i in range(10)]

    if not os.path.isdir(os.path.join(data_path, versions[-1])):
        print("----- Processed Data not found ------")
        ds = load_dataset("zh-plus/tiny-imagenet")
        process_data(data_set=ds, data_path=data_path, num_versions=10, args=args)
    else:
        print("----- Using Cached Dataset ----- ")

    resize_transform = transforms.Compose(
        [transforms.Resize((224, 224), antialias=True)]
    )
    train_ds = PrecomputedDataset(
        data_path, version_folders=versions, transform=resize_transform
    )
    val_ds = PrecomputedDataset(
        data_path, version_folders=["val"], transform=resize_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    base_model = resnet50(weights="IMAGENET1K_V2")
    if args.modelFT == "PI_IMG":
        model = PIFineTuneModel(
            base_model=base_model,
            image_channels=3,
            num_classes=200,
            device=device,
            args=args,
        )
    elif args.modelFT == "ResNet50":
        model = ResNetFineTune(
            base_model=base_model,
            image_channels=3,
            num_classes=200,
            device=device,
            args=args,
        )
    elif args.modelFT == "RN50_S":
        model = PH_ResNet50(image_channels=3, num_classes=200, args=args)
    elif args.modelFT == "RN50_Atn":
        model = ResNet_AttnTopo(image_channels=3, num_classes=200, args=args)
    else:
        raise Exception(f"Unrecognized modelFT argument: {args.modelFT}")
    model.to(device)
    model = torch.compile(model, mode="reduce-overhead")
    count_parameters(model)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4, fused=True, eps=1e-4)
    try:
        backbone_params, topo_params = model.get_params()
        print("Using AdamW optimizer")
        optimizer = optim.AdamW(
            [
                {"params": topo_params, "lr": 3e-4, "weight_decay": 0.05},
                {"params": backbone_params, "lr": 3e-6, "weight_decay": 0.01},
            ]
        )
    except Exception as e:
        print(f"failed to retrieve topo and backbone paramters with error {e}")
        active_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.Adam(
            active_params, lr=args.lr, weight_decay=1e-4, fused=True, eps=1e-4
        )

    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    resume_path = None

    model, _ = train_model(
        model=model,
        dataloaders={"train": train_loader, "val": val_loader},
        criterion=criterion,
        args=args,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        resume_path=resume_path,
    )
    if args.sm:
        print("savingModel")
        torch.save(model.state_dict(), f"./saveModels/{args.name}.pkl")
