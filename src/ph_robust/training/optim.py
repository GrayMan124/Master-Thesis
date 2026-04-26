import torch.optim as optim

# from scripts.train import topo_params

# from scripts.train import backbone_params


def build_optimizer(model, cfg):
    try:
        backbone_params, topo_params = model.get_params()
        print("Using AdamW with seperate topo/backbone param groups")
        return optim.AdamW(
            [
                {"params": topo_params, "lr": cfg.optim.lr, "weight_decay": 0.05},
                {"params": backbone_params, "lr": cfg.optim.lr_b, "weight_decay": 0.01},
            ]
        )
    except Exception as e:
        print(f"Failed to build AdamW, using single-group Adam: {e}")
        active = [p for p in model.parameters() if p.requires_grad]
        return optim.Adam(
            active, lr=cfg.optim.lr, weight_decay=1e-4, fused=True, eps=1e4
        )


def build_scheduler(optimizer, cfg):
    if cfg.optim.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
    elif cfg.optim.scheduler == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
    else:
        raise ValueError(f"Unknown scheduler: {cfg.optim.scheduler}")
