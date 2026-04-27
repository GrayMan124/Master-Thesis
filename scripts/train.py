# Update, version from: 27-04-2026
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
import torch.nn as nn


from ph_robust.data_processing.loaders import build_dataloaders
from ph_robust.models.registry import build_model
from ph_robust.training.optim import build_optimizer, build_scheduler
from ph_robust.training.train import train_model
from ph_robust.training.seed import seed_all
from ph_robust.training.utils import count_parameters
from ph_robust.conf.schema import Config


torch.autograd.set_detect_anomaly(False)

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    print("------ Running Fine Tuning with arguments------")
    print(OmegaConf.to_yaml(cfg))

    seed_all(cfg.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader = build_dataloaders(cfg)

    model = build_model(cfg, device)
    count_parameters(model)

    optimizer = build_optimizer(model, cfg)
    lr_scheduler = build_scheduler(optimizer, cfg)
    criterion = nn.CrossEntropyLoss()

    resume_path = None

    model, _ = train_model(
        model=model,
        dataloaders={"train": train_loader, "val": val_loader},
        criterion=criterion,
        args=cfg,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        resume_path=resume_path,
    )
    if cfg.train.save_model:
        print("savingModel")
        torch.save(model.state_dict(), f"./saveModels/{cfg.run_name}.pkl")
