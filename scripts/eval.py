import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.config_store import ConfigStore
import torch.nn as nn


from ph_robust.data_processing.loaders import build_test_loader
from ph_robust.models.registry import build_model
from ph_robust.training.eval import test_model
from ph_robust.training.seed import seed_all
from ph_robust.conf.schema import Config


torch.autograd.set_detect_anomaly(False)

cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:

    print("------ Running Fine Tests with arguments------")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(OmegaConf.to_yaml(cfg))

    seed_all(cfg.seed)

    data_path = cfg.data.data_path
    print(f"Looking for data in {data_path}")
    test_loader = build_test_loader(cfg=cfg)

    state_dict = torch.load(f"./saveModels/{cfg.run_name}.pkl", weights_only=True)
    model = build_model(cfg=cfg, device=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    criterion = nn.CrossEntropyLoss()
    # run = wandb.init(
    #     project = "ph-robust-img",
    #     id = args.run_id,
    #     resume = "must"
    # )

    loss, top1, top5 = test_model(
        model=model, dataloader=test_loader, criterion=criterion, cfg=cfg
    )
    print(f"""--------- TEST RESULTS -----------
    Tested model : {cfg.run_name}
    Test Acc Top1: {top1}
    Test Acc Top5: {top5}
    Test Loss: {loss}
    """)
    # run.summary.update({
    #     "test/top1": top1,
    #     "test/top5" :  top5,
    #     "test/loss": loss })
    # wandb.finish()


if __name__ == "__main__":
    main()
