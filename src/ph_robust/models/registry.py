from uu import Error
import torch.nn as nn
import torch
from torchvision.models import resnet50

from .finetune_pi import PIFineTuneModel
from .finetune_resnet import ResNetFineTune
from .ph_resnet50 import PH_ResNet50
from .resnet50_attn_topo import ResNet_AttnTopo

_REGISTRY = {
    "PI_IMG": lambda cfg, device: PIFineTuneModel(
        base_model=resnet50(weights="IMAGENET1K_V2"),
        image_channels=3,
        num_classes=cfg.data.num_classes,
        device=device,
        args=cfg,
    ),
    "ResNet50": lambda cfg, device: ResNetFineTune(
        base_model=resnet50(weights="IMAGENET1K_V2"),
        image_channels=3,
        num_classes=cfg.data.num_classes,
        device=device,
        args=cfg,
    ),
    "RN50_S": lambda cfg, device: PH_ResNet50(
        image_channels=3,
        num_classes=cfg.data.num_classes,
        args=cfg,
    ),
    "RN50_Atn": lambda cfg, device: ResNet_AttnTopo(
        image_channels=3,
        num_classes=cfg.data.num_classes,
        args=cfg,
    ),
}


def build_model(cfg, device):
    if cfg.model.kind not in _REGISTRY:
        raise Error(f"Uknown model {cfg.model.kind}")
    model = _REGISTRY[cfg.model.kind](cfg, device)
    model.to(device)
    if cfg.train.compile:
        model = torch.compile(model)
    return model


def layer_from_config(layer_config):
    layer_type = layer_config["type"]
    params = {k: v for k, v in layer_config.items() if k != "type"}

    # Dynamically instantiate the layer
    if hasattr(nn, layer_type):
        return getattr(nn, layer_type)(**params)
    else:
        raise ValueError(f"Layer type {layer_type} is not supported.")


class GatedFusion(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.gate = nn.Parameter(torch.tensor(0.5))

    def forward(self, img_feat, topo_feat):
        g = torch.sigmoid(self.gate)
        return torch.cat([g * img_feat, (1 - g) * topo_feat], dim=1)
