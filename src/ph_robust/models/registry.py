import torch.nn as nn
import torch


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
