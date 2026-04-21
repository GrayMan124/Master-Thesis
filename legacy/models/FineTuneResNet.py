import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import json


def layer_from_config(layer_config):
    layer_type = layer_config["type"]
    params = {k: v for k, v in layer_config.items() if k != "type"}
    # Dynamically instantiate the layer
    if hasattr(nn, layer_type):
        return getattr(nn, layer_type)(**params)
    else:
        raise ValueError(f"Layer type {layer_type} is not supported.")


#Resnet with Topological features as topological images (in the IMG form)
class ResNetFineTune(nn.Module):

    def __init__(self, base_model, image_channels, num_classes, device, args):

        super(ResNetFineTune, self).__init__()
        self.device = device

        self.base_model = base_model
        for _, module in self.base_model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eps = 1e-4
        if args.freeze_weights:
            for param in self.base_model.parameters():
                param.requires_grad = False

        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_features, args.hidden_size)

        if args.config: #if the config was set
            raise('This part is not yet inplemented in the config - TopoPI')
            layers = [layer_from_config(layer_config) for layer_config in config["topo_net"]]
            self.topo_net = nn.Sequential(*layers)

        
        if args.config:
            layers = [layer_from_config(layer_config) for layer_config in args.config["fc"]]
            self.fc = nn.Sequential(*layers)
        else:
            self.fc = nn.Sequential(nn.Linear(args.hidden_size, args.hidden_size),
                nn.ReLU(),
                nn.Linear(args.hidden_size,num_classes )
            )


    def forward(self, x):

        x, _= x
        # x = torch.nn.functional.interpolate(x, size= (224,224), mode = 'bilinear', align_corners= False)
        x = self.base_model(x)
        x = self.fc(x)
        return x

    def identity_downsample(self, in_channels, out_channels):

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )
