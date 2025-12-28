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




#Residual block
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x


#Resnet with Topological features as topological images (in the IMG form)
class ResNetFineTune(nn.Module):

    def __init__(self, base_model, image_channels, num_classes, device, args):

        super(ResNetFineTune, self).__init__()
        self.device = device

        self.base_model = base_model
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
                nn.Linear(args.hidden_size,num_classes ),
                nn.Softmax()
            )


    def forward(self, x):
        #suppose we do have the topo info in the dataset
        x, _= x
        x = x.to('cuda:0')
        # topo = topo.to('cuda:0')
        x = torch.nn.functional.interpolate(x, size= (224,224), mode = 'bilinear', align_corners= False)
        x = self.base_model(x)
        # x = torch.cat([x,x_2],dim=1)
        x = self.fc(x)
        return x

    def identity_downsample(self, in_channels, out_channels):

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )
