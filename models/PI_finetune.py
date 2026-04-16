import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from models.ResNet import hidden_size
import torch
import torch.nn as nn
import json

from ResNet50_AttTopo import TopoAttentionEncoder


def layer_from_config(layer_config):
    layer_type = layer_config["type"]
    params = {k: v for k, v in layer_config.items() if k != "type"}
    # Dynamically instantiate the layer
    if hasattr(nn, layer_type):
        return getattr(nn, layer_type)(**params)
    else:
        raise ValueError(f"Layer type {layer_type} is not supported.")


# Residual block
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
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


class TopoIMG_ResNet(
    nn.Module
):  # this is based on the resnet implementation on ResNet (using ResNet as the base to process the images)
    def __init__(self, image_channels, hidden_size, args):

        super(TopoIMG_ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # resnet layers
        if args.tbs == "small":
            self.topo_net = nn.Sequential(
                self.__make_layer(64, 64, stride=1),
                self.__make_layer(64, 128, stride=2),
            )
            self.fc = nn.Linear(128, hidden_size)

        elif args.tbs == "normal":
            self.topo_net = nn.Sequential(
                self.__make_layer(64, 64, stride=1),
                self.__make_layer(64, 128, stride=2),
                self.__make_layer(128, 256, stride=2),
                # self.__make_layer(256, 512, stride=2)
            )
            self.fc = nn.Linear(256, hidden_size)

        elif args.tbs == "large":
            self.topo_net = nn.Sequential(
                self.__make_layer(64, 64, stride=1),
                self.__make_layer(64, 128, stride=2),
                self.__make_layer(128, 256, stride=2),
                self.__make_layer(256, 512, stride=2),
            )
            self.fc = nn.Linear(512, hidden_size)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def __make_layer(self, in_channels, out_channels, stride):

        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)

        return nn.Sequential(
            Block(
                in_channels,
                out_channels,
                identity_downsample=identity_downsample,
                stride=stride,
            ),
            Block(out_channels, out_channels),
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.topo_net(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def identity_downsample(self, in_channels, out_channels):

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
        )


# Resnet with Topological features as topological images (in the IMG form)
class PIFineTuneModel(nn.Module):
    def __init__(self, base_model, image_channels, num_classes, device, args):

        super(PIFineTuneModel, self).__init__()
        self.device = device

        self.base_model = base_model
        if args.freeze_weights:
            for param in self.base_model.parameters():
                param.requires_grad = False

        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_features, args.hidden_size)
        self.args = args

        in_channels = 2 if args.topodim_concat else 1

        if args.ft_attn:
            self.topo_net = TopoAttentionEncoder(
                hidden_size=args.hidden_size,
                img_size=64,
                patch_size=8,
                in_channels=1,
                num_heads=2,
                depth=2,
                dropout=0.1,
            )
        else:
            self.topo_net = TopoIMG_ResNet(in_channels, args.hidden_size, args=args)
        if args.config:
            layers = [
                layer_from_config(layer_config) for layer_config in args.config["fc"]
            ]
            self.fc = nn.Sequential(*layers)
        else:
            self.fc = nn.Sequential(
                nn.Linear(args.hidden_size * 2, args.hidden_size),
                nn.ReLU(inplace=True),
                nn.Linear(args.hidden_size, num_classes),
            )

    def unfreeze(self):
        if self.args.freeze_weights:
            for param in self.base_model.parameters():
                param.requires_grad = True

    def forward(self, x):
        # suppose we do have the topo info in the dataset
        x, topo = x
        x = torch.nn.functional.interpolate(
            x, size=(224, 224), mode="bilinear", align_corners=False
        )
        x = self.base_model(x)
        x_2 = self.topo_net(topo)
        x_2 = x_2.squeeze(1)
        x = torch.cat([x, x_2], dim=1)
        x = self.fc(x)
        return x

    def identity_downsample(self, in_channels, out_channels):

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
        )
