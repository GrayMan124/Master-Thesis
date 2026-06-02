import torch

import torch.nn as nn
from .blocks import Block
from .pi_trans_model import TopoIMG_transModel


class TupleSequential(nn.Sequential):
    def forward(self, input):
        for module in self:
            input = module(input)
        return input


class PH_TEN(nn.Module):
    def __init__(self, image_channels, cfg, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.topo_embed = TopoIMG_transModel(cfg=cfg)
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.__make_layer(
            in_channels=64, inter_channels=64, out_channels=256, stride=1, num_blocks=2
        )
        self.layer2 = self.__make_layer(
            in_channels=256,
            inter_channels=128,
            out_channels=512,
            stride=2,
            num_blocks=2,  # NOTE: This is standard
        )
        self.layer3 = self.__make_layer(
            in_channels=512,
            inter_channels=256,
            out_channels=1024,
            stride=2,
            num_blocks=2,  # NOTE: This is the standard
        )
        self.layer4 = self.__make_layer(
            in_channels=1024,
            inter_channels=512,
            out_channels=2048,
            stride=2,
            num_blocks=2,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_t = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(nn.Linear(2048, cfg.model.hidden_size), nn.ReLU())

    def __make_layer(
        self, in_channels, inter_channels, out_channels, stride, num_blocks
    ):

        identity_downsample = None
        if stride != 1 or in_channels != out_channels:
            identity_downsample = self.identity_downsample(
                in_channels, out_channels, stride=stride
            )

        layers = [
            Block(
                in_channels=in_channels,
                inter_channels=inter_channels,
                out_channels=out_channels,
                identity_downsample=identity_downsample,
                stride=stride,
            )
        ]  # Iintial block has downsampling
        for _ in range(num_blocks - 1):
            layers.append(
                Block(
                    in_channels=out_channels,
                    inter_channels=inter_channels,
                    out_channels=out_channels,
                )
            )

        # return nn.Sequential(*layers)
        return TupleSequential(*layers)

    def forward(self, x):

        # x = transforms.functional.resize(x, (112, 112))
        x = self.topo_embed(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)

        x = self.fc(x)

        return x

    def unfreeze(self):
        pass

    def identity_downsample(self, in_channels, out_channels, stride):

        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, padding=0
            ),
            nn.BatchNorm2d(out_channels),
        )


class PH_TRN_RN50(nn.Module):
    def __init__(self, image_channels, num_classes, cfg):

        super().__init__()
        self.cfg = cfg
        self.ten = PH_TEN(image_channels=image_channels, cfg=cfg)
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.maxpool_t = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # resnet layers
        self.layer1 = self.__make_layer(
            in_channels=64, inter_channels=64, out_channels=256, stride=1, num_blocks=3
        )
        self.layer2 = self.__make_layer(
            in_channels=256,
            inter_channels=128,
            out_channels=512,
            stride=2,
            num_blocks=4,  # NOTE: This is standard
        )
        self.layer3 = self.__make_layer(
            in_channels=512,
            inter_channels=256,
            out_channels=1024,
            stride=2,
            num_blocks=6,  # NOTE: This is the standard
        )
        self.layer4 = self.__make_layer(
            in_channels=1024,
            inter_channels=512,
            out_channels=2048,
            stride=2,
            num_blocks=3,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.res_net_fc = nn.Sequential(
            nn.Linear(2048, cfg.model.hidden_size), nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(cfg.model.hidden_size * 2, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def __make_layer(
        self, in_channels, inter_channels, out_channels, stride, num_blocks
    ):

        identity_downsample = None
        if stride != 1 or in_channels != out_channels:
            identity_downsample = self.identity_downsample(
                in_channels, out_channels, stride=stride
            )

        layers = [
            Block(
                in_channels=in_channels,
                inter_channels=inter_channels,
                out_channels=out_channels,
                identity_downsample=identity_downsample,
                stride=stride,
            )
        ]  # Iintial block has downsampling
        for _ in range(num_blocks - 1):
            layers.append(
                Block(
                    in_channels=out_channels,
                    inter_channels=inter_channels,
                    out_channels=out_channels,
                )
            )

        # return nn.Sequential(*layers)
        return TupleSequential(*layers)

    def forward(self, x):

        # x = transforms.functional.resize(x, (112, 112))
        x, topo = x
        topo = self.ten(topo)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.res_net_fc(x)

        x = torch.cat([x, topo], dim=1)

        x = self.fc(x)

        return x

    def unfreeze(self):
        pass

    def identity_downsample(self, in_channels, out_channels, stride):

        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, padding=0
            ),
            nn.BatchNorm2d(out_channels),
        )
