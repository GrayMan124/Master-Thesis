import torch

import torch.nn as nn
from .blocks import Block


class TopoAttentionEncoder(nn.Module):
    def __init__(
        self,
        hidden_size,
        img_size=64,
        patch_size=8,
        in_channels=1,
        num_heads=4,
        depth=3,
        dropout=0.0,
    ) -> None:
        super().__init__()
        num_patches = (img_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        self.patch_size = patch_size
        self.patch_embed = nn.Linear(patch_dim, hidden_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, hidden_size))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):

        B = x.shape[0]
        p = self.patch_size

        x = x.unfold(2, p, p).unfold(3, p, p)
        x = x.contiguous().view(B, -1, x.shape[1] * p * p)

        x = self.patch_embed(x)

        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.dropout(x)

        x = self.transformer(x)
        x = self.norm(x[:, 0])
        return x


# Base ResNet-50
class ResNet_AttnTopo(nn.Module):
    def __init__(self, image_channels, num_classes, cfg):

        super(ResNet_AttnTopo, self).__init__()
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # resnet layers
        self.layer1 = self.__make_layer(
            in_channels=64, inter_channels=64, out_channels=256, stride=1, num_blocks=3
        )
        self.layer2 = self.__make_layer(
            in_channels=256,
            inter_channels=128,
            out_channels=512,
            stride=2,
            num_blocks=4,
        )
        self.layer3 = self.__make_layer(
            in_channels=512,
            inter_channels=256,
            out_channels=1024,
            stride=2,
            num_blocks=6,
        )
        self.layer4 = self.__make_layer(
            in_channels=1024,
            inter_channels=512,
            out_channels=2048,
            stride=2,
            num_blocks=3,
        )
        self.topo_net = TopoAttentionEncoder(
            hidden_size=cfg.model.hidden_size,
            img_size=64,
            patch_size=8,
            in_channels=2 if cfg.topodim_concat else 1,
            num_heads=4,
            depth=2,
            dropout=0.2,
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_img = nn.Sequential(nn.Linear(2048, cfg.model.hidden_size), nn.ReLU())
        self.fc = nn.Linear(2 * cfg.model.hidden_size, num_classes)

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
                in_channels,
                inter_channels,
                out_channels,
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

        return nn.Sequential(*layers)

    def forward(self, x):
        x, topo = x
        # x = transforms.functional.resize(x, (112, 112))
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
        x = self.fc_img(x)
        topo = self.topo_net(topo)

        x = torch.cat([x, topo], dim=1)

        x = self.fc(x)
        return x

    def identity_downsample(self, in_channels, out_channels, stride):

        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, padding=0
            ),
            nn.BatchNorm2d(out_channels),
        )

    def unfreeze(self):
        pass
