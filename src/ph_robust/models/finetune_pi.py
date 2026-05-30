import torch
import torch.nn as nn

from .resnet50_attn_topo import TopoAttentionEncoder

# from .registry import GatedFusion
from .blocks import BlockSmall, GatedFusion


class TopoIMG_ResNet(
    nn.Module
):  # this is based on the resnet implementation on ResNet (using ResNet as the base to process the images)
    def __init__(self, image_channels, hidden_size, cfg):

        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # resnet layers
        if cfg.model.tbs == "small":
            self.topo_net = nn.Sequential(
                self.__make_layer(64, 64, stride=1),
                self.__make_layer(64, 128, stride=2),
            )
            self.fc = nn.Linear(128, hidden_size)

        elif cfg.model.tbs == "normal":
            self.topo_net = nn.Sequential(
                self.__make_layer(64, 64, stride=1),
                self.__make_layer(64, 128, stride=2),
                self.__make_layer(128, 256, stride=2),
                # self.__make_layer(256, 512, stride=2)
            )
            self.fc = nn.Linear(256, hidden_size)

        elif cfg.model.tbs == "large":
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
            BlockSmall(
                in_channels,
                out_channels,
                identity_downsample=identity_downsample,
                stride=stride,
            ),
            BlockSmall(out_channels, out_channels),
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


# Resnet with Topological features as topological images (in the IMG form) used for fine tuning
class PIFineTuneModel(nn.Module):
    def __init__(self, base_model, image_channels, num_classes, device, cfg):

        super().__init__()
        self.device = device

        self.base_model = base_model
        if cfg.model.freeze_weights:
            for param in self.base_model.parameters():
                param.requires_grad = False

        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(num_features, cfg.model.hidden_size)
        self.cfg = cfg

        in_channels = 2 if cfg.topo.concat else 1

        if cfg.model.ft_attn:
            self.topo_net = TopoAttentionEncoder(
                hidden_size=cfg.model.hidden_size,
                img_size=64,
                patch_size=8,
                in_channels=1,
                num_heads=2,
                depth=2,
                dropout=0.1,
            )
        else:
            self.topo_net = TopoIMG_ResNet(in_channels, cfg.model.hidden_size, cfg=cfg)

        self.fc = nn.Sequential(
            nn.Linear(cfg.model.hidden_size * 2, cfg.model.hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.model.hidden_size, num_classes),
        )
        if cfg.model.fg:
            self.fusion = GatedFusion(cfg.model.hidden_size)
        else:
            self.fusion = None

    def get_params(self):
        backbone_params = [p for p in self.base_model.parameters() if p.requires_grad]
        topo_params = list(self.topo_net.parameters()) + list(self.fc.parameters())
        if self.fusion:
            topo_params += list(self.fusion.parameters())
        return backbone_params, topo_params

    def unfreeze(self):
        if self.cfg.model.freeze_weights:
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
        if self.fusion:
            x = self.fusion(x, x_2)
        else:
            x = torch.cat([x, x_2], dim=1)
        x = self.fc(x)
        return x

    def identity_downsample(self, in_channels, out_channels):

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
        )
