import torch.nn as nn


class TopoIMG_transModel(
    nn.Module
):  # This model is specificaly designed to transform the input of 1x64x64 into 3x32x32 (usable in topoblock configugartion)
    def __init__(self, cfg):
        super().__init__()
        # NOTE: For this implementation, I could stick with the base image size 1x64x64 since the overall images are the same size, but let's keep it for now
        if cfg.topo.concat:
            in_ch = 2
        else:
            in_ch = 1

        if cfg.model.tbs == "small":
            self.conv_network = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=64,
                    kernel_size=7,
                    stride=2,
                    padding=3,
                ),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )

        elif cfg.model.tbs == "normal":
            self.conv_network = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=32,
                    out_channels=64,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )

        elif cfg.model.tbs == "large":
            self.conv_network = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=32,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=32,
                    out_channels=48,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                ),
                nn.BatchNorm2d(48),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels=48,
                    out_channels=64,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(64),
                nn.ReLU(),
            )

    def forward(self, x):
        return self.conv_network(x)
