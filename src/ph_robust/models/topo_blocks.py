class PIBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        inter_channels,
        out_channels,
        args,
        identity_downsample=None,
        identity_downsample_t=None,
        stride=1,
    ):
        super(PIBlock, self).__init__()
        self.args = args
        # Base ResNet Block
        self.conv1 = nn.Conv2d(
            in_channels, inter_channels, kernel_size=1, stride=stride, padding=0
        )
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(
            inter_channels, inter_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv3 = nn.Conv2d(
            inter_channels, out_channels, kernel_size=1, stride=1, padding=0
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

        self.identity_downsample_t = identity_downsample_t

        # Topo Section
        if args.tbs == "small":
            self.topo_net = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=stride, padding=1
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        elif (
            args.tbs == "normal"
        ):  # NOTE: Here we use a bottleneck design, since the channles are huge in ResNet50
            self.topo_net = nn.Sequential(
                nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(),
                nn.Conv2d(
                    inter_channels,
                    inter_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(),
                nn.Conv2d(inter_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        elif args.tbs == "large":
            self.topo_net = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=stride, padding=1
                ),
                nn.BatchNorm2d(out_channels),
                nn.Conv2d(
                    out_channels, out_channels, kernel_size=3, stride=1, padding=1
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )

    def forward(self, x):
        x, topo = x
        identity = x
        identity_t = topo
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        topo = self.topo_net(topo)

        if self.identity_downsample is not None:
            identity_t = self.identity_downsample_t(identity_t)

        x += identity

        # aligned_t = nn.functional.interpolate(identity_t, size=(x.shape[2],x.shape[3]),mode='bilinear',align_corners=False)
        x += topo

        if self.args.tb_add_t:
            topo += identity_t

        # Adding the
        # if self.args.tb_add_x:
        #     topo+=identity

        x = self.relu(x)

        return x, topo
