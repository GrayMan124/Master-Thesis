import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torchvision.transforms as transforms
import torch.nn as nn
import json

from config import args


model_saving_path = 'models/'+ args.name + args.model + '_' + args.tv + '_' + str(args.lr) + '_' + str(args.res) + '_' + str(args.seed) + '_' + str(args.topodim) + '_' + args.bw +'.pkl'
tensor_board_path = 'runs/' + args.name + args.model + '_' + args.tv + '_' + str(args.lr) + '_' + str(args.res) + '_' + str(args.seed) + '_' + str(args.topodim) + '_' + args.bw


def layer_from_config(layer_config):
    layer_type = layer_config["type"]
    params = {k: v for k, v in layer_config.items() if k != "type"}

    # Dynamically instantiate the layer
    if hasattr(nn, layer_type):
        return getattr(nn, layer_type)(**params)
    else:
        raise ValueError(f"Layer type {layer_type} is not supported.")
    
if args.config:
    with open(args.config, "r") as f:
        config = json.load(f)
    hidden_size = config['hidden_size']
else:
    hidden_size = 64


#Residual block for ResNet50
class Block(nn.Module):
    def __init__(self, in_channels, inter_channels, out_channels, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv3 = nn.Conv2d(inter_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
    
    def forward(self, x):
        identity = x
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
        x += identity
        x = self.relu(x)
        return x

#Base ResNet-50
class ResNet_50(nn.Module):

    def __init__(self, image_channels, num_classes):

        super(ResNet_50, self).__init__()
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding = 3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding = 1)

        #resnet layers
        self.layer1 = self.__make_layer(in_channels = 64, inter_channels = 64, out_channels = 256, stride=1, num_blocks= 3 )
        self.layer2 = self.__make_layer(in_channels = 256, inter_channels = 128, out_channels = 512, stride=2, num_blocks= 4 )
        self.layer3 = self.__make_layer(in_channels = 512 , inter_channels = 256, out_channels = 1024, stride=2, num_blocks= 6 )
        self.layer4 = self.__make_layer(in_channels = 1024, inter_channels = 512, out_channels = 2048, stride=2, num_blocks= 3 )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def __make_layer(self, in_channels, inter_channels, out_channels, stride, num_blocks):

        identity_downsample = None
        if stride != 1 or in_channels != out_channels:
            identity_downsample = self.identity_downsample(in_channels, out_channels, stride = stride )
        
        layers = [Block(in_channels,inter_channels, out_channels, identity_downsample=identity_downsample, stride=stride)] # Iintial block has downsampling
        for i in range(num_blocks - 1):
            layers.append(Block(in_channels = out_channels, inter_channels = inter_channels, out_channels = out_channels))
        

        return nn.Sequential(*layers)         

    def forward(self, x):

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
        x = self.fc(x)
        return x

    def identity_downsample(self, in_channels, out_channels, stride):

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
            # nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),# padding=1),
            nn.BatchNorm2d(out_channels)
        )
