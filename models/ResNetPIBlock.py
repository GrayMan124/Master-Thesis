#Update, version from: 09_02_2025
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pickle
import gudhi as gd
import gudhi.representations
import argparse
from matplotlib import pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import cv2
from PIL import Image
from torchvision.transforms import v2
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, set_start_method
import time
from torch.utils.tensorboard import SummaryWriter
import copy
import json

from config import args

model_saving_path = 'models/'+ args.name + args.model + '_' + args.tv + '_' + str(args.lr) + '_' + str(args.res) + '_' + str(args.seed) + '_' + str(args.topodim) + '_' + args.bw +'.pkl'
tensor_board_path = 'runs/' + args.name + args.model + '_' + args.tv + '_' + str(args.lr) + '_' + str(args.res) + '_' + str(args.seed) + '_' + str(args.topodim) + '_' + args.bw

if args.config:
    with open(args.config, "r") as f:
        config = json.load(f)
    hidden_size = config['hidden_size']
else:
    hidden_size = 64
    

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
    

class LayerPIBlock(nn.Module):

    def __init__(self,in_channels, out_channels, identity_downsample, stride):
        super(LayerPIBlock,self).__init__()
        self.block1 = PIBlock(in_channels, out_channels,identity_downsample, stride=stride)
        self.block2 = PIBlock(out_channels, out_channels)

    def forward(self,x):
        x,topo = x
        x,topo= self.block1((x,topo))
        x,topo = self.block2((x,topo))
        return x, topo

class PIBlock(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, stride=1):
        super(PIBlock, self).__init__()

        #Base ResNet Block
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

        #Topo Section
        if args.tbs == 'small':
            self.topo_net = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU()    
                                    )
        elif args.tbs == 'normal':
            self.topo_net = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                                nn.BatchNorm2d(out_channels),
                                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU()    
                                    )
        elif args.tbs == 'large':
            self.topo_net = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                                nn.BatchNorm2d(out_channels),
                                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU()    
                                    )
        
        self.identity_downsample_t = identity_downsample



    def forward(self, x):
        x,topo = x
        identity = x
        identity_t = topo
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        topo = self.topo_net(topo)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)


        if self.identity_downsample is not None:
            identity_t = self.identity_downsample_t(identity_t)

        x += identity
        x += identity_t

        if args.tb_add_t:
            topo += identity_t

        #Adding the
        if args.tb_add_x:
            topo+=identity

        x = self.relu(x)
        return x, topo

class TopoIMG_transModel(nn.Module): #This model is specificaly designed to transform the input of 1x64x64 into 3x32x32 (usable in topoblock configugartion)

    def __init__(self,tmp): #why is tmp here KEKW
        super(TopoIMG_transModel,self).__init__()

        #This implementation gives as output 3x32x32 (given an input of 1x64x64)

        #TODO: Add some options here as well 
        if args.topodim_concat:
            in_ch = 2
        else:
            in_ch = 1
        

        #TODO: Check if this actually works TBH KEKW
        
        if args.tbs == 'small':
            self.conv_network = nn.Sequential(
                nn.Conv2d(in_channels = in_ch ,out_channels = 16, kernel_size = (5, 5), stride=2,padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels = 16 ,out_channels = 32, kernel_size = (4, 4), stride=1,padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels = 32 ,out_channels = 64, kernel_size = (5, 5), stride=1,padding=1),
                nn.ReLU()
            )    

        elif args.tbs == 'normal':
            self.conv_network = nn.Sequential(
                nn.Conv2d(in_channels = in_ch ,out_channels = 28, kernel_size = (5, 5), stride=2,padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels = 28 ,out_channels = 48, kernel_size = (4, 4), stride=1,padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels = 48 ,out_channels = 64, kernel_size = (5, 5), stride=1,padding=1),
                nn.ReLU()
            )

        elif args.tbs == 'large':
            self.conv_network = nn.Sequential(
                nn.Conv2d(in_channels = in_ch ,out_channels = 32, kernel_size = (5, 5), stride=2,padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels = 32 ,out_channels = 48, kernel_size = (4, 4), stride=1,padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels = 48 ,out_channels = 64, kernel_size = (4, 4), stride=1,padding=1),
                nn.ReLU(),
                nn.Conv2d(in_channels = 64 ,out_channels = 64, kernel_size = (4, 4), stride=1,padding=1),
                nn.ReLU()
            )

    def forward(self,x):
        return self.conv_network(x)

class ResNet_18_PIBlock(nn.Module):

    def __init__(self, image_channels, num_classes,device):

        super(ResNet_18_PIBlock, self).__init__()
        self.in_channels = 64
        self.device = device

        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #topological section
        self.topo_emb = TopoIMG_transModel(0)
        self.conv1_t = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1_t = nn.BatchNorm2d(64)
        self.relu_t = nn.ReLU()
        self.maxpool_t = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool_t = nn.AdaptiveAvgPool2d((1, 1))

        if args.config:
            layers = [layer_from_config(layer_config) for layer_config in config["res_net_fc"]]
            self.res_net_fc = nn.Sequential(*layers)
        else:
            self.res_net_fc = nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128,64),
                nn.ReLU()
            )

        #Setup for the res_net_fc_topo
        if args.config:
            layers = [layer_from_config(layer_config) for layer_config in config["res_net_fc_topo"]]
            self.res_net_fc_topo = nn.Sequential(*layers)
        else:
            self.res_net_fc_topo = nn.Sequential(
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128,64),
                nn.ReLU()
            )
        if args.config:
            layers = [layer_from_config(layer_config) for layer_config in config["fc"]]
            self.fc = nn.Sequential(*layers)
        else:
            self.fc = nn.Sequential(nn.Linear(128,64),
                nn.ReLU(),
                nn.Linear(64,32),
                nn.ReLU(),
                nn.Linear(32,num_classes),
                nn.Softmax()
            )

    def __make_layer(self, in_channels, out_channels, stride):

        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)


        return LayerPIBlock(in_channels, out_channels, identity_downsample, stride)


    def forward(self, x):
        #suppose we do have the topo info in the dataset

        x, topo = x

        x = x.to(self.device)
        topo = topo.to(self.device)

        x = transforms.functional.resize(x, (112, 112))
        x_topo = self.topo_emb(topo)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x,x_topo = self.layer1((x,x_topo))

        x,x_topo = self.layer2((x,x_topo))
        x,x_topo = self.layer3((x,x_topo))
        x,x_topo = self.layer4((x,x_topo))

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.res_net_fc(x)

        x_topo = self.avgpool_t(x_topo)
        x_topo = x_topo.view(x_topo.shape[0], -1)
        x_topo = self.res_net_fc_topo(x_topo)

        x = torch.cat([x,x_topo],dim=1)

        x = self.fc(x)

        return x

    def identity_downsample(self, in_channels, out_channels):

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )
