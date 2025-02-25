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




#The argparser part
argparser = argparse.ArgumentParser(fromfile_prefix_chars='@')
argparser.add_argument("--lr", default=0.0003, type=float, help="Meta-learning rate (used on query set - potentially acoss tasks)")
argparser.add_argument("--seed", default=119, type=int, help="Seed to use")
argparser.add_argument("--model", default="TBR", type=str, help="Select model, avaliable models are ResNet, TR, TBR")
argparser.add_argument("--tv", default="land", type=str, help="Topological vectorization method used, methods available - check readme.txt")
argparser.add_argument("--res", default=100, type=int, help="Resolution for the Landscape vectorization method")
argparser.add_argument("--tbs", default="normal", type=str, help="Topo block size")
argparser.add_argument("--sm", default=False, action="store_true", help="Enables saving the model")
argparser.add_argument("--bw", default="cv2", type=str, help="Select the black-white transformation option")
argparser.add_argument("--topodim", default=1, type=int, help="Which dimension of the topology groups to use")
argparser.add_argument("--topodim_concat", default=False, action="store_true", help="Concatenating both dimensions of the topology features on 0 and 1 dim")
argparser.add_argument("--epochs", default=50, type=int, help="Number of epochs to train on")
argparser.add_argument("--cores", default=8, type=int, help="Number of cores to use for multiprocessing")
argparser.add_argument("--batch_size", default=64, type=int, help="Batchsize of the training")
argparser.add_argument("--val_size", default=0.2, type=float, help="Size of the validation set")
argparser.add_argument("--num_workers", default=2, type=int, help="Number of workers fo the dataloaders")
argparser.add_argument("--config", default=None, type=str, help="Path to config file, containing the Nerual architectures")
argparser.add_argument("--name", default="", type=str, help="Name of the run")
argparser.add_argument("--tb_add_x", default=False, action="store_true", help="Add the x into the topological resnet when using PIBlock")

args = argparser.parse_args()


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


#Base ResNet-18
class ResNet_18(nn.Module):

    def __init__(self, image_channels, num_classes):

        super(ResNet_18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def __make_layer(self, in_channels, out_channels, stride):

        identity_downsample = None
        if stride != 1:
            identity_downsample = self.identity_downsample(in_channels, out_channels)

        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride),
            Block(out_channels, out_channels)
        )

    def forward(self, x):

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

    def identity_downsample(self, in_channels, out_channels):

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )
