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


torch.autograd.set_detect_anomaly(True)

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


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


torch.manual_seed(args.seed)


model_saving_path = 'models/'+ args.name + args.model + '_' + args.tv + '_' + str(args.lr) + '_' + str(args.res) + '_' + str(args.seed) + '_' + str(args.topodim) + '_' + args.bw +'.pkl'
tensor_board_path = 'runs/' + args.name + args.model + '_' + args.tv + '_' + str(args.lr) + '_' + str(args.res) + '_' + str(args.seed) + '_' + str(args.topodim) + '_' + args.bw


base_dataset_path = 'data/'
train_set_path = base_dataset_path + 'train_set_' + args.tv + '_' + str(args.res) + '_' + str(args.topodim) + '_' + str(args.topodim_concat) +  '_' + args.bw +'.pkl'
train_set_target_path = base_dataset_path + 'train_set_target_' + args.tv + '_' + str(args.res) + '_' + str(args.topodim)+ '_' + str(args.topodim_concat) + '_' + args.bw  +'.pkl'

test_set_path = base_dataset_path + 'test_set_' + args.tv + '_' + str(args.res) + '_' + str(args.topodim) + '_' + str(args.topodim_concat) + '_' + args.bw + '.pkl'
test_set_target_path = base_dataset_path + 'test_set_target_' + args.tv + '_' + str(args.res) + '_' + str(args.topodim) + '_' +  str(args.topodim_concat) + '_' + args.bw  + '.pkl'


result_file = 'results.txt'

writer = SummaryWriter(log_dir = tensor_board_path)

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


class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Return a tuple (data, label)
        return self.data[idx], self.labels[idx]

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
        self.conv1_t = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1_t = nn.BatchNorm2d(out_channels)
        self.conv2_t = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2_t = nn.BatchNorm2d(out_channels)
        self.relu_t = nn.ReLU()
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

        #Topological information
        topo = self.conv1_t(topo)
        topo = self.relu_t(topo)
        topo = self.bn1_t(topo)
        topo = self.conv2_t(topo)
        topo = self.bn2_t(topo)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)


        if self.identity_downsample is not None:
            identity_t = self.identity_downsample_t(identity_t)

        x += identity
        x += identity_t


        topo += identity_t

        #Adding the
        if args.tb_add_x:
            topo+=identity

        x = self.relu(x)
        return x, topo



class LayerTopoBlock(nn.Module):

    def __init__(self,in_channels, out_channels, identity_downsample, stride):
        super(LayerTopoBlock,self).__init__()
        self.block1 = TopoBlock(in_channels, out_channels,identity_downsample, hidden_size, stride=stride)
        self.block2 = TopoBlock(out_channels, out_channels)

    def forward(self,x):
        x,topo = x
        x,_ = self.block1((x,topo))
        x,_ = self.block2((x,topo))
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

#Residual block

class TopoBlock(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, topo_size_in = hidden_size, stride=1):
        super(TopoBlock, self).__init__()

        if args.tbs == 'small':
            self.topo_enc = nn.Sequential(nn.Linear(topo_size_in,64),
                nn.ReLU(),
                nn.Linear(64,out_channels)
                )
        elif args.tbs == 'normal':
            self.topo_enc = nn.Sequential(nn.Linear(topo_size_in,64),
                nn.ReLU(),
                nn.Linear(64,32),
                nn.ReLU(),
                nn.Linear(32,out_channels)
                )
        elif args.tbs == 'large':
            self.topo_enc = nn.Sequential(nn.Linear(topo_size_in,128),
                nn.ReLU(),
                nn.Linear(128,64),
                nn.ReLU(),
                nn.Linear(64,out_channels)
                )

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self, x):

        x, topo = x
        identity = x
        topo_in = self.topo_enc(topo).squeeze(1)

        topo_expand =  topo_in[:, :, None, None]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x += topo_expand
        x = self.relu(x)
        return x,topo_in

class TopoIMG_transModel(nn.Module): #This model is specificaly designed to transform the input of 1x64x64 into 3x32x32 (usable in topoblock configugartion)

    def __init__(self,tmp): #why is tmp here KEKW
        super(TopoIMG_transModel,self).__init__()

        #This implementation gives as output 3x32x32 (given an input of 1x64x64)

        self.conv_network = nn.Sequential(
            nn.Conv2d(in_channels = 1 ,out_channels = 16, kernel_size = (3, 3), stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 16 ,out_channels = 32, kernel_size = (3, 3), stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels = 32 ,out_channels = 3, kernel_size = (3, 3), stride=1,padding=1),
            nn.ReLU()
        )

    def forward(self,x):
        return self.conv_network(x)

class TopoIMG_ResNet(nn.Module): #this is based on the resnet implementation on ResNet (using ResNet as the base to process the images)

    def __init__(self, image_channels, hidden_size):

        super(TopoIMG_ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        #resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        # self.layer4 = self.__make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, hidden_size)

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
        # x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

    def identity_downsample(self, in_channels, out_channels):

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )


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

#Resnet that also uses Topological features
class ResNet_18_Topo(nn.Module):

    def __init__(self, image_channels, num_classes,device):

        super(ResNet_18_Topo, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.device = device

        #resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if args.config: #if the config was set
            layers = [layer_from_config(layer_config) for layer_config in config["topo_net"]]
            self.topo_net = nn.Sequential(*layers)
        else:
            self.topo_net = nn.Sequential(
                nn.Linear(500,128),
                nn.ReLU(),
                nn.Linear(128,128),
                nn.ReLU(),
                nn.Linear(128,64),
                nn.ReLU()
            )
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

        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride),
            Block(out_channels, out_channels)
        )

    def forward(self, x):
        #suppose we do have the topo info in the dataset


        x, topo = x
        x = x.to('cuda:0')
        topo = topo.to('cuda:0')

        x = transforms.functional.resize(x, (112, 112))
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
        #here x = 512, batch size


        x_2 = self.topo_net(topo)
        x_2 = x_2.squeeze(1)
        x = torch.cat([x,x_2],dim=1)
        x = self.fc(x)
        return x

    def identity_downsample(self, in_channels, out_channels):

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )

#Resnet with Topological features as topological images (in the IMG form)
class ResNet_18_TopoPI(nn.Module):

    def __init__(self, image_channels, num_classes,device):

        super(ResNet_18_TopoPI, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.device = device

        #resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if args.config: #if the config was set
            raise('This part is not yet inplemented in the config - TopoPI')
            layers = [layer_from_config(layer_config) for layer_config in config["topo_net"]]
            self.topo_net = nn.Sequential(*layers)
        else: #Based on resnet - for now a potential change in the future KEKW
            self.topo_net = TopoIMG_ResNet(1,64)
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

        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride),
            Block(out_channels, out_channels)
        )

    def forward(self, x):
        #suppose we do have the topo info in the dataset


        x, topo = x
        x = x.to('cuda:0')
        topo = topo.to('cuda:0')

        x = transforms.functional.resize(x, (112, 112))
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
        #here x = 512, batch size


        x_2 = self.topo_net(topo)
        x_2 = x_2.squeeze(1)
        x = torch.cat([x,x_2],dim=1)
        x = self.fc(x)
        return x

    def identity_downsample(self, in_channels, out_channels):

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )


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
        self.topo_emb = TopoIMG_transModel(0) #Why is tmp there? KEKW
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

        # There is no need for topo net

        # if args.config: #if the config was set
        #     layers = [layer_from_config(layer_config) for layer_config in config["topo_net"]]
        #     self.topo_net = nn.Sequential(*layers)
        # else:
        #     self.topo_net = nn.Sequential(
        #         nn.Linear(500,128),
        #         nn.ReLU(),
        #         nn.Linear(128,128),
        #         nn.ReLU(),
        #         nn.Linear(128,64),
        #         nn.ReLU()
        #     )
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

        x_topo = transforms.functional.resize(x_topo, (112, 112))

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_topo = self.conv1_t(x_topo)
        x_topo = self.bn1_t(x_topo)
        x_topo = self.relu_t(x_topo)
        x_topo = self.maxpool_t(x_topo)


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

class ResNet_18_Topo_2dim(nn.Module): #This model is for the input that consists of two topological dimensions that are concatenated (but we want to split them)

    def __init__(self, image_channels, num_classes,device):

        super(ResNet_18_Topo_2dim, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.device = device

        #resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        #Setting first self.topo_net from config (for the 0 dim homology)
        if args.config:
            layers = [layer_from_config(layer_config) for layer_config in config["topo_net"]]
            self.topo_net = nn.Sequential(*layers)
        else:
            self.topo_net = nn.Sequential(
                nn.Linear(500,256),
                nn.ReLU(),
                nn.Linear(256,128),
                nn.ReLU(),
                nn.Linear(128,64),
                nn.ReLU()
            )

        #setting the second topo_net (for the 1 dim homology)
        if args.config:
            layers = [layer_from_config(layer_config) for layer_config in config["topo_net"]]
            self.topo_net_2 = nn.Sequential(*layers)
        else:
            self.topo_net_2 = nn.Sequential(
                nn.Linear(500,256),
                nn.ReLU(),
                nn.Linear(256,128),
                nn.ReLU(),
                nn.Linear(128,64),
                nn.ReLU()
            )

        #Setting the res_net_fc from config
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

        #setting the fc network (final network) from config
        if args.config:
            layers = [layer_from_config(layer_config) for layer_config in config["fc"]]
            self.fc = nn.Sequential(*layers)
        else:
            self.fc = nn.Sequential(nn.Linear(192,64), #the size of the Sequential should be 3xhidden_size
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

        return nn.Sequential(
            Block(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride),
            Block(out_channels, out_channels)
        )

    def forward(self, x):
        #suppose we do have the topo info in the dataset


        x, topo = x
        x = x.to('cuda:0')
        topo = topo.to('cuda:0')
        topo = torch.split(topo,int(topo.shape[2]/2),dim=-1) #to not split the batch but the homology tensors

        topo_0 = topo[0]
        topo_1 = topo[1]


        x = transforms.functional.resize(x, (112, 112))
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
        #here x = 512, batch size


        topo_0 = self.topo_net(topo_0)
        topo_0 = topo_0.squeeze(1)

        topo_1 = self.topo_net(topo_1)
        topo_1 = topo_1.squeeze(1)

        x = torch.cat([x,topo_0,topo_1],dim=1)
        x = self.fc(x)
        return x

    def identity_downsample(self, in_channels, out_channels):

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )


class ResNet_18_Topo_Block(nn.Module):

    def __init__(self, image_channels, num_classes,device):

        super(ResNet_18_Topo_Block, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.device = device

        #resnet layers
        self.layer1 = self.__make_layer(64, 64, stride=1)
        self.layer2 = self.__make_layer(64, 128, stride=2)
        self.layer3 = self.__make_layer(128, 256, stride=2)
        self.layer4 = self.__make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))



        if args.config: #if the config was set
            layers = [layer_from_config(layer_config) for layer_config in config["topo_net"]]
            self.topo_net = nn.Sequential(*layers)
        else:
            self.topo_net = nn.Sequential(
                nn.Linear(500,128),
                nn.ReLU(),
                nn.Linear(128,128),
                nn.ReLU(),
                nn.Linear(128,64),
                nn.ReLU()
            )
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


        return LayerTopoBlock(in_channels, out_channels, identity_downsample, stride)


    def forward(self, x):
        #suppose we do have the topo info in the dataset

        x, topo = x

        x = x.to('cuda:0')
        topo = topo.to('cuda:0')

        x = transforms.functional.resize(x, (112, 112))
        x_topo = self.topo_net(topo)
        x_topo = x_topo.squeeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1((x,x_topo))

        x = self.layer2((x,x_topo))
        x = self.layer3((x,x_topo))
        x = self.layer4((x,x_topo))

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        x = self.res_net_fc(x)
        #here x = 512, batch size

        x = torch.cat([x,x_topo],dim=1)
        x = self.fc(x)
        return x

    def identity_downsample(self, in_channels, out_channels):

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )




#Helper function for counting trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)





#Training function
def train_model(model, dataloaders, criterion, optimizer, num_epochs=50, is_inception=False):
    print('Traning model')

    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0


    train_acc = 0.0
    train_loss = 10
    val_loss = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']: # Each epoch has a training and validation phase
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]): # Iterate over data
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                # inputs = transforms.functional.resize(inputs, (112, 112))
                x1,x2 = inputs
                x1.to(device)
                x2.to(device)
                inputs = (x1,x2)

                labels = labels.to(device)

                optimizer.zero_grad() # Zero the parameter gradients

                with torch.set_grad_enabled(phase == 'train'): # Forward. Track history if only in train

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train': # Backward + optimize only if in training phase
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs[0].size(0)
                running_corrects += torch.sum(preds == labels.data)


            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if phase =='val':
                writer.add_scalar("Loss/train", epoch_loss, epoch)
                writer.add_scalar("Accuracy/train", epoch_acc, epoch)
                if epoch_loss < val_loss:
                    val_loss = epoch_loss
            else:
                if epoch_loss < train_loss:
                    train_loss = epoch_loss
                if epoch_acc > train_acc:
                    train_acc = epoch_acc
                writer.add_scalar("Loss/Val", epoch_loss, epoch)
                writer.add_scalar("Accuracy/Val", epoch_acc, epoch)

            if phase == 'val': # Adjust learning rate based on val loss
                lr_scheduler.step(epoch_loss)



            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    writer.flush()
    writer.close()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    with open(result_file,'a') as f:
        new_line = f'Model: {args.name} train_loss: {train_loss} val_loss: {val_loss} train_acc: {train_acc*100} val_acc: {best_acc*100}\n'
        f.writelines(new_line)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def test_model(model, dataloader):
    print('Testing model')
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for inputs, labels in tqdm(dataloader):

        if args.model != 'ResNet':
            x1,x2 = inputs
            x1.to(device)
            x2.to(device)
            inputs = (x1,x2)

        labels = labels.to(device)

        optimizer.zero_grad() # Zero the parameter gradients

        with torch.set_grad_enabled(False): # Forward. Track history if only in train

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

        # Statistics
        running_loss += loss.item() * inputs[0].size(0)
        running_corrects += torch.sum(preds == labels.data)


    total_loss = running_loss / len(dataloader.dataset)
    total_acc = running_corrects.double() / len(dataloader.dataset)
    print('{} Loss: {:.4f} Acc: {:.4f}'.format('Test', total_loss, total_acc))


#Function to process the image that is:
# - Change the image to Gray-scale
# - Calculate the topological features
def process_img_topo_land(data, to_grayscale = transforms.Grayscale(num_output_channels=1)): #Processing to Landscapes
    #Grayscale using provided function
    try:
        transform=transforms.Compose(
            [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        # transforms = v2.Compose([
        #     v2.RandomResizedCrop(size=(224, 224), antialias=True),
        #     v2.RandomHorizontalFlip(p=0.5),
        #     v2.ToDtype(torch.float32, scale=True),
        #     v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        #     ])


        # test_transform = transforms.Compose([
        #         transforms.ToTensor(),
        #         transforms.Normalize(mean=[0.5, 0.5, 0.5])
        #     ])
        # transform = transforms.Compose([
        #         transforms.RandomHorizontalFlip(),    # Randomly flip the image horizontally
        #         transforms.RandomCrop(32, padding=4), # Randomly crop the image with padding
        #         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Color adjustments
        #         transforms.RandomRotation(15),        # Randomly rotate the image
        #         transforms.ToTensor(),                # Convert image to PyTorch tensor
        #         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize the image
        #     ])

        if args.bw =='cv2':
            transform_bw=transforms.Compose(
                [transforms.ToTensor(),
                    transforms.Normalize(0.5, 0.5, 0.5)])
            bw_img = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
            img = Image.fromarray(data)
            bw_img = Image.fromarray(bw_img)
            image = transform(img)
            bw_img = transform_bw(bw_img)

        elif args.bw == 'torch':
            pass #TODO

        gray_scale_img = bw_img # to_grayscale(image)
        # calcuating the cubical complex
        cubical_complex = gd.CubicalComplex(dimensions=gray_scale_img.shape, top_dimensional_cells=gray_scale_img.flatten())
        # Calculating persistance
        diag = cubical_complex.persistence()
        # Calculating Landscape
        LS = gd.representations.Landscape(resolution=args.res)

        if args.topodim_concat:
            LS_0 = LS.fit_transform([cubical_complex.persistence_intervals_in_dimension(0)[:-1]])
            L_t_0 = torch.tensor(LS_0,dtype=torch.float)
            LS_1 = LS.fit_transform([cubical_complex.persistence_intervals_in_dimension(1)])
            L_t_1 = torch.tensor(LS_1,dtype=torch.float)
            L_t = torch.cat([L_t_0,L_t_1],dim=1)

        elif args.topodim == 0:
            LS = LS.fit_transform([cubical_complex.persistence_intervals_in_dimension(args.topodim)[:-1]])
            L_t = torch.tensor(LS,dtype=torch.float)

        elif args.topodim == 1:
            LS = LS.fit_transform([cubical_complex.persistence_intervals_in_dimension(args.topodim)])
            L_t = torch.tensor(LS,dtype=torch.float)


        # L = LS.fit_transform([cubical_complex.persistence_intervals_in_dimension(args.topodim)])
        # L_t = torch.tensor(L,dtype=torch.float)





        # L_t = L_t[:,:200]

        if L_t is None:
            raise('None in the Landscape processing L_T')

        return image, L_t

    except Exception as e:
        print(f"Error with item {item}: {e}")
        return None

def process_img_topo_betti_curve(data, to_grayscale = transforms.Grayscale(num_output_channels=1)): #Processing to Betti_curves
    #Grayscale using provided function


    transform=transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    # transforms = v2.Compose([
    #     v2.RandomResizedCrop(size=(224, 224), antialias=True),
    #     v2.RandomHorizontalFlip(p=0.5),
    #     v2.ToDtype(torch.float32, scale=True),
    #     v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    #     ])

    # transform = transforms.Compose([
    #         transforms.RandomHorizontalFlip(),    # Randomly flip the image horizontally
    #         transforms.RandomCrop(32, padding=4), # Randomly crop the image with padding
    #         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Color adjustments
    #         transforms.RandomRotation(15),        # Randomly rotate the image
    #         transforms.ToTensor(),                # Convert image to PyTorch tensor
    #         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize the image
    #     ])

    transform_bw=transforms.Compose(
            [transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5, 0.5)])

    bw_img = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(data)
    bw_img = Image.fromarray(bw_img)
    image = transform(img)
    bw_img = transform_bw(bw_img)
    # print(image.shape)
    gray_scale_img = bw_img # to_grayscale(image)
    # calcuating the cubical complex
    cubical_complex = gd.CubicalComplex(dimensions=gray_scale_img.shape, top_dimensional_cells=gray_scale_img.flatten())
    # Calculating persistance
    diag = cubical_complex.persistence()
    # Calculating BettiCurve
    BC = gd.representations.vector_methods.BettiCurve()

    #This is created, because of the error caused by the  [x,+inf] persistant interval (connected components)
    if args.topodim_concat:
        BC_0 = BC.fit_transform([cubical_complex.persistence_intervals_in_dimension(0)[:-1]])
        L_t_0 = torch.tensor(BC_0,dtype=torch.float)
        BC_1 = BC.fit_transform([cubical_complex.persistence_intervals_in_dimension(1)])
        L_t_1 = torch.tensor(BC_1,dtype=torch.float)
        L_t = torch.cat([L_t_0,L_t_1],dim=1)

    elif args.topodim == 0:
        BC = BC.fit_transform([cubical_complex.persistence_intervals_in_dimension(args.topodim)[:-1]])
        L_t = torch.tensor(BC,dtype=torch.float)

    elif args.topodim == 1:
        BC = BC.fit_transform([cubical_complex.persistence_intervals_in_dimension(args.topodim)])
        L_t = torch.tensor(BC,dtype=torch.float)



    return image, L_t

#Processing using the topological Silhouette
def process_img_topo_silh(data, to_grayscale = transforms.Grayscale(num_output_channels=1)): #Processing to Silhouette
    #Grayscale using provided function


    transform=transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    # transforms = v2.Compose([
    #     v2.RandomResizedCrop(size=(224, 224), antialias=True),
    #     v2.RandomHorizontalFlip(p=0.5),
    #     v2.ToDtype(torch.float32, scale=True),
    #     v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    #     ])

    # transform = transforms.Compose([
    #         transforms.RandomHorizontalFlip(),    # Randomly flip the image horizontally
    #         transforms.RandomCrop(32, padding=4), # Randomly crop the image with padding
    #         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Color adjustments
    #         transforms.RandomRotation(15),        # Randomly rotate the image
    #         transforms.ToTensor(),                # Convert image to PyTorch tensor
    #         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize the image
    #     ])

    transform_bw=transforms.Compose(
            [transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5, 0.5)])

    bw_img = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(data)
    bw_img = Image.fromarray(bw_img)
    image = transform(img)
    bw_img = transform_bw(bw_img)
    # print(image.shape)
    gray_scale_img = bw_img # to_grayscale(image)
    # calcuating the cubical complex
    cubical_complex = gd.CubicalComplex(dimensions=gray_scale_img.shape, top_dimensional_cells=gray_scale_img.flatten())
    # Calculating persistance
    diag = cubical_complex.persistence()
    # Calculating BettiCurve
    SI = gd.representations.vector_methods.Silhouette()

    #This is created, because of the error caused by the  [x,+inf] persistant interval (connected components)
    if args.topodim_concat:
        SI_0 = SI.fit_transform([cubical_complex.persistence_intervals_in_dimension(0)[:-1]])
        L_t_0 = torch.tensor(SI_0,dtype=torch.float)
        SI_1 = SI.fit_transform([cubical_complex.persistence_intervals_in_dimension(1)])
        L_t_1 = torch.tensor(SI_1,dtype=torch.float)
        L_t = torch.cat([L_t_0,L_t_1],dim=1)

    elif args.topodim == 0:
        SI = SI.fit_transform([cubical_complex.persistence_intervals_in_dimension(args.topodim)[:-1]])
        L_t = torch.tensor(SI,dtype=torch.float)

    elif args.topodim == 1:
        SI = SI.fit_transform([cubical_complex.persistence_intervals_in_dimension(args.topodim)])
        L_t = torch.tensor(SI,dtype=torch.float)



    return image, L_t


def process_img_topo_pi_v(data, to_grayscale = transforms.Grayscale(num_output_channels=1)): #Processing to Persistant images
    #Grayscale using provided function


    transform=transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    # transforms = v2.Compose([
    #     v2.RandomResizedCrop(size=(224, 224), antialias=True),
    #     v2.RandomHorizontalFlip(p=0.5),
    #     v2.ToDtype(torch.float32, scale=True),
    #     v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    #     ])

    # transform = transforms.Compose([
    #         transforms.RandomHorizontalFlip(),    # Randomly flip the image horizontally
    #         transforms.RandomCrop(32, padding=4), # Randomly crop the image with padding
    #         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Color adjustments
    #         transforms.RandomRotation(15),        # Randomly rotate the image
    #         transforms.ToTensor(),                # Convert image to PyTorch tensor
    #         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize the image
    #     ])

    transform_bw=transforms.Compose(
            [transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5, 0.5)])

    bw_img = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(data)
    bw_img = Image.fromarray(bw_img)
    image = transform(img)
    bw_img = transform_bw(bw_img)
    # print(image.shape)
    gray_scale_img = bw_img # to_grayscale(image)
    # calcuating the cubical complex
    cubical_complex = gd.CubicalComplex(dimensions=gray_scale_img.shape, top_dimensional_cells=gray_scale_img.flatten())
    # Calculating persistance
    diag = cubical_complex.persistence()
    # Calculating BettiCurve
    PI = gd.representations.PersistenceImage(bandwidth=0.05,resolution=[64,64],weight=lambda x: x[1]**2, im_range=[0,0.6,0,0.6])

    #This is created, because of the error caused by the  [x,+inf] persistant interval (connected components)
    if args.topodim_concat:
        PI_0 = PI.fit_transform([cubical_complex.persistence_intervals_in_dimension(0)[:-1]])
        L_t_0 = torch.tensor(PI_0,dtype=torch.float)
        PI_1 = PI.fit_transform([cubical_complex.persistence_intervals_in_dimension(1)])
        L_t_1 = torch.tensor(PI_1,dtype=torch.float)
        L_t = torch.cat([L_t_0,L_t_1],dim=1)

    elif args.topodim == 0:
        PI = PI.fit_transform([cubical_complex.persistence_intervals_in_dimension(args.topodim)[:-1]])
        L_t = torch.tensor(PI,dtype=torch.float)

    elif args.topodim == 1:
        PI = PI.fit_transform([cubical_complex.persistence_intervals_in_dimension(args.topodim)])
        L_t = torch.tensor(PI,dtype=torch.float)



    return image, L_t


def process_img_topo_pi_img(data, to_grayscale = transforms.Grayscale(num_output_channels=1)): #Processing to Persistant images
    #Grayscale using provided function


    transform=transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    #This transform is for the augmentation methods

    # transforms = v2.Compose([
    #     v2.RandomResizedCrop(size=(224, 224), antialias=True),
    #     v2.RandomHorizontalFlip(p=0.5),
    #     v2.ToDtype(torch.float32, scale=True),
    #     v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    #     ])

    # transform = transforms.Compose([
    #         transforms.RandomHorizontalFlip(),    # Randomly flip the image horizontally
    #         transforms.RandomCrop(32, padding=4), # Randomly crop the image with padding
    #         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Color adjustments
    #         transforms.RandomRotation(15),        # Randomly rotate the image
    #         transforms.ToTensor(),                # Convert image to PyTorch tensor
    #         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # Normalize the image
    #     ])

    transform_bw=transforms.Compose(
            [transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5, 0.5)])

    bw_img = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(data)
    bw_img = Image.fromarray(bw_img)
    image = transform(img)
    bw_img = transform_bw(bw_img)
    # print(image.shape)
    gray_scale_img = bw_img # to_grayscale(image)
    # calcuating the cubical complex
    cubical_complex = gd.CubicalComplex(dimensions=gray_scale_img.shape, top_dimensional_cells=gray_scale_img.flatten())
    # Calculating persistance
    diag = cubical_complex.persistence()
    # Calculating BettiCurve
    PI = gd.representations.PersistenceImage(bandwidth=0.05,resolution=[64,64],weight=lambda x: x[1]**2, im_range=[0,0.6,0,0.6])

    #For the Persistent Images, the concat output gives 2 images - a simple solution
    if args.topodim_concat:
        PI_0 = PI.fit_transform([cubical_complex.persistence_intervals_in_dimension(0)[:-1]])
        L_t_0 = torch.tensor(PI_0,dtype=torch.float)
        PI_1 = PI.fit_transform([cubical_complex.persistence_intervals_in_dimension(1)])
        L_t_1 = torch.tensor(PI_1,dtype=torch.float)
        L_t = (L_t_0,L_t_1)

    elif args.topodim == 0:
        PI = PI.fit_transform([cubical_complex.persistence_intervals_in_dimension(args.topodim)[:-1]])
        L_t = torch.tensor(PI,dtype=torch.float).reshape([1,64,64])

    elif args.topodim == 1:
        PI = PI.fit_transform([cubical_complex.persistence_intervals_in_dimension(args.topodim)])
        L_t = torch.tensor(PI,dtype=torch.float).reshape([1,64,64])



    return image, L_t

def process_data_topo(dataset, train_set = True, from_train = None):
    data = dataset.data

    data_len = data.shape[0]

    if args.tv == 'land':
        print('Processing data using landscape vectorization')
        if args.cores <= 1:
            results = []
            print('processing using a single core - NO multiprocessing')
            for sample in tqdm(data):
                results.append(process_img_topo_land(sample))
        else:
            with Pool(args.cores) as pool: #multiprocessing the topological data transform
                results = list(tqdm(pool.imap(process_img_topo_land,data),total=len(data)))


    elif args.tv == 'bc':
        print('Processing data using betti curve vectorization')

        if args.cores <= 1:
            results = []
            print('processing using a single core - NO multiprocessing')
            for sample in tqdm(data):
                results.append(process_img_topo_betti_curve(sample))
        else:
            with Pool(args.cores) as pool: #multiprocessing the topological data transform
                results = list(tqdm(pool.imap(process_img_topo_betti_curve,data),total=len(data)))

    elif args.tv == 'pi_v':
        print('Processing data using PI (Vector)')

        if args.cores <= 1:
            results = []
            print('processing using a single core - NO multiprocessing')
            for sample in tqdm(data):
                results.append(process_img_topo_pi_v(sample))
        else:
            with Pool(args.cores) as pool: #multiprocessing the topological data transform
                results = list(tqdm(pool.imap(process_img_topo_pi_v,data),total=len(data)))

    elif args.tv == 'pi_img':
        print('Processing data using PI (Image)')

        if args.cores <= 1:
            results = []
            print('processing using a single core - NO multiprocessing')
            for sample in tqdm(data):
                results.append(process_img_topo_pi_img(sample))
        else:
            with Pool(args.cores) as pool: #multiprocessing the topological data transform
                results = list(tqdm(pool.imap(process_img_topo_pi_img,data),total=len(data)))

    elif args.tv == 'silh':
        print('Processing data using Silhouette vectorization')

        if args.cores <= 1:
            results = []
            print('processing using a single core - NO multiprocessing')
            for sample in tqdm(data):
                results.append(process_img_topo_silh(sample))
        else:
            with Pool(args.cores) as pool: #multiprocessing the topological data transform
                results = list(tqdm(pool.imap(process_img_topo_silh,data),total=len(data)))

    else:
        raise('Error - invalid topological vectorization method')



    tmp_topo_data = [item[1] for item in results]
    tensor_topo_data = torch.cat(tmp_topo_data,dim=0)

    if from_train is not None:
        max = from_train[0]
        min = from_train[1]
    else:
        min = tensor_topo_data.min()
        max = tensor_topo_data.max()

    new_res = []
    for img,topo in results:
        stand_topo = (topo - min)/(max-min)
        new_res.append((img,stand_topo))
    results = new_res

    if train_set:
        return results, (max,min)

    return results




#Main function
if __name__ == "__main__":

    print(args)

    print('Trying to load the data')

    try:
        with open(train_set_path, 'rb') as f:
            train_set = pickle.load(f)

        with open(train_set_target_path, 'rb') as f:
            train_set_target = pickle.load(f)

        with open(test_set_path, 'rb') as f:
            test_set = pickle.load(f)

        with open(test_set_target_path, 'rb') as f:
            test_set_target = pickle.load(f)

        trainset = MyDataset(train_set,train_set_target)

        train_size = int((1 - args.val_size) * len(trainset))
        test_size = len(trainset) - train_size

        trainset, valset  = random_split(trainset, [train_size, test_size])

        testset = MyDataset(test_set,test_set_target)
        print('Data loading succesfull')

    except:
        print('Failed to load the data, processing the data and saving')

        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True)#, transform=transform)
        data_fin_train, from_train = process_data_topo(trainset)

        with open(train_set_path, 'wb') as f:
            pickle.dump(data_fin_train, f)

        with open(train_set_target_path, 'wb') as f:
            pickle.dump(trainset.targets, f)

        trainset = MyDataset(data_fin_train,trainset.targets)

        train_size = int((1-args.val_size) * len(trainset))
        test_size = len(trainset) - train_size

        trainset, valset  = random_split(trainset, [train_size, test_size])

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True)#, transform=transform)
        data_fin_test = process_data_topo(testset,train_set= False, from_train = from_train)


        with open(test_set_path, 'wb') as f:
            pickle.dump(data_fin_test, f)

        with open(test_set_target_path, 'wb') as f:
            pickle.dump(testset.targets, f)

        testset = MyDataset(data_fin_test,testset.targets)




    if args.model == 'ResNet':
        print('Using the datasets WITHOUT the topological features')
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

        train_size = int(0.8 * len(trainset))
        test_size = len(trainset) - train_size

        trainset, valset  = random_split(trainset, [train_size, test_size])

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                              shuffle=True, num_workers=args.num_workers)

        valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                              shuffle=True, num_workers=args.num_workers)

        testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                                         shuffle=False, num_workers=args.num_workers)

    else: #Creating the dataset Loaders with the topological options
        print('Using the datasets WITH the topological features')

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                      shuffle=True, num_workers=args.num_workers)

        valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,

                                                              shuffle=True, num_workers=args.num_workers)
        testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                                 shuffle=False, num_workers=args.num_workers)



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    if args.model =='ResNet':
        model = ResNet_18(3,10)
    elif args.model == 'TR':
        model = ResNet_18_Topo(3,10,device)
    elif args.model =='TBR':
        model = ResNet_18_Topo_Block(3,10,device)
    elif args.model =='TR_2dim':
        model = ResNet_18_Topo_2dim(3,10,device)
    elif args.model =='TR_img':
        if args.tv != 'pi_img':
            raise('Wrong vectorization fo TimgRes')
        model = ResNet_18_TopoPI(3,10,device)
    elif args.model =='TBR_img':
        if args.tv != 'pi_img':
            raise('Wrong vectorization fo TimgRes')
        model = ResNet_18_PIBlock(3,10,device)
    else:
        print('Error - Incorrect model option')
        raise('Error - Incorrect model option')
        # return

    model.to(device)

    print(f'Currenttly running model: {args.model} with {args.tv} ')
    print(f'Number of paramters: {count_parameters(model)}')

    epochs = args.epochs
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

    model, _ = train_model(model, {"train": trainloader, "val": valloader}, criterion, optimizer, epochs)
    test_model(model,testloader)

    if args.sm:
        print('Saving model')
        torch.save(model.state_dict(), model_saving_path)
