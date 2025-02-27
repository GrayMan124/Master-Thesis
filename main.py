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

#imports from the files
from utils import test_model, train_model, count_parameters
from models.ResNet import *
from data_processing import *
from models.TopoResNet import *
from models.ResNetPIBlock import *
from models.ResNetTopo2Dim import *
from models.ResNetTopoBlock import *
from models.TopoResNet import *
from models.TopoResNetPI import * 
from models.ResNet import *

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
argparser.add_argument("--tb_add_t", default=False, action="store_true", help="Add the topological out back into the topological resnet when using PIBlock")


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
    test_model(model,testloader,criterion, optimizer)

    if args.sm:
        print('Saving model')
        torch.save(model.state_dict(), model_saving_path)
