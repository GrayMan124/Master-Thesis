#Update, version from: 09_02_2025
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pickle
import gudhi as gd
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from PIL import Image
import torch.optim as optim
from torch.utils.data import Dataset, random_split
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
import copy
import json
from torch.utils.data import Subset

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

from config import args

 
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(args.seed)


model_saving_path = 'models/saved_models/'+ args.name + args.model + '_' + args.tv + '_' + str(args.lr) + '_' + str(args.res) + '_' + str(args.seed) + '_' + str(args.topodim) + '_' + args.bw +'.pkl'
tensor_board_path = 'runs/' + args.name + args.model + '_' + args.tv + '_' + str(args.lr) + '_' + str(args.res) + '_' + str(args.seed) + '_' + str(args.topodim) + '_' + args.bw


base_dataset_path = 'data/'
train_set_path = base_dataset_path +  'train_set_' + args.tv + '_' + str(args.res) + '_' + str(args.topodim) + '_' + str(args.topodim_concat) +  '_' + args.bw +'.pkl'
train_set_target_path = base_dataset_path +  'train_set_target_' + args.tv + '_' + str(args.res) + '_' + str(args.topodim)+ '_' + str(args.topodim_concat) + '_' + args.bw  +'.pkl'

aug_set_path = base_dataset_path +  'aug_set_' + str(args.aug*100) +args.aug_type + args.tv + '_' + str(args.res) + '_' + str(args.topodim) + '_' + str(args.topodim_concat) +  '_' + args.bw +'.pkl'
aug_set_target_path = base_dataset_path +  'aug_set_target_' + str(args.aug*100) + args.aug_type+ args.tv + '_' + str(args.res) + '_' + str(args.topodim)+ '_' + str(args.topodim_concat) + '_' + args.bw  +'.pkl'


test_set_path = base_dataset_path +  'test_set_' + args.tv + '_' + str(args.res) + '_' + str(args.topodim) + '_' + str(args.topodim_concat) + '_' + args.bw + '.pkl'
test_set_target_path = base_dataset_path +  'test_set_target_' + args.tv + '_' + str(args.res) + '_' + str(args.topodim) + '_' +  str(args.topodim_concat) + '_' + args.bw  + '.pkl'


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

    #loading the data
    if args.model != 'ResNet':
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

            #Splitting train set into validation set
            train_size = int((1 - args.val_size) * len(trainset))
            test_size = len(trainset) - train_size

            trainset, valset  = random_split(trainset, [train_size, test_size])
            
            testset = MyDataset(test_set,test_set_target)
            


            #Loading augmented data
            if args.aug > 0:
                with open(aug_set_path, 'rb') as f:
                    aug_set = pickle.load(f)
                
                with open(aug_set_target_path, 'rb') as f:
                    aug_set_target = pickle.load(f)
                

                aug_data_set = MyDataset(aug_set,aug_set_target)
                
                #Transforming the train set tu be combined with the augmentation set later
                subset_train = [trainset.dataset[i] for i in trainset.indices]

                subset_train_data = [sample[0] for sample in subset_train]
                subset_train_label = [sample[1] for sample in subset_train]

                subset_train_my_data = MyDataset(subset_train_data,subset_train_label)

                #Final train set with the augmentation
                trainset = ConcatDataset([aug_data_set,subset_train_my_data])


            print('Data loading succesfull')

        except:
            print('Failed to load the data, processing the data and saving')
            
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True)
            
            data_fin_train, from_train = process_data_topo(trainset)

            with open(train_set_path, 'wb') as f:
                pickle.dump(data_fin_train, f)

            with open(train_set_target_path, 'wb') as f:
                pickle.dump(trainset.targets, f)

            trainset_main = MyDataset(data_fin_train,trainset.targets)

            train_size = int((1-args.val_size) * len(trainset))
            test_size = len(trainset) - train_size

            trainset, valset  = random_split(trainset_main, [train_size, test_size])
            
            testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                                download=True)
            
            data_fin_test = process_data_topo(testset,train_set= False, from_train = from_train)


            with open(test_set_path, 'wb') as f:
                pickle.dump(data_fin_test, f)

            with open(test_set_target_path, 'wb') as f:
                pickle.dump(testset.targets, f)

            testset = MyDataset(data_fin_test,testset.targets)
            
            if args.aug > 0:
                
                if args.aug_type =='all':
                    transform_aug = transforms.Compose([
                            transforms.RandomHorizontalFlip(),    # Randomly flip the image horizontally
                            transforms.RandomVerticalFlip(),
                            transforms.RandomErasing(),
                            transforms.RandomResizedCrop(6), # Randomly crop the image with padding
                            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Color adjustments
                            transforms.RandomRotation(15),        # Randomly rotate the image
                            transforms.GaussianBlur((5,5)),
                            transforms.RandomPerspective()
                        ])
                    
                elif args.aug_type =='non-topo':
                    transform_aug = transforms.Compose([
                            transforms.RandomHorizontalFlip(),    # Randomly flip the image horizontally
                            transforms.RandomVerticalFlip(),
                            # transforms.RandomResizedCrop(16, padding=4), # Randomly crop the image with padding
                            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Color adjustments
                            # transforms.RandomRotation(15),        # Randomly rotate the image
                            transforms.GaussianBlur((5,5))
                        ])
                    
                elif args.aug_type =='topo':
                    transform_aug = transforms.Compose([
                            # transforms.RandomHorizontalFlip(),    # Randomly flip the image horizontally
                            # transforms.RandomVerticalFlip(),
                            transforms.RandomErasing(),
                            transforms.RandomResizedCrop(6), # Randomly crop the image with padding
                            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Color adjustments
                            transforms.RandomRotation(15),        # Randomly rotate the image
                            transforms.RandomPerspective()
                            # transforms.GaussianBlur((5,5))
                        ])
        
                aug_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform = transform_aug )
                

                slice_size = int(args.aug * len(aug_set))


                aug_targets = aug_set.targets[:slice_size]

                
                subset_train = [trainset.dataset[i] for i in trainset.indices]


                subset_train_data = [sample[0] for sample in subset_train]
                subset_train_label = [sample[1] for sample in subset_train]

                subset_train_my_data = MyDataset(subset_train_data,subset_train_label)
                aug_set = process_data_topo(aug_set, train_set= False, from_train = from_train, slice = slice_size)


                aug_data_set = MyDataset(aug_set,aug_targets)


                print("Priting the samples from the cocnatenated dataset")
                trainset = ConcatDataset([aug_data_set,subset_train_my_data])
                print(f'Trainset: {trainset}')

                with open(aug_set_path, 'wb') as f:
                    aug_set = pickle.dump(aug_set,f)
                
                with open(aug_set_target_path, 'wb') as f:
                    aug_set_target = pickle.dump(aug_targets,f)

            print('Using the datasets WITH the topological features')

            trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                                        shuffle=True, num_workers=args.num_workers)

            valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,

                                                                shuffle=True, num_workers=args.num_workers)
            testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                                 shuffle=False, num_workers=args.num_workers)

    elif args.model == 'ResNet':
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
        
        if args.aug > 0:
            
            if args.aug_type =='all':
                transform_aug = transforms.Compose([
                        transforms.RandomHorizontalFlip(),    # Randomly flip the image horizontally
                        transforms.RandomVerticalFlip(),
                        transforms.RandomErasing(),
                        transforms.RandomResizedCrop(6), # Randomly crop the image with padding
                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Color adjustments
                        transforms.RandomRotation(15),        # Randomly rotate the image
                        transforms.GaussianBlur((5,5)),
                        transforms.RandomPerspective(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
                
            elif args.aug_type =='non-topo':
                transform_aug = transforms.Compose([
                        transforms.RandomHorizontalFlip(),    # Randomly flip the image horizontally
                        transforms.RandomVerticalFlip(),
                        # transforms.RandomResizedCrop(16, padding=4), # Randomly crop the image with padding
                        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Color adjustments
                        # transforms.RandomRotation(15),        # Randomly rotate the image
                        transforms.GaussianBlur((5,5)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
                
            elif args.aug_type =='topo':
                transform_aug = transforms.Compose([
                        # transforms.RandomHorizontalFlip(),    # Randomly flip the image horizontally
                        # transforms.RandomVerticalFlip(),
                        transforms.RandomErasing(),
                        transforms.RandomResizedCrop(6), # Randomly crop the image with padding
                        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Color adjustments
                        transforms.RandomRotation(15),        # Randomly rotate the image
                        transforms.RandomPerspective(),
                        # transforms.GaussianBlur((5,5))
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ])
    
            aug_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                               download=True, transform = transform_aug )
            

            slice_size = int(args.aug * len(aug_set))


            aug_targets = aug_set.targets[:slice_size]

            
            subset_train = [trainset.dataset[i] for i in trainset.indices]


            subset_train_data = [sample[0] for sample in subset_train]
            subset_train_label = [sample[1] for sample in subset_train]

            subset_train_my_data = MyDataset(subset_train_data,subset_train_label)
            
            trainset = ConcatDataset([aug_set,subset_train_my_data])
        

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
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)#, verbose=True)

    model, _ = train_model(model, {"train": trainloader, "val": valloader}, criterion, optimizer, epochs)
    test_model(model,testloader,criterion, optimizer)

    if args.sm:
        print('Saving model')
        torch.save(model.state_dict(), model_saving_path)
