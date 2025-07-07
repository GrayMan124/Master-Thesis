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
import pandas as pd

from config import args


model_saving_path = 'models/'+ args.name + args.model + '_' + args.tv + '_' + str(args.lr) + '_' + str(args.res) + '_' + str(args.seed) + '_' + str(args.topodim) + '_' + args.bw +'.pkl'
tensor_board_path = 'runs/' + args.name + args.model + '_' + args.tv + '_' + str(args.lr) + '_' + str(args.res) + '_' + str(args.seed) + '_' + str(args.topodim) + '_' + args.bw





def test_model(model, dataloader,criterion, optimizer):
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
        else:
            inputs = inputs.to(device)

        labels = labels.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
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



#Helper function for counting trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)




result_file = 'results.csv'

writer = SummaryWriter(log_dir = tensor_board_path)


#Training function
def train_model(model, dataloaders, criterion, optimizer, num_epochs=50, is_inception=False):
    print('Traning model')

    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)#, verbose=True)

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
                if args.model != 'ResNet':
                    x1,x2 = inputs
                    x1.to(device)
                    x2.to(device)
                    inputs = (x1,x2)
                else:
                    inputs = inputs.to(device)
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
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    writer.flush()
    writer.close()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    results_dict = {
        "Model_Name":args.name,
        "Topology_Vect":args.tv,
        "Model_arch":args.model,
        "LR":args.lr,
        "TopoDim_concat":args.topodim_concat,
        "TopoDim": args.topodim,
        "Config_file":args.config,
        "Augmentation":args.aug,
        "Augmentation_Type":args.aug_type,
        "Train_loss":train_loss,
        "Train_Acc":train_acc*100,
        "Val_Loss":best_loss,
        "Val_Acc":best_acc*100
    }

    result_df = pd.DataFrame([results_dict])

    try:
        df = pd.read_csv(result_file)
        new_df = pd.concat([df,result_df],ignore_index = True)
        new_df.to_csv(result_file,index= False)
    except:
        result_df.to_csv(result_file,index=False) 
    
    # with open(result_file,'a') as f:
    #     new_line = f'Model: {args.name} train_loss: {train_loss} val_loss: {val_loss} train_acc: {train_acc*100} val_acc: {best_acc*100}\n'
    #     f.writelines(new_line)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history
