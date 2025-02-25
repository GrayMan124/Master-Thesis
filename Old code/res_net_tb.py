import numpy as np
import pandas as pd
import pickle
import gudhi as gd
import sklearn
import matplotlib
import gudhi.representations
from sklearn import manifold
from sklearn.datasets import make_circles
from pylab import *
from matplotlib import pyplot as plt
import sklearn
import torch
import torchvision
import torchvision.transforms as transforms
import torch
import torchvision
import torch.nn as nn
import cv2
from PIL import Image
from torchvision.transforms import v2
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import copy
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from torch.utils.data import Dataset, DataLoader
# import time

torch.autograd.set_detect_anomaly(True)
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


class LayerTopoBlock(nn.Module):

    def __init__(self,in_channels, out_channels, identity_downsample, stride):
        super(LayerTopoBlock,self).__init__()
        self.block1 = TopoBlock(in_channels, out_channels,identity_downsample,500, stride)
        self.block2 = TopoBlock(out_channels, out_channels)

    def forward(self,x):
        x,topo = x
        x,_ = self.block1((x,topo))
        x,_ = self.block2((x,topo))
        return x
#Residual block
class TopoBlockIN(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, topo_size_in = 500, stride=1):
        super(TopoBlockIN, self).__init__()
        self.topo_enc = nn.Sequential(nn.Linear(topo_size_in,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,in_channels)
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
        x += topo_expand
        # print(f'TopoBlockI')
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x,topo_in

class TopoBlock(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample=None, topo_size_in =500, stride=1):
        super(TopoBlock, self).__init__()
        self.topo_enc = nn.Sequential(nn.Linear(topo_size_in,64),
            nn.ReLU(),
            nn.Linear(64,128),
            nn.ReLU(),
            nn.Linear(128,out_channels)
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

#Resnet that also uses Topological images
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

        self.topo_net = nn.Sequential(
            nn.Linear(500,256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            # nn.Linear(128,64),
            # nn.ReLU()
        )

        self.res_net_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU()
        )

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
        # print(x.shape)
        # print(x_2.shape)
        x_2 = x_2.squeeze(1)
        x = torch.cat([x,x_2],dim=1)
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

        self.topo_net = nn.Sequential(
            nn.Linear(500,256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU(),
            # nn.Linear(128,64),
            # nn.ReLU()
        )

        self.res_net_fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256,64),
            nn.ReLU()
        )

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
        # return nn.Sequential(
        #     TopoBlockIN(in_channels, out_channels, identity_downsample=identity_downsample, stride=stride),
        #     TopoBlockIN(out_channels, out_channels)
        # )

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

        x = self.layer1((x,topo))

        x = self.layer2((x,topo))
        x = self.layer3((x,topo))
        x = self.layer4((x,topo))

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

    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

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

            if phase == 'val': # Adjust learning rate based on val loss
                lr_scheduler.step(epoch_loss)

            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


#Function to process the image that is:
# - Change the image to Gray-scale
# - Calculate the topological features
def process_img_topo(data, to_grayscale = transforms.Grayscale(num_output_channels=1)):
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
    # Calculating Landscape
    LS = gd.representations.Landscape(resolution=100)
    L = LS.fit_transform([cubical_complex.persistence_intervals_in_dimension(1)])
    L_t = torch.tensor(L,dtype=torch.float)
    # L_t = L_t[:,:100]
    # print(L_t.shape)
    # raise("Jeszcze jak")
    # L = L[:100]
    # results.append(L)

    return image, L_t

def process_data_topo(dataset):
    data = dataset.data
    #
    # for i in tqdm(data):
    #     img = process_img_topo(i)

    with Pool(8) as pool:
        results = list(tqdm(pool.imap(process_img_topo,data),total=len(data)))
    return results

#Main function
if __name__ == "__main__":

    # train_set_path = 'train_set_land.npy'
    # train_set_target_path = 'train_set_land_target.npy'

    train_set_path = 'train_set_land.pkl'
    train_set_target_path = 'train_set_land_target.pkl'

    # test_set_path = 'test_set_land.npy'
    # test_set_target_path = 'test_set_land_target.npy'

    test_set_path = 'test_set_land.pkl'
    test_set_target_path = 'test_set_land_target.pkl'

    print('Trying to load the data')
    try:
        # raise('still error')
        # train_set = np.load(train_set_path,allow_pickle = True)
        # train_set_target = np.load(train_set_target_path,allow_pickle = True)
        #
        # test_set= np.load(test_set_path,allow_pickle = True)
        # test_set_target = np.load(test_set_target_path,allow_pickle = True)
        #
        with open(train_set_path, 'rb') as f:
            train_set = pickle.load(f)

        with open(train_set_target_path, 'rb') as f:
            train_set_target = pickle.load(f)

        with open(test_set_path, 'rb') as f:
            test_set = pickle.load(f)

        with open(test_set_target_path, 'rb') as f:
            test_set_target = pickle.load(f)

        # train_set = torch.load(train_set_path)
        # train_set_target = torch.load(train_set_target_path)
        #
        # test_set= torch.load(test_set_path)
        # test_set_target = torch.load(test_set_target_path)

        trainset = MyDataset(train_set,train_set_target)
        testset = MyDataset(test_set,test_set_target)

    except:
        print('Failed to load the data, processing the data and saving')
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True)#, transform=transform)
        data_fin_train = process_data_topo(trainset)



        # print(data_fin_train)
        # tmp =  [len(t) for t in data_fin_train]
        # print()
        # print(type(data_fin_train))
        # print(tmp)
        #
        # print(data_fin_train[1])
        # torch.save(train_set_path,torch.tensor(data_fin_train))
        # torch.save(train_set_target_path,torch.tensor(trainset.targets))

        with open(train_set_path, 'wb') as f:
            pickle.dump(data_fin_train, f)

        with open(train_set_target_path, 'wb') as f:
            pickle.dump(trainset.targets, f)

        trainset = MyDataset(data_fin_train,trainset.targets)

        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True)#, transform=transform)
        data_fin_test = process_data_topo(testset)

        # np.save(test_set_path,np.array(data_fin_test))
        # np.save(test_set_target_path,np.array(testset.targets))

        with open(test_set_path, 'wb') as f:
            pickle.dump(data_fin_test, f)

        with open(test_set_target_path, 'wb') as f:
            pickle.dump(testset.targets, f)

        testset = MyDataset(data_fin_test,testset.targets)




    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                              shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64,
                                         shuffle=False, num_workers=2)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ResNet_18_Topo_Block(3,10,device)
    model.to(device)

    print(f'Number of paramters: {count_parameters(model)}')

    epochs = 50
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True)

    model, _ = train_model(model, {"train": trainloader, "val": testloader}, criterion, optimizer, epochs)
