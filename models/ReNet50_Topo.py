import os
import torch
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torchvision.transforms as transforms
import torch.nn as nn
import json


def layer_from_config(layer_config):
    layer_type = layer_config["type"]
    params = {k: v for k, v in layer_config.items() if k != "type"}

    # Dynamically instantiate the layer
    if hasattr(nn, layer_type):
        return getattr(nn, layer_type)(**params)
    else:
        raise ValueError(f"Layer type {layer_type} is not supported.")

class TopoIMG_transModel(nn.Module): #This model is specificaly designed to transform the input of 1x64x64 into 3x32x32 (usable in topoblock configugartion)

    def __init__(self, args):
        super(TopoIMG_transModel,self).__init__()

        #NOTE: For this implementation, I could stick with the base image size 1x64x64 since the overall images are the same size, but let's keep it for now 

        if args.topodim_concat:
            in_ch = 2
        else:
            in_ch = 1
        
        
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


class PIBlock(nn.Module):
    def __init__(self, in_channels, out_channels, inter_channels, args, identity_downsample=None, stride=1):
        super(PIBlock, self).__init__()
        self.args = args 
        #Base ResNet Block
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=1, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(inter_channels)
        self.conv2 = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.conv3 = nn.Conv2d(inter_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

        self.identity_downsample_t = identity_downsample

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

    def forward(self, x):
        x,topo = x
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
        x += identity_t

        x = self.relu(x)
        if self.args.tb_add_t:
            topo += identity_t

        #Adding the
        if self.args.tb_add_x:
            topo+=identity

        x = self.relu(x)

        return x, topo


#Base ResNet-50
class PH_ResNet50(nn.Module):

    def __init__(self, image_channels, num_classes,args):


        super(PH_ResNet50, self).__init__()
        self.args = args
        self.topo_embed = TopoIMG_transModel(args = self.args)
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
        # self.fc = nn.Linear(2048, args.hidden_size)
        self.avgpool_t = nn.AdaptiveAvgPool2d((1, 1))
        
        self.res_net_fc = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024,args.hidden_size),
                nn.ReLU()
            )

        self.res_net_fc_topo = nn.Sequential(
                nn.Linear(512, args.hidden_size),
                nn.ReLU()
            )
       
        self.fc = nn.Sequential(nn.Linear(args.hidden_size * 2 ,1024),
                nn.ReLU(),
                nn.Linear(1024,512),
                nn.ReLU(),
                nn.Linear(512,num_classes),
                nn.Softmax()
            )


    def __make_layer(self, in_channels, inter_channels, out_channels, stride, num_blocks):

        identity_downsample = None
        if stride != 1 or in_channels != out_channels:
            identity_downsample = self.identity_downsample(in_channels, out_channels, stride = stride )
        
        layers = [PIBlock(in_channels,inter_channels, out_channels, args = self.args, identity_downsample=identity_downsample, stride=stride)] # Iintial block has downsampling
        for _ in range(num_blocks - 1):
            layers.append(PIBlock(in_channels = out_channels, inter_channels = inter_channels, out_channels = out_channels, args=self.args))
        
        return nn.Sequential(*layers)         

    def forward(self, x):

        # x = transforms.functional.resize(x, (112, 112))
        x,topo = x 
        topo = self.topo_embed(topo)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x,x_topo = self.layer1((x,topo))
        x, x_topo = self.layer2((x,x_topo))
        x, x_topo = self.layer3((x,x_topo))
        x, x_topo = self.layer4((x,x_topo))

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)


        x_topo = self.avgpool_t(x_topo)
        x_topo = x_topo.view(x_topo.shape[0], -1)
        x_topo = self.res_net_fc_topo(x_topo)

        x = torch.cat([x,x_topo],dim=1)

        x = self.fc(x)

        return x

    def identity_downsample(self, in_channels, out_channels, stride):

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
            # nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),# padding=1),
            nn.BatchNorm2d(out_channels)
        )
