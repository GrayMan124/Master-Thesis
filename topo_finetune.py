#Update, version from: 24-11-2025 
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
import json
from torchvision.datasets import Caltech256 , Caltech101
from torchvision.models import resnet50 

from topoTransform import get_topo_DS
from models.PI_finetune import PIFineTuneModel

from utils import test_model, train_model, count_parameters
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
    hidden_size = 256 



def collate_fn(batch):
  return {
      'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
      'labels': torch.tensor([x['labels'] for x in batch])
    }




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

    args.tbs == 'large'
    print(args)
    
    train_loader, val_loader = get_topo_DS(dir_path= './data/', dataset= Caltech256)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    base_model = resnet50(weights = "IMAGENET1K_V2")
    
    model =  PIFineTuneModel(base_model = base_model, image_channels = 3, num_classes = 257, device= device)
    model.to(device)

    print(f'Currenttly running model: {args.model} with {args.tv} ')
    print(f'Number of paramters: {count_parameters(model)}')

    epochs = args.epochs
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)#, verbose=True)

    model, _ = train_model(model, {"train": train_loader, "val": val_loader}, criterion, optimizer, epochs)
    # test_model(model,testloader,criterion, optimizer)
    #
    # if args.sm:
    #     print('Saving model')
    #     torch.save(model.state_dict(), model_saving_path)
