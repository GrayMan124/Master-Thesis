import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
import copy
from config.config import args




def count_parameters(model):
    """Counts the total, trainable, and non-trainable parameters 
    and estimates their memory usage."""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    
    print(f"--- Parameter Count ---")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable params: {non_trainable_params:,}")
    
    if total_params > 0:
        param_dtype = next(model.parameters()).dtype
        
        if param_dtype == torch.float32:
            bytes_per_param = 4
        elif param_dtype == torch.float16 or param_dtype == torch.bfloat16:
            bytes_per_param = 2
        elif param_dtype == torch.float64:
            bytes_per_param = 8
        else:
            try:
                bytes_per_param = torch.finfo(param_dtype).bits // 8
            except TypeError:
                try:
                    bytes_per_param = torch.iinfo(param_dtype).bits // 8
                except TypeError:
                    bytes_per_param = 4 # Default assumption
                    
        total_memory_bytes = total_params * bytes_per_param
        total_memory_mb = total_memory_bytes / (1024 ** 2)
        
        print(f"\n--- Memory Usage (Model Weights Only) ---")
        print(f"Assuming all params are: {param_dtype}")
        print(f"Bytes per parameter:   {bytes_per_param}")
        print(f"Total memory (MB):     {total_memory_mb:.2f} MB")
        
    else:
        print("\nModel has no parameters.")

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
        optimizer.zero_grad() 

        with torch.set_grad_enabled(False): 

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

        # Statistics
        running_loss += loss.item() * inputs[0].size(0)
        running_corrects += torch.sum(preds == labels.data)


    total_loss = running_loss / len(dataloader.dataset)
    total_acc = running_corrects.double() / len(dataloader.dataset)
    print('{} Loss: {:.4f} Acc: {:.4f}'.format('Test', total_loss, total_acc))



result_file = 'results.csv'

def train_model(model, dataloaders, criterion, args, tensor_board_path, resume_path=None):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

    start_epoch = 0
    best_acc = 0.0
    best_loss = float('inf') # Initialize this safely
    
    if resume_path and os.path.isfile(resume_path):
        start_epoch, loss = load_checkpoint(model=model,
                                            optimizer=optimizer,
                                            scheduler=lr_scheduler,
                                            file_name=resume_path) 
    writer = SummaryWriter(log_dir=tensor_board_path)
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(start_epoch, args.epochs):
        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                
                if args.model != 'ResNet':
                    x1, x2 = inputs
                    x1 = x1.to(device)
                    x2 = x2.to(device)
                    inputs = (x1, x2)
                    current_batch_size = x1.size(0) 
                else:
                    inputs = inputs.to(device)
                    current_batch_size = inputs.size(0)

                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * current_batch_size
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            if phase == 'train':
                writer.add_scalar("Loss/train", epoch_loss, epoch)
                writer.add_scalar("Accuracy/train", epoch_acc, epoch)
            else:
                writer.add_scalar("Loss/Val", epoch_loss, epoch)
                writer.add_scalar("Accuracy/Val", epoch_acc, epoch)
                
                lr_scheduler.step(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Deep copy the model if it's the best
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_loss = epoch_loss # Capture best loss too
                best_model_wts = copy.deepcopy(model.state_dict())
                
                save_checkpoint(model=model, optimizer=optimizer, scheduler=lr_scheduler,epoch=epoch,loss=best_loss, file_name="best_checkpoint.pth")

            if phase == 'val':
                val_acc_history.append(epoch_acc)

        
            save_checkpoint(model=model, optimizer=optimizer, scheduler=lr_scheduler,epoch=epoch,loss=best_loss, file_name="checkpoint.pth")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    
    return model, val_acc_history

#Training function
# def train_model(model, dataloaders, criterion, optimizer, args, tensor_board_path):
#     print('Traning model')
#
#     writer = SummaryWriter(log_dir = tensor_board_path)
#     since = time.time()
#     val_acc_history = []
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_acc = 0.0
#
#     optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
#     lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)#, verbose=True)
#
#     train_acc = 0.0
#     train_loss = 10
#     val_loss = 0.0
#
#     num_epochs = args.epochs
#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#         print('-' * 10)
#
#         for phase in ['train', 'val']: # Each epoch has a training and validation phase
#             if phase == 'train':
#                 model.train()  # Set model to training mode
#             else:
#                 model.eval()   # Set model to evaluate mode
#
#             running_loss = 0.0
#             running_corrects = 0
#
#             for inputs, labels in tqdm(dataloaders[phase]): # Iterate over data
#                 device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#                 # inputs = transforms.functional.resize(inputs, (112, 112))
#                 if args.model != 'ResNet':
#                     x1,x2 = inputs
#                     x1.to(device)
#                     x2.to(device)
#                     inputs = (x1,x2)
#                 else:
#                     inputs = inputs.to(device)
#                 labels = labels.to(device)
#
#                 optimizer.zero_grad() # Zero the parameter gradients
#
#                 with torch.set_grad_enabled(phase == 'train'): # Forward. Track history if only in train
#
#                     outputs = model(inputs)
#                     loss = criterion(outputs, labels)
#                     _, preds = torch.max(outputs, 1)
#
#                     if phase == 'train': # Backward + optimize only if in training phase
#                         loss.backward()
#                         optimizer.step()
#
#                 if args.model != 'ResNet':
#                     # batch_size = inputs.size(0)
#                     batch_size = inputs[0].size(0)
#                 else:
#                     batch_size = inputs[0].size(0)
#                 # Statistics
#                 running_loss += loss.item() * batch_size 
#                 running_corrects += torch.sum(preds == labels.data)
#
#             epoch_loss = running_loss / len(dataloaders[phase].dataset)
#             epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
#
#             if phase =='val':
#                 writer.add_scalar("Loss/train", epoch_loss, epoch)
#                 writer.add_scalar("Accuracy/train", epoch_acc, epoch)
#                 if epoch_loss < val_loss:
#                     val_loss = epoch_loss
#             else:
#                 if epoch_loss < train_loss:
#                     train_loss = epoch_loss
#                 if epoch_acc > train_acc:
#                     train_acc = epoch_acc
#                 writer.add_scalar("Loss/Val", epoch_loss, epoch)
#                 writer.add_scalar("Accuracy/Val", epoch_acc, epoch)
#
#             if phase == 'val': # Adjust learning rate based on val loss
#                 lr_scheduler.step(epoch_loss)
#
#
#
#             print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
#
#             # deep copy the model
#             if phase == 'val' and epoch_acc > best_acc:
#                 best_acc = epoch_acc
#                 best_loss = val_loss
#                 best_model_wts = copy.deepcopy(model.state_dict())
#             if phase == 'val':
#                 val_acc_history.append(epoch_acc)
#
#         print()
#
#     writer.flush()
#     writer.close()
#     time_elapsed = time.time() - since
#     print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
#     print('Best val Acc: {:4f}'.format(best_acc))
#
#     results_dict = {
#         "Model_Name":args.name,
#         "Topology_Vect":args.tv,
#         "Model_arch":args.model,
#         "LR":args.lr,
#         "TopoDim_concat":args.topodim_concat,
#         "TopoDim": args.topodim,
#         "Config_file":args.config,
#         "Augmentation":args.aug,
#         "Augmentation_Type":args.aug_type,
#         "Train_loss":train_loss,
#         "Train_Acc":train_acc*100,
#         "Val_Loss":best_loss,
#         "Val_Acc":best_acc*100
#     }
#
#     result_df = pd.DataFrame([results_dict])
#
#     try:
#         df = pd.read_csv(result_file)
#         new_df = pd.concat([df,result_df],ignore_index = True)
#         new_df.to_csv(result_file,index= False)
#     except:
#         result_df.to_csv(result_file,index=False) 
#
#     model.load_state_dict(best_model_wts)
#     return model, val_acc_history
#

def save_checkpoint(model, optimizer, scheduler, epoch, loss, file_name="checkpoint.pth"):
    checkpoint = {
        'epoch':epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'loss': loss
    }
    torch.save(checkpoint, file_name)
    print(f"Saved Checkpoint at {file_name}")

def load_checkpoint(model, optimizer, scheduler, file_name="checkpoint.pth"):
    if not os.path.isfile(file_name):
        raise FileNotFoundError("Failed to load checkpoint -> file doesn't exist")

    checkpoint = torch.load(file_name, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict']) 

    optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 
    
    if scheduler and checkpoint['scheduler_state_dict']:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict']) 
    start_epoch = checkpoint['epoch'] + 1
    loss = checkpoint['loss']

    print(f"Resumed training from {file_name} checkpoint\nResuming from epoch {start_epoch}")

    return start_epoch, loss 

def collate_fn(batch):
  return {
      'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
      'labels': torch.tensor([x['labels'] for x in batch])
    }


def layer_from_config(layer_config):
    layer_type = layer_config["type"]
    params = {k: v for k, v in layer_config.items() if k != "type"}

    if hasattr(nn, layer_type):
        return getattr(nn, layer_type)(**params)
    else:
        raise ValueError(f"Layer type {layer_type} is not supported.")



