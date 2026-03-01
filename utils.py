import os
import wandb
from wandb.sdk.wandb_run import wandb_metric
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import copy
from config.config import args
# from torch.cuda.amp import autocast, GradScaler
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
import random 
import numpy as np

def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False # Set true for final result training 
    torch.backends.cudnn.benchmark = True


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

def accuracy_test(output,target, topk = (1,5)):
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1 , True, True)
        pred = pred.t()

        correct = pred.eq(target.view(1,-1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul(1.0/ batch_size).item())
        return res


@torch.inference_mode()
def test_model(model, dataloader,criterion ):
    print('Testing model')
    model.eval()
    running_loss = 0.0
    running_top1 = 0.0
    running_top5 = 0.0
    total_samples = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for inputs, labels in tqdm(dataloader):

        if args.model != 'ResNet':
            x1,x2 = inputs
            x1 = x1.to(device)
            x2 = x2.to(device)
            inputs = (x1, x2)
            current_batch_size = x1.size(0) 
        else:
            inputs = inputs.to(device)
            current_batch_size = inputs.size(0)
        labels = labels.to(device)
        if not args.ph_test:
            outputs = model(inputs)
        else:
            x1 = torch.nn.functional.interpolate(inputs[0], size= (224,224), mode = 'bilinear', align_corners= False)
            x1_out = model.base_model(x1)
            x2_out = model.topo_net(inputs[1])
            x2_out = x2_out.squeeze(1)
            new_ids = torch.randperm(x2_out.size(0))
            x2_perm_out = x2_out[new_ids]

            fused = torch.cat([x1_out,x2_perm_out],dim=1)
            outputs = model.fc(fused)


        loss = criterion(outputs, labels)
        acc1, acc5 = accuracy_test(output= outputs, target = labels, topk=(1,5))

        # Statistics
        running_loss += loss.item() * current_batch_size
        running_top1 += acc1 * current_batch_size
        running_top5 += acc5 * current_batch_size
        total_samples += current_batch_size


    avg_loss = running_loss / total_samples
    avg_top1 = running_top1 / total_samples
    avg_top5 = running_top5 / total_samples
    print('Test Results:\nLoss: {:.4f}\n Top-1: {:.4f}\nTop-5: {:.4f}'.format(avg_loss, avg_top1, avg_top5))
    return avg_loss, avg_top1, avg_top5

# @torch.inference_mode()
# def test_model(model, dataloader,criterion ):
#     print('Testing model')
#     model.eval()
#     running_loss = 0.0
#     running_corrects = 0
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
#     for inputs, labels in tqdm(dataloader):
#
#         if args.model != 'ResNet':
#             x1,x2 = inputs
#             x1 = x1.to(device)
#             x2 = x2.to(device)
#             inputs = (x1, x2)
#             current_batch_size = x1.size(0) 
#         else:
#             inputs = inputs.to(device)
#             current_batch_size = inputs.size(0)
#         labels = labels.to(device)
#
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         _, preds = torch.max(outputs, 1)
#
#         # Statistics
#         running_loss += loss.item() * current_batch_size
#         running_corrects += torch.sum(preds == labels.data).item()
#
#
#     total_loss = running_loss / len(dataloader.dataset)
#     total_acc = running_corrects / len(dataloader.dataset)
#     print('{} Loss: {:.4f} Acc: {:.4f}'.format('Test', total_loss, total_acc))
#


result_file = 'results.csv'

def train_model(model, dataloaders, criterion, args,  resume_path=None):
    wandb.init(
        project = "ph-robust-img",
        name = args.name if args.name else "unnamed run",
        config=vars(args)
    )
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
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())

    scaler = GradScaler('cuda')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for epoch in range(start_epoch, args.epochs):
        print('Epoch {}/{}'.format(epoch, args.epochs - 1))
        print('-' * 10)
        
        wandb_metrics = {
            "epoch": epoch,
            "train/loss": None,
            "train/acc": None,
            "val/loss":None,
            "val/acc":None,
        }
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                
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
                    with autocast('cuda'):
                        outputs = model(inputs)
                    
                    loss = criterion(outputs, labels)
                    
                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"WARNING: Loss is {loss.item()} at Epoch {epoch}")

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer = optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm= 1.0)
                        scaler.step(optimizer = optimizer)
                        scaler.update()

                running_loss += loss.item() * current_batch_size
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            # refactor this 
            wandb_metrics[f"{phase}/Loss"] = epoch_loss
            wandb_metrics[f"{phase}/Acc"] = epoch_acc
                
            # This should be outside the phase loop monka Hmm 

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # Deep copy the model if it's the best
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_loss = epoch_loss # Capture best loss too
                best_model_wts = copy.deepcopy(model.state_dict())
                
            if phase == 'val': #this should be here (?)
                lr_scheduler.step(epoch_loss)
                val_acc_history.append(epoch_acc)
        
        wandb_metrics['lr'] = optimizer.param_groups[0]['lr']
        wandb.log(wandb_metrics,step = epoch)

        save_checkpoint(model=model, optimizer=optimizer, scheduler=lr_scheduler,epoch=epoch,loss=best_loss, file_name="checkpoint.pth")
        if epoch == 10: #Unfreeze weights after 10 epochs
            model.unfreeze()
            existing_params = set(p for group in optimizer.param_groups for p in group['params'])
            new_params = [p for p in model.parameters() if p.requires_grad and p not in existing_params]
            if new_params:
                backbone_lr = args.lr * 0.1  # 10x smaller than the main LR
                optimizer.add_param_group({'params': new_params, 'lr': backbone_lr})
                print(f"Added {len(new_params)} newly unfrozen parameter tensors to the optimizer with LR {backbone_lr}.")

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    
    return model, val_acc_history


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



