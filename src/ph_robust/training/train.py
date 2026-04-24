import os
import wandb

import torch
from tqdm import tqdm
import time
import copy

from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

from .checkpoint import load_checkpoint, save_checkpoint


def train_model(
    model, dataloaders, criterion, args, optimizer, lr_scheduler, resume_path=None
):
    wandb.init(
        project="ph-robust-img",
        name=args.name if args.name else "unnamed run",
        config=vars(args),
    )

    start_epoch = 0
    best_acc = 0.0
    best_loss = float("inf")  # Initialize this safely

    if resume_path and os.path.isfile(
        resume_path
    ):  # TODO: Fix the load_checkpoint import
        start_epoch, loss = load_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            file_name=resume_path,
        )
    since = time.time()
    val_acc_history = []
    best_model_wts = copy.deepcopy(model.state_dict())

    scaler = GradScaler("cuda")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for epoch in range(start_epoch, args.epochs):
        print("Epoch {}/{}".format(epoch, args.epochs - 1))
        print("-" * 10)

        wandb_metrics = {
            "epoch": epoch,
            "train/loss": None,
            "train/acc": None,
            "val/loss": None,
            "val/acc": None,
        }
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], desc=phase):
                if args.model != "ResNet":
                    x1, x2 = inputs
                    x1 = x1.to(device)
                    x2 = x2.to(device)
                    inputs = (x1, x2)
                    current_batch_size = x1.size(0)
                else:
                    inputs = inputs.to(device)
                    current_batch_size = inputs.size(0)

                labels = labels.to(device)

                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == "train"):
                    with autocast("cuda"):
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    if torch.isnan(loss) or torch.isinf(loss):
                        print(f"WARNING: Loss is {loss.item()} at Epoch {epoch}")

                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer=optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer=optimizer)
                        scaler.update()

                running_loss += loss.item() * current_batch_size
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            # refactor this
            wandb_metrics[f"{phase}/Loss"] = epoch_loss
            wandb_metrics[f"{phase}/Acc"] = epoch_acc

            # This should be outside the phase loop monka Hmm

            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            # Deep copy the model if it's the best
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_loss = epoch_loss  # Capture best loss too
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == "val":  # this should be here (?)
                lr_scheduler.step(epoch_loss)
                val_acc_history.append(epoch_acc)

        wandb_metrics["lr"] = optimizer.param_groups[0]["lr"]
        wandb.log(wandb_metrics, step=epoch)

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            epoch=epoch,
            loss=best_loss,
            file_name="checkpoint.pth",
        )
        if epoch == 10:  # Unfreeze weights after 10 epochs
            model.unfreeze()
            existing_params = set(
                p for group in optimizer.param_groups for p in group["params"]
            )
            new_params = [
                p
                for p in model.parameters()
                if p.requires_grad and p not in existing_params
            ]
            if new_params:
                # backbone_lr = args.lr * 0.1  # 10x smaller than the main LR
                # optimizer.add_param_group({'params': new_params, 'lr': backbone_lr})
                optimizer.add_param_group({"params": new_params, "lr": 3e-5})
                print(
                    f"Added {len(new_params)} newly unfrozen parameter tensors to the optimizer with LR {args.lr}."
                )

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))

    model.load_state_dict(best_model_wts)

    return model, val_acc_history


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }
