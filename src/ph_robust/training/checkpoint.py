import os
import torch


def save_checkpoint(
    model, optimizer, scheduler, epoch, loss, file_name="checkpoint.pth"
):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "loss": loss,
    }
    torch.save(checkpoint, file_name)
    print(f"Saved Checkpoint at {file_name}")


def load_checkpoint(model, optimizer, scheduler, file_name="checkpoint.pth"):
    if not os.path.isfile(file_name):
        raise FileNotFoundError("Failed to load checkpoint -> file doesn't exist")

    checkpoint = torch.load(file_name, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler and checkpoint["scheduler_state_dict"]:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    loss = checkpoint["loss"]

    print(
        f"Resumed training from {file_name} checkpoint\nResuming from epoch {start_epoch}"
    )

    return start_epoch, loss
