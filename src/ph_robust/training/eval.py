import torch
from tqdm import tqdm


def accuracy_test(output, target, topk=(1, 5)):
    with torch.inference_mode():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()

        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul(1.0 / batch_size).item())
        return res


@torch.inference_mode()
def test_model(model, dataloader, criterion, cfg):
    print("Testing model")
    model.eval()
    running_loss = 0.0
    running_top1 = 0.0
    running_top5 = 0.0
    total_samples = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for inputs, labels in tqdm(dataloader):
        if cfg.model.kind != "ResNet":
            x1, x2 = inputs
            x1 = x1.to(device)
            x2 = x2.to(device)
            inputs = (x1, x2)
            current_batch_size = x1.size(0)
        else:
            inputs = inputs.to(device)
            current_batch_size = inputs.size(0)
        labels = labels.to(device)

        if not cfg.eval.ph_test:  # TODO: add this to the configs
            outputs = model(inputs)
        else:
            x1 = torch.nn.functional.interpolate(
                inputs[0], size=(224, 224), mode="bilinear", align_corners=False
            )
            x1_out = model.base_model(x1)
            x2_out = model.topo_net(inputs[1])
            x2_out = x2_out.squeeze(1)
            new_ids = torch.randperm(x2_out.size(0))
            x2_perm_out = x2_out[new_ids]

            fused = torch.cat([x1_out, x2_perm_out], dim=1)
            outputs = model.fc(fused)

        loss = criterion(outputs, labels)
        acc1, acc5 = accuracy_test(output=outputs, target=labels, topk=(1, 5))

        # Statistics
        running_loss += loss.item() * current_batch_size
        running_top1 += acc1 * current_batch_size
        running_top5 += acc5 * current_batch_size
        total_samples += current_batch_size

    avg_loss = running_loss / total_samples
    avg_top1 = running_top1 / total_samples
    avg_top5 = running_top5 / total_samples
    print(
        "Test Results:\nLoss: {:.4f}\n Top-1: {:.4f}\nTop-5: {:.4f}".format(
            avg_loss, avg_top1, avg_top5
        )
    )
    return avg_loss, avg_top1, avg_top5
