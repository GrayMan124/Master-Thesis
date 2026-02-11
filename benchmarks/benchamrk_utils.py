import torch 
import torch.nn as nn
from autoattack import AutoAttack
from torchvision import transforms
import numpy as np

class ModelWrapper(nn.Module):
    
    def __init__(self,model, topo_func, device):
        super().__init__()
        self.model = model 
        self.topo_func = topo_func
        self.device = device

        self.normalize = transforms.Normalize(
                mean=[0.485,0.456,0.406],
            std=[0.229,0.224,0.225]
        )
    
    def forward(self, x):
        x_norm = self.normalize(x)
        x_cpu = x.detach().mul(255).type(torch.uint8).permute(0,2,3,1).cpu().numpy()

        topo_features = self.topo_func(x_cpu)
        topo_features = topo_features.to(self.device)

        return self.model((x_norm, topo_features))


def run_auto_attack(model, x_test,y_test,log_path, eps = 8/255):
    print(f"Stargint AutoAttack testing")
    
    adversary = AutoAttack(
            model,
            norm= 'Linf',
            eps= eps,
            version = 'custom',
            attacks_to_run = ['apgd-ce','apgd-t'],#['square'],
            log_path = log_path
    )

    adversary.apgd.n_restarts = 1

    with torch.no_grad():
        results  = adversary.run_standard_evaluation(x_test,y_test)

    return results


class CleanImageDatasetLoader(torch.utils.data.DataLoader):
    
    def __init__(self, original_dataset):
        self.dataset = original_dataset
        self.inv_normalize = transforms.Compose([
            transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
            transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Your dataset returns ((img_tensor, topo), label)
        (img_tensor, _), label = self.dataset[idx]
        
        # Revert normalization: Model input needs to be [0,1] for AutoAttack
        img_raw = self.inv_normalize(img_tensor)
        
        return img_raw, label
