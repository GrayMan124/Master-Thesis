import torch 
import torch.nn as nn
from autoattack import AutoAttack
from torchvision import transforms
import numpy as np

def process_topo_batch(numpy_batch, topo_vectorization):

    
    results = []
    for sample in numpy_batch:
        results.append(topo_vectorization(sample))
    
    # topo_features = [item[1] for item in results]
    tensor_topo_data = torch.stack(results,dim=0)

    return tensor_topo_data 
class ModelWrapper(nn.Module):

    def __init__(self,model, topo_func, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = model 
        self.topo_vectorization = topo_func 

        self.final_image_transform_val = transforms.Compose([
            transforms.ToTensor(), # Converts (H, W, C) NumPy to (C, H, W) Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        self.pi_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=0 , std = 1247.8710)
            ])
    
    def predict(self,x):
        output = self.model(x)
        _, prediction = torch.max(output,1)
        return prediction

    def forward(self,x):
        if x.device != self.model.device:
            x = x.to(self.model.device)
        
        
        current_device = x.device
        
        x_detached = x.detach().cpu()
        numpy_batch = (x_detached.permute(0,2,3,1)*255).numpy().astype(np.uint8)
        # topo_features = process_topo_batch(numpy_batch, self.topo_vectorization, self.from_train)
        topo_features = process_topo_batch(numpy_batch=numpy_batch, topo_vectorization=self.topo_vectorization) 
        topo_features = self.pi_transform(topo_features)
        topo_features = topo_features.to(current_device)

        x = self.final_image_transform_val(x)
        
        return self.model((x,topo_features))

def run_auto_attack(model, x_test, y_test, log_path, eps = 8/255):
    print(f"Stargint AutoAttack testing")
    
    adversary = AutoAttack(
            model,
            norm= 'Linf',
            eps= eps,
            version = 'custom',
            attacks_to_run = ['apgd-ce','apgd-t'],#['square'],
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
