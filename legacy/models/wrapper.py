import numpy as np
from torch import nn
from torchvision import transforms


def transform_initial_data(data):

    data = data.numpy()
    data = data * 256
    data = data.astype(np.uint8)
    # output = [Image.fromarray(np.transpose(sample,(1,2,0))) for sample in data ]
    output = [np.transpose(sample,(1,2,0)) for sample in data ]
    return np.array(output)

class ModelWrapper(nn.Module):

    def __init__(self,model, params_path, from_train, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = model 
        self.from_train = from_train
        self.topo_vectorization = get_vector_function() 


    
    def predict(self,x):
        output = self.model(x)
        _, prediction = torch.max(output,1)
        return prediction

    def forward(self,x):
        if x.device != self.model.device:
            x = x.to(self.model.device)
        
        
        normalization = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        current_device = x.device
        
        x_detached = x.detach().cpu()
        numpy_batch = (x_detached.permute(0,2,3,1)*256).numpy().astype(np.uint8)
        topo_features = process_topo_batch(numpy_batch, self.topo_vectorization, self.from_train)

        topo_features = topo_features.to(current_device)
        x = normalization(x)
        return self.model((x,topo_features))

class ResNet_Wrapper(nn.Module):
    def __init__(self, model,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model 
    
    def forward(self,x):
        normalization = transforms.Compose([transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        x = normalization(x)
        return self.model(x)

