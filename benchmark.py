from robustbench.data import load_cifar10c, load_cifar10
import torch 
from robustbench.utils import load_model, clean_accuracy
import time

from config import args
from data_processing import process_data_topo
from models.ResNetPIBlock import ResNet_18_PIBlock
# from models.ResNetTopoBlock import *
import torch.nn as nn 
from main import MyDataset
from PIL import Image
import numpy as np
from tqdm import tqdm

def transform_initial_data(data):
    data = data.numpy()
    data = data * 256
    data = data.astype(np.uint8)
    # output = [Image.fromarray(np.transpose(sample,(1,2,0))) for sample in data ]
    output = [np.transpose(sample,(1,2,0)) for sample in data ]
    return np.array(output)

class ModelWrapper(nn.Module):

    def __init__(self,model, params_path,  *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = model 
        self.load_paramas(params_path)

    def load_paramas(self,params_path):
        self.model.load_state_dict(torch.load(params_path, map_location = self.model.device))
    
    def predict(self,x):
        output = self.model(x)
        print(output,flush=True)
        prediction = torch.argmax(output)
        print(prediction)
        return prediction

    def forward(self,x):
        return self.model(x)

def run_test(model,loader,device):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in tqdm(loader,desc="Testing"):
            x1,x2 = data
            x1 = x1.to(device)
            x2 = x2.to(device)
            labels = labels.to(device)

            inputs = (x1,x2)
            outputs = model(inputs)

            _, pred_class = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (pred_class == labels).sum().item()
    
    accuracy = 100* correct / total

    print("\n --- Test Complete ---")
    print(f"Total Samples: {total}")
    print(f"Total correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}")

    return accuracy

if __name__ == '__main__':
    all_corruption_types = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate',
        'jpeg_compression'
        ]
    # x,y = load_cifar10c(n_examples=10000,corruptions=all_corruption_types,severity=1)
    x,y = load_cifar10c(n_examples=10000)

    # x,y = load_cifar10()
    print(args)
    x_np = x.numpy()
    y_np = y.numpy()
    x_train = transform_initial_data(x)
    args.cores = 1

    data = MyDataset(x_train,y_np)
    processed_data, _  = process_data_topo(data)
    data_set = MyDataset(processed_data,y)
    trainloader = torch.utils.data.DataLoader(data_set, batch_size=32,shuffle=True, num_workers=1)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(f'Using device: {device}')

    model = ResNet_18_PIBlock(3,10,device)
    model_wrapped = ModelWrapper(model,"./models/PI_IMG_19_param.pkl")
    model_wrapped.to(device)

    run_test(model_wrapped,trainloader,device)

    x_test, y_test = load_cifar10c(n_examples=10000)   

    for model_name in ['Standard', 'Engstrom2019Robustness', 'Rice2020Overfitting', 'Carmon2019Unlabeled']:
        model = load_model(model_name)
        acc = clean_accuracy(model, x_test, y_test)
        print('Model: {}, CIFAR-10-C accuracy: {:.1%}'.format(model_name, acc))


