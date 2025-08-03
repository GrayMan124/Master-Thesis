from robustbench.data import load_cifar10c, load_cifar10
import torch 
from robustbench.utils import load_model, clean_accuracy
from robustbench.eval import benchmark
import time

from config import args
from data_processing import *
import torch.nn as nn 
from main import MyDataset
from PIL import Image
import numpy as np
from tqdm import tqdm

from models.ResNet import *
from data_processing import *
from models.TopoResNet import *
from models.ResNetPIBlock import *
from models.ResNetTopo2Dim import *
from models.ResNetTopoBlock import *
from models.TopoResNet import *
from models.TopoResNetPI import * 
from models.ResNet import *
from autoattack import AutoAttack

def get_vector_function():
    if args.tv == 'land':
        return process_img_topo_land
    elif args.tv == 'bc':
        return process_img_topo_betti_curve
    
    elif args.tv == 'pi_v':
        return process_img_topo_pi_v

    elif args.tv == 'pi_img':
        return process_img_topo_pi_img

    elif args.tv == 'silh':
        return process_img_topo_silh
    else:
        raise('Error - invalid topological vectorization method')


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

def load_paramas(self,params_path):
    model.load_state_dict(torch.load(params_path, map_location = self.model.device))

def run_test(model,x_test,y_test,test_name, eps):
    model.eval()

    # with torch.no_grad():
    log_file_path = f"results/adv_eval_inf_square/log_RT_{test_name}.txt"

    adversary = AutoAttack(model, norm='Linf', eps=eps, version='custom', attacks_to_run=['apgd-ce'],log_path=log_file_path)
    adversary.apgd.n_restarts = 1
    adversary.run_standard_evaluation(x_test,y_test)
    print("\n --- Test Complete ---")

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model_saving_path = 'models/saved_models/'+ args.name + args.model + '_' + args.tv + '_' + str(args.lr) + '_' + str(args.res) + '_' + str(args.seed) + '_' + str(args.topodim) + '_' + args.bw +'.pkl'
    # model_saving_path =  'models/PI_IMG_19_param.pkl'
    if args.model =='ResNet':
        model = ResNet_18(3,10)
        model_saving_path = 'models/saved_models/resnet_aug_05_nt.pkl'
    elif args.model == 'TR':
        model = ResNet_18_Topo(3,10,device)
    elif args.model =='TBR':
        model = ResNet_18_Topo_Block(3,10,device)
    elif args.model =='TR_2dim':
        model = ResNet_18_Topo_2dim(3,10,device)
    elif args.model =='TR_img':
        if args.tv != 'pi_img':
            raise('Wrong vectorization fo TimgRes')
        model = ResNet_18_TopoPI(3,10,device)
    elif args.model =='TBR_img':
        if args.tv != 'pi_img':
            raise('Wrong vectorization fo TimgRes')
        model = ResNet_18_PIBlock(3,10,device)
    else:
        print('Error - Incorrect model option')
        raise('Error - Incorrect model option')
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.device = device 
    model.load_state_dict(torch.load(model_saving_path, map_location = device))
    model.eval()

    if args.model != 'ResNet': 
        _, from_train = process_data_topo(trainset)
        model_wrapped = ModelWrapper(model,model_saving_path, from_train)
    else:
        model_wrapped = ResNet_Wrapper(model)
    
    results_to_json = []
    x_test, y_test = load_cifar10(n_examples=1000)    
    model_wrapped = model_wrapped.to(device)
    model_wrapped.eval()
    test_eps = [ 1/255, 2/255, 4/255, 8/255]
    for eps in test_eps:
        run_test(model_wrapped,x_test,y_test,args.name + str(eps*255),eps = eps) 
    # with open(f'./results/benchmark_cifar10c/{args.name}.json','w') as file:
        # json.dump(results_to_json,file)    

