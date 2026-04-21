from robustbench.data import load_cifar10c, load_cifar10
import torch 
from robustbench.utils import load_model, clean_accuracy
from robustbench.eval import benchmark

from config import args
from data_processing import *

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
    test_eps = [0.1,0.2,0.3,0.4,0.5 ]
    for eps in test_eps:
        run_test(model_wrapped,x_test,y_test,args.name + str(eps*10),eps = eps) 
    # with open(f'./results/benchmark_cifar10c/{args.name}.json','w') as file:
        # json.dump(results_to_json,file)    

