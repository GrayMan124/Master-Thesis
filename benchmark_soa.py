from robustbench.data import load_cifar10c, load_cifar10
import torch 
from robustbench.utils import load_model, clean_accuracy
import time

from autoattack import AutoAttack
from config import args
from data_processing import process_data_topo
from models.ResNetPIBlock import ResNet_18_PIBlock
# from models.ResNetTopoBlock import *
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







def run_test(model,loader,device,test_name):
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in tqdm(loader,desc="Testing"):
            if args.model != 'ResNet':
                x1,x2 = data
                x1 = x1.to(device)
                x2 = x2.to(device)
                labels = labels.to(device)

                inputs = (x1,x2)
            else:
                inputs = data 
            outputs = model(inputs)

            _, pred_class = torch.max(outputs, 1)

            total += labels.size(0)
            if args.model == 'ResNet':
                labels = labels.to('cpu')
                pred_class = pred_class.to('cpu')
            correct += (pred_class == labels).sum().item()
    
    accuracy = 100* correct / total
    
    print("\n --- Test Complete ---")
    # print(f"Total Samples: {total}")
    # print(f"Total correct: {correct}")
    print(f"Accuracy: {accuracy:.2f}")
    json_dict = {
        'test_name': test_name,
        'accuracy': accuracy,
        'total_samples': total,
        'correct': correct
    } 
    
    return json_dict    

if __name__ == '__main__':
    all_corruption_types = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
        'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate',
        'jpeg_compression'
        ]
    # x,y = load_cifar10c(n_examples=10000,corruptions=all_corruption_types,severity=1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # model_saving_path = 'models/saved_models/'+ args.name + args.model + '_' + args.tv + '_' + str(args.lr) + '_' + str(args.res) + '_' + str(args.seed) + '_' + str(args.topodim) + '_' + args.bw +'.pkl'
    # if args.model =='ResNet':
    #     model = ResNet_18(3,10)
    #     model_saving_path = 'models/saved_models/resnet_aug_05_nt.pkl'
    # elif args.model == 'TR':
    #     model = ResNet_18_Topo(3,10,device)
    # elif args.model =='TBR':
    #     model = ResNet_18_Topo_Block(3,10,device)
    # elif args.model =='TR_2dim':
    #     model = ResNet_18_Topo_2dim(3,10,device)
    # elif args.model =='TR_img':
    #     if args.tv != 'pi_img':
    #         raise('Wrong vectorization fo TimgRes')
    #     model = ResNet_18_TopoPI(3,10,device)
    # elif args.model =='TBR_img':
    #     if args.tv != 'pi_img':
    #         raise('Wrong vectorization fo TimgRes')
    #     model = ResNet_18_PIBlock(3,10,device)
    # else:
    #     print('Error - Incorrect model option')
    #     raise('Error - Incorrect model option')
    # # return
    # if args.model != 'ResNet':
    #     model_wrapped = ModelWrapper(model,model_saving_path)
    # else:
    #     print('Loading ResNEt model')
    #     model.load_state_dict(torch.load(model_saving_path, map_location =device))
    #     model_wrapped = model
    # results_to_json = []
    # for corruption_type in all_corruption_types:
        
    #     x,y = load_cifar10c(n_examples=10000,corruptions=[corruption_type])
    #     print(f"Running: {corruption_type}")
    #     if args.model != 'ResNet':
    #         x_np = x.numpy()
    #         y_np = y.numpy()
    #         x_train = transform_initial_data(x)
    #         args.cores = 1

    #         data = MyDataset(x_train,y_np)
    #         processed_data, _  = process_data_topo(data, from_train=model_wrapped.from_train)
        
    #         data_set = MyDataset(processed_data,y)
    #         data_loader = torch.utils.data.DataLoader(data_set, batch_size=32,shuffle=True, num_workers=1)
    #     else:
    #         x = x.to(device)
    #         data_loader= torch.utils.data.DataLoader(MyDataset(x,y),batch_size=32,shuffle=True, num_workers=0)
    #     # print(f'Using device: {device}')

    #     model_wrapped.to(device)

    #     results = run_test(model_wrapped,data_loader,device,corruption_type)
    #     results_to_json.append(results)

    # with open(f'./results/benchmark_cifar10c/{args.name}.json','w') as file:
    #     json.dump(results_to_json,file)    



    for model_name in ['Addepalli2022Efficient_RN18', 'Sehwag2021Proxy_R18',
                   'Rade2021Helper_R18_ddpm']:
        print(f"Running model: {model_name}")
        # x_test, y_test = load_cifar10c(n_examples=10000, corruptions = all_corruption_types)
        model = load_model(model_name, dataset='cifar10')
        x_test, y_test = load_cifar10(n_examples=1000)    
        # acc = clean_accuracy(model, x_test, y_test)
        # print(f'Model: {model_name}, CIFAR-10-C accuracy: {np.mean(acc):.1%}')

        log_file_path = f"./results/{model_name}_linf.txt"
        adversary = AutoAttack(model, norm='Linf', eps=8/255, version='custom', attacks_to_run=['apgd-ce','apgd-t','square'],log_path=log_file_path)
        adversary.apgd.n_restarts = 1
        adversary.run_standard_evaluation(x_test,y_test)
# Addepalli2022Efficient_RN18, Sehwag2021Proxy_R18, Modas2021PRIMEResNet18 - models to compare the benchmark to 