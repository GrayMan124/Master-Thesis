from topoTransform import process_PI, AugmentAndCalculateFeatures
from tqdm import tqdm
import torch
from datasets import load_dataset
import os
import json

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["HF_HOME"] = "/mnt/sam/pi_data/"

transform = AugmentAndCalculateFeatures(train= True) 

def get_min_max(data_set, save_path):
    # ds_train = Caltech256(root = './data/', transform = transform, download= False)
    pi_img_list = []
    data_set_len = len(data_set) 
    for i in tqdm(range(data_set_len)):

        tmp = ds_train[i]['image']
        label = ds_train[i]['label']
        _, pi_img = transform(tmp)
        pi_img_list.append(pi_img)
    
    full_data = torch.vstack(pi_img_list)
    output_dict = {"MEAN":full_data.mean(), 
                   "STD":full_data.std(),
                    "MAX": full_data.max(),
                    "MIN":full_data.min()
                   }
    with open(save_path, 'wb') as file:
        json.dump(output_dict, file)

    print("Getting dataset-meta data finished")
