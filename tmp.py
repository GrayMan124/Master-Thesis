from datasets import load_dataset
import torch 
import matplotlib.pyplot as plt
import random
import os 
from dataProcessing.topology.topologicalProcessing import AugmentAndCalculateFeatures 
from config.config import args
from tqdm import tqdm

from dataProcessing.processing import process_data
if __name__ == "__main__":
    ds = load_dataset("zh-plus/tiny-imagenet")
    ds_train = ds['train']
   

    processing_train = AugmentAndCalculateFeatures(train=True, args = args)
    final_images = []
    for idx in tqdm(range(len(ds_train))):
        img,pi_img = processing_train(ds_train[idx]['image'])
        final_images.append(pi_img)

    stack = torch.vstack(final_images)
    print(stack.max())
    print(stack.min())
