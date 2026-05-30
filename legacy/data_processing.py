import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import pickle
import gudhi as gd
import gudhi.representations
import argparse
from matplotlib import pyplot as plt
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import cv2
from PIL import Image
from torchvision.transforms import v2
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, set_start_method

from config import args


import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

def process_img_topo_land(data, to_grayscale = transforms.Grayscale(num_output_channels=1)): #Processing to Landscapes
    try:
        transform=transforms.Compose(
            [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        if args.bw =='cv2':
            transform_bw=transforms.Compose(
                [transforms.ToTensor(),
                    transforms.Normalize(0.5, 0.5, 0.5)])
            bw_img = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
            img = Image.fromarray(data)
            bw_img = Image.fromarray(bw_img)
            image = transform(img)
            bw_img = transform_bw(bw_img)

        elif args.bw == 'torch':
            pass #TODO

        gray_scale_img = bw_img # to_grayscale(image)
        # calcuating the cubical complex
        cubical_complex = gd.CubicalComplex(dimensions=gray_scale_img.shape, top_dimensional_cells=gray_scale_img.flatten())
        # Calculating persistance
        diag = cubical_complex.persistence()
        # Calculating Landscape
        LS = gd.representations.Landscape(resolution=args.res)

        if args.topodim_concat:
            LS_0 = LS.fit_transform([cubical_complex.persistence_intervals_in_dimension(0)[:-1]])
            L_t_0 = torch.tensor(LS_0,dtype=torch.float)
            LS_1 = LS.fit_transform([cubical_complex.persistence_intervals_in_dimension(1)])
            L_t_1 = torch.tensor(LS_1,dtype=torch.float)
            L_t = torch.cat([L_t_0,L_t_1],dim=1)

        elif args.topodim == 0:
            LS = LS.fit_transform([cubical_complex.persistence_intervals_in_dimension(args.topodim)[:-1]])
            L_t = torch.tensor(LS,dtype=torch.float)

        elif args.topodim == 1:
            LS = LS.fit_transform([cubical_complex.persistence_intervals_in_dimension(args.topodim)])
            L_t = torch.tensor(LS,dtype=torch.float)


        if L_t is None:
            raise('None in the Landscape processing L_T')

        return image, L_t

    except Exception as e:
        print(f"Error with item: {e}")
        return None

def process_img_topo_betti_curve(data, to_grayscale = transforms.Grayscale(num_output_channels=1)): #Processing to Betti_curves
    #Grayscale using provided function


    transform=transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_bw=transforms.Compose(
            [transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5, 0.5)])

    bw_img = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(data)
    bw_img = Image.fromarray(bw_img)
    image = transform(img)
    bw_img = transform_bw(bw_img)
    # print(image.shape)
    gray_scale_img = bw_img # to_grayscale(image)
    # calcuating the cubical complex
    cubical_complex = gd.CubicalComplex(dimensions=gray_scale_img.shape, top_dimensional_cells=gray_scale_img.flatten())
    # Calculating persistance
    diag = cubical_complex.persistence()
    # Calculating BettiCurve
    BC = gd.representations.vector_methods.BettiCurve()

    #This is created, because of the error caused by the  [x,+inf] persistant interval (connected components)
    if args.topodim_concat:
        BC_0 = BC.fit_transform([cubical_complex.persistence_intervals_in_dimension(0)[:-1]])
        L_t_0 = torch.tensor(BC_0,dtype=torch.float)
        BC_1 = BC.fit_transform([cubical_complex.persistence_intervals_in_dimension(1)])
        L_t_1 = torch.tensor(BC_1,dtype=torch.float)
        L_t = torch.cat([L_t_0,L_t_1],dim=1)

    elif args.topodim == 0:
        BC = BC.fit_transform([cubical_complex.persistence_intervals_in_dimension(args.topodim)[:-1]])
        L_t = torch.tensor(BC,dtype=torch.float)

    elif args.topodim == 1:
        BC = BC.fit_transform([cubical_complex.persistence_intervals_in_dimension(args.topodim)])
        L_t = torch.tensor(BC,dtype=torch.float)

    return image, L_t

#Processing using the topological Silhouette
def process_img_topo_silh(data, to_grayscale = transforms.Grayscale(num_output_channels=1)): #Processing to Silhouette
    #Grayscale using provided function


    transform=transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    transform_bw=transforms.Compose(
            [transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5, 0.5)])

    bw_img = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(data)
    bw_img = Image.fromarray(bw_img)
    image = transform(img)
    bw_img = transform_bw(bw_img)
    # print(image.shape)
    gray_scale_img = bw_img # to_grayscale(image)
    # calcuating the cubical complex
    cubical_complex = gd.CubicalComplex(dimensions=gray_scale_img.shape, top_dimensional_cells=gray_scale_img.flatten())
    # Calculating persistance
    diag = cubical_complex.persistence()
    # Calculating BettiCurve
    SI = gd.representations.vector_methods.Silhouette()

    #This is created, because of the error caused by the  [x,+inf] persistant interval (connected components)
    if args.topodim_concat:
        SI_0 = SI.fit_transform([cubical_complex.persistence_intervals_in_dimension(0)[:-1]])
        L_t_0 = torch.tensor(SI_0,dtype=torch.float)
        SI_1 = SI.fit_transform([cubical_complex.persistence_intervals_in_dimension(1)])
        L_t_1 = torch.tensor(SI_1,dtype=torch.float)
        L_t = torch.cat([L_t_0,L_t_1],dim=1)

    elif args.topodim == 0:
        SI = SI.fit_transform([cubical_complex.persistence_intervals_in_dimension(args.topodim)[:-1]])
        L_t = torch.tensor(SI,dtype=torch.float)

    elif args.topodim == 1:
        SI = SI.fit_transform([cubical_complex.persistence_intervals_in_dimension(args.topodim)])
        L_t = torch.tensor(SI,dtype=torch.float)



    return image, L_t


def process_img_topo_pi_v(data, to_grayscale = transforms.Grayscale(num_output_channels=1)): #Processing to Persistant images
    #Grayscale using provided function


    transform=transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_bw=transforms.Compose(
            [transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5, 0.5)])

    bw_img = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(data)
    bw_img = Image.fromarray(bw_img)
    image = transform(img)
    bw_img = transform_bw(bw_img)
    # print(image.shape)
    gray_scale_img = bw_img # to_grayscale(image)
    # calcuating the cubical complex
    cubical_complex = gd.CubicalComplex(dimensions=gray_scale_img.shape, top_dimensional_cells=gray_scale_img.flatten())
    # Calculating persistance
    diag = cubical_complex.persistence()
    # Calculating BettiCurve
    PI = gd.representations.PersistenceImage(bandwidth=0.05,resolution=[64,64],weight=lambda x: x[1]**2, im_range=[0,0.6,0,0.6])

    #This is created, because of the error caused by the  [x,+inf] persistant interval (connected components)
    if args.topodim_concat:
        PI_0 = PI.fit_transform([cubical_complex.persistence_intervals_in_dimension(0)[:-1]])
        L_t_0 = torch.tensor(PI_0,dtype=torch.float)
        PI_1 = PI.fit_transform([cubical_complex.persistence_intervals_in_dimension(1)])
        L_t_1 = torch.tensor(PI_1,dtype=torch.float)
        L_t = torch.cat([L_t_0,L_t_1],dim=1)

    elif args.topodim == 0:
        PI = PI.fit_transform([cubical_complex.persistence_intervals_in_dimension(args.topodim)[:-1]])
        L_t = torch.tensor(PI,dtype=torch.float)

    elif args.topodim == 1:
        PI = PI.fit_transform([cubical_complex.persistence_intervals_in_dimension(args.topodim)])
        L_t = torch.tensor(PI,dtype=torch.float)



    return image, L_t


def process_img_topo_pi_img(data, to_grayscale = transforms.Grayscale(num_output_channels=1)): #Processing to Persistant images
    #Grayscale using provided function


    transform=transforms.Compose(
        [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    transform_bw=transforms.Compose(
            [transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5, 0.5)])

    bw_img = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    img = Image.fromarray(data)
    bw_img = Image.fromarray(bw_img)
    image = transform(img)
    bw_img = transform_bw(bw_img)
    # print(image.shape)
    gray_scale_img = bw_img # to_grayscale(image)
    # calcuating the cubical complex
    cubical_complex = gd.CubicalComplex(dimensions=gray_scale_img.shape, top_dimensional_cells=gray_scale_img.flatten())
    # Calculating persistance
    diag = cubical_complex.persistence()
    # Calculating BettiCurve
    PI = gd.representations.PersistenceImage(bandwidth=0.05,resolution=[64,64],weight=lambda x: x[1]**2, im_range=[0,0.6,0,0.6])

    #For the Persistent Images, the concat output gives 2 images - a simple solution
    if args.topodim_concat:
        PI_0 = PI.fit_transform([cubical_complex.persistence_intervals_in_dimension(0)[:-1]])
        L_t_0 = torch.tensor(PI_0,dtype=torch.float).reshape([1,64,64])
        PI_1 = PI.fit_transform([cubical_complex.persistence_intervals_in_dimension(1)])
        L_t_1 = torch.tensor(PI_1,dtype=torch.float).reshape([1,64,64])
        L_t = torch.cat([L_t_0,L_t_1],dim = 0)

    elif args.topodim == 0:
        PI = PI.fit_transform([cubical_complex.persistence_intervals_in_dimension(args.topodim)[:-1]])
        L_t = torch.tensor(PI,dtype=torch.float).reshape([1,64,64])

    elif args.topodim == 1:
        PI = PI.fit_transform([cubical_complex.persistence_intervals_in_dimension(args.topodim)])
        L_t = torch.tensor(PI,dtype=torch.float).reshape([1,64,64])



    return image, L_t

def process_data_topo(dataset, train_set = True, from_train = None, slice = None):

    # if 
    augmentation_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), # Small shifts
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.GaussianBlur(kernel_size=3)]
        )


    data = dataset.data
    if slice is not None:
        data = dataset.data[:slice]

    data_len = data.shape[0]

    if args.tv == 'land':
        print('Processing data using landscape vectorization')
        if args.cores <= 1:
            results = []
            print('processing using a single core - NO multiprocessing')
            for sample in tqdm(data):
                results.append(process_img_topo_land(sample))
        else:
            with Pool(args.cores) as pool: #multiprocessing the topological data transform
                results = list(tqdm(pool.imap(process_img_topo_land,data),total=len(data)))


    elif args.tv == 'bc':
        print('Processing data using betti curve vectorization')

        if args.cores <= 1:
            results = []
            print('processing using a single core - NO multiprocessing')
            for sample in tqdm(data):
                results.append(process_img_topo_betti_curve(sample))
        else:
            with Pool(args.cores) as pool: #multiprocessing the topological data transform
                results = list(tqdm(pool.imap(process_img_topo_betti_curve,data),total=len(data)))

    elif args.tv == 'pi_v':
        print('Processing data using PI (Vector)')

        if args.cores <= 1:
            results = []
            print('processing using a single core - NO multiprocessing')
            for sample in tqdm(data):
                results.append(process_img_topo_pi_v(sample))
        else:
            with Pool(args.cores) as pool: #multiprocessing the topological data transform
                results = list(tqdm(pool.imap(process_img_topo_pi_v,data),total=len(data)))

    elif args.tv == 'pi_img':
        print('Processing data using PI (Image)')

        if args.cores <= 1:
            results = []
            print('processing using a single core - NO multiprocessing')
            for sample in tqdm(data):
                results.append(process_img_topo_pi_img(sample))
        else:
            with Pool(args.cores) as pool: #multiprocessing the topological data transform
                results = list(tqdm(pool.imap(process_img_topo_pi_img,data),total=len(data)))

    elif args.tv == 'silh':
        print('Processing data using Silhouette vectorization')

        if args.cores <= 1:
            results = []
            print('processing using a single core - NO multiprocessing')
            for sample in tqdm(data):
                results.append(process_img_topo_silh(sample))
        else:
            with Pool(args.cores) as pool: #multiprocessing the topological data transform
                results = list(tqdm(pool.imap(process_img_topo_silh,data, maxtaskperchild = 1),total=len(data)))

    else:
        raise('Error - invalid topological vectorization method')



    tmp_topo_data = [item[1] for item in results]
    tensor_topo_data = torch.cat(tmp_topo_data,dim=0)

    if from_train is not None:
        max = from_train[0]
        min = from_train[1]
    else:
        min = tensor_topo_data.min()
        max = tensor_topo_data.max()

    new_res = []
    for img,topo in results:
        stand_topo = (topo - min)/(max-min)
        new_res.append((img,stand_topo))
    results = new_res

    if train_set:
        return results, (max,min)

    return results



def process_topo_batch(numpy_batch, topo_vectorization, from_train):

    
    if args.cores > 1:
        with Pool(args.cores) as pool:
            results = list(pool.imap(topo_vectorization,numpy_batch))
    else:
        results = []
        for sample in numpy_batch:
            results.append(topo_vectorization(sample))
    
    topo_features = [item[1] for item in results]
    tensor_topo_data = torch.stack(topo_features,dim=0)

    max_val, min_val = from_train
    stand_topo = (tensor_topo_data - min_val)/ (max_val - min_val)

    return stand_topo
