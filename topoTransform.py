import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
import cv2  
import gudhi as gd 
import gudhi.representations
from PIL import Image
from config import args

def process_PI(input): #Processing to Persistant images

    image_np = np.array(input)
    bw_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    # bw_img = Image.fromarray(bw_img)
    # calcuating the cubical complex
    cubical_complex = gd.CubicalComplex(dimensions=bw_img.shape, top_dimensional_cells=bw_img.flatten())
    # Calculating persistance
    diag = cubical_complex.persistence()
    # Calculating BettiCurve
    PI = gd.representations.PersistenceImage(bandwidth=5,resolution=[64,64],weight=lambda x: x[1]**2, im_range=[0,256,0,256])

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

    return L_t

class AugmentAndCalculateFeatures:
    def __init__(self, train=True):
        self.train = train
        
        self.base_augmentations = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Color adjustments
            transforms.RandomRotation(15),        # Randomly rotate the image
            transforms.GaussianBlur((5,5)),
            transforms.RandomPerspective()
        ])
        
        self.final_image_transform_train = transforms.Compose([
            transforms.ToTensor(), # Converts (H, W, C) NumPy to (C, H, W) Tensor
            transforms.RandomErasing(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        self.final_image_transform_val = transforms.Compose([
            transforms.ToTensor(), # Converts (H, W, C) NumPy to (C, H, W) Tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        ])
        self.pi_transform = transforms.Compose([
             # transforms.ToTensor(),
            transforms.Normalize(mean=6.56658, std = 34.323)
            ])
    def __call__(self, pil_image):
        
        if self.train:
            pil_image = self.base_augmentations(pil_image)
        else:
            pil_image = transforms.Resize((224, 224))(pil_image)
            
        pil_image = pil_image.convert('RGB')
        image_np = np.array(pil_image)
        
        # bw_image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        topo_features = self.pi_transform(process_PI(pil_image)) 
        #Add normalization here brther

        if self.train:
            image_tensor = self.final_image_transform_train(image_np)
        else:
            image_tensor = self.final_image_transform_val(image_np)
        
        return (image_tensor, topo_features)


def get_topo_DS(dir_path = '.', dataset = None):
    train_transform = AugmentAndCalculateFeatures(train= True)
    val_transform = AugmentAndCalculateFeatures(train= False)
    
    train_set_full = dataset(root = dir_path, transform = train_transform, download = False)
    val_set_full = dataset(root = dir_path, transform = val_transform, download = False)

    dataset_len = len(train_set_full)
    indicies = list(range(dataset_len))

    train_size = int(dataset_len * ( 1 - args.val_size))
    val_size = dataset_len - train_size

    np.random.seed(42)
    np.random.shuffle(indicies)
    train_idx, val_idx = indicies[:train_size] , indicies[train_size:] 

    train_subset = Subset(train_set_full, train_idx)
    val_subset = Subset(val_set_full, val_idx)

    print(f"Training samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_subset)}")


    train_loader = DataLoader(
        train_subset,
        batch_size = args.batch_size,
        shuffle = True, 
        num_workers = args.num_workers,
        pin_memory = True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size = args.batch_size,
        shuffle = False, 
        num_workers = args.num_workers,
        pin_memory = True
    )

    return train_loader, val_loader
