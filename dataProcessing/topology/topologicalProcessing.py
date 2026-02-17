import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import cv2  
import gudhi as gd 
import gudhi.representations
from tqdm import tqdm

# def process_PI(input, args): #Processing to Persistant images
#
#     image_np = np.array(input)
#     bw_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
#     # bw_img = Image.fromarray(bw_img)
#     # calcuating the cubical complex
#     cubical_complex = gd.CubicalComplex(dimensions=bw_img.shape, top_dimensional_cells=bw_img.flatten())
#     # Calculating persistance
#     diag = cubical_complex.persistence()
#     # Calculating BettiCurve
#     PI = gd.representations.PersistenceImage(bandwidth=5,resolution=[64,64],weight=lambda x: x[1]**2, im_range=[0,256,0,256])
#
#     #For the Persistent Images, the concat output gives 2 images - a simple solution
#     if args.topodim_concat:
#         PI_0 = PI.fit_transform([cubical_complex.persistence_intervals_in_dimension(0)[:-1]])
#         L_t_0 = torch.tensor(PI_0,dtype=torch.float).reshape([1,64,64])
#         PI_1 = PI.fit_transform([cubical_complex.persistence_intervals_in_dimension(1)])
#         L_t_1 = torch.tensor(PI_1,dtype=torch.float).reshape([1,64,64])
#         L_t = torch.cat([L_t_0,L_t_1],dim = 0)
#
#     elif args.topodim == 0:
#         PI = PI.fit_transform([cubical_complex.persistence_intervals_in_dimension(args.topodim)[:-1]])
#         L_t = torch.tensor(PI,dtype=torch.float).reshape([1,64,64])
#
#     elif args.topodim == 1:
#         PI = PI.fit_transform([cubical_complex.persistence_intervals_in_dimension(args.topodim)])
#         L_t = torch.tensor(PI,dtype=torch.float).reshape([1,64,64])
#
#     return L_t

def process_PI(input, args): #Processing to Persistant images

    image_np = np.array(input)
    bw_img = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    # bw_img = Image.fromarray(bw_img)
    # calcuating the cubical complex
    cubical_complex = gd.CubicalComplex(dimensions=bw_img.shape, top_dimensional_cells=bw_img.flatten())
    # Calculating persistance
    diag = cubical_complex.persistence()
    # Calculating BettiCurve
    PI = gd.representations.PersistenceImage(bandwidth=2,resolution=[64,64],weight=lambda x: (x[0] - x[1])**2, im_range=[0,256,0,256])

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
    def __init__(self, args, train=True, pi_mean = None, pi_std = None):
        self.train = train
        self.args = args
        self.base_augmentations = transforms.Compose([
            transforms.RandomResizedCrop(64, scale= (0.08,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # Color adjustments
            transforms.RandomRotation(15),        # Randomly rotate the image
            transforms.GaussianBlur((3,3)),
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
        if pi_mean is not None and pi_std is not None: 
            self.pi_normalize = transforms.Normalize(mean=pi_mean, std=pi_std)
        else: 
            self.pi_normalize = transforms.Normalize(mean=[0.0], std=[1.0])

    # tinyImage net max value: 0.13958996534347534 #TODO: Change this to loading a config file 1247.8710
    # Caltech256 max value: 15003.3369140625)
    def __call__(self, pil_image):
        
        if self.train:
            pil_image = self.base_augmentations(pil_image)
        else:
            # pil_image = transforms.Resize((224, 224))(pil_image)
            pass
            
        pil_image = pil_image.convert('RGB')
        image_np = np.array(pil_image)
        
        topo_features = self.pi_normalize(process_PI(input = pil_image, args = self.args)) 
        # topo_features = process_PI(input = pil_image, args = self.args) 
        #Add normalization here brther

        if self.train:
            image_tensor = self.final_image_transform_train(image_np)
        else:
            image_tensor = self.final_image_transform_val(image_np)
        
        return (image_tensor, topo_features)


def get_topo_DS(dir_path = '.', dataset = None, args = None):
    train_transform = AugmentAndCalculateFeatures(train= True, args= args)
    val_transform = AugmentAndCalculateFeatures(train= False, args= args)
    
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

def calculate_dataset_stats(dataset_loader):
    """
    Iterates through the dataset to calculate Mean and Std for Topology channel.
    """
    print("--- Calculating Topology Statistics (This takes a moment) ---")
    cnt = 0
    fst_moment = torch.empty(3) # Placeholder
    snd_moment = torch.empty(3)
    
    # We only need to check the first batch to get channel count
    for i, (images, topo_features) in enumerate(tqdm(dataset_loader)):
        # topo_features shape: (Batch, Channels, H, W)
        b, c, h, w = topo_features.shape
        nb_pixels = b * h * w
        
        if i == 0:
            fst_moment = torch.zeros(c)
            snd_moment = torch.zeros(c)

        # Calculate sum and sum_of_squares across (Batch, H, W)
        # leaving Channels dimension intact
        sum_ = torch.sum(topo_features, dim=[0, 2, 3])
        sum_sq_ = torch.sum(topo_features ** 2, dim=[0, 2, 3])
        
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_sq_) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean = fst_moment
    std = torch.sqrt(snd_moment - fst_moment ** 2)
    
    print(f"Calculated Mean: {mean}")
    print(f"Calculated Std: {std}")
    return mean.tolist(), std.tolist()


def calculate_accurate_stats_two_pass(dataset):
    """
    Robust Two-Pass algorithm to calculate Mean and Std.
    Essential for high-variance data like Persistent Images.
    """
    # Create a simple loader just for stats
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print("--- [Pass 1/2] Calculating Global Mean ---")
    pixel_sum = None
    total_pixels = 0
    n_channels = None

    # PASS 1: Mean
    for i, (images, topo_features) in enumerate(tqdm(loader)):
        # topo_features: (B, C, H, W)
        if n_channels is None:
            n_channels = topo_features.shape[1]
            pixel_sum = torch.zeros(n_channels, dtype=torch.float64) # Float64 for safety

        # Sum over Batch(0), Height(2), Width(3) -> Result (C,)
        pixel_sum += torch.sum(topo_features, dim=[0, 2, 3]).double()
        
        # Count pixels (B * H * W)
        total_pixels += topo_features.shape[0] * topo_features.shape[2] * topo_features.shape[3]

    global_mean = pixel_sum / total_pixels
    print(f"Global Mean calculated: {global_mean.tolist()}")

    # PASS 2: Std
    print("--- [Pass 2/2] Calculating Global Std ---")
    sum_squared_diff = torch.zeros(n_channels, dtype=torch.float64)

    for i, (images, topo_features) in enumerate(tqdm(loader)):
        topo_features = topo_features.double()
        # Reshape mean for broadcasting: (1, C, 1, 1)
        mean_view = global_mean.view(1, n_channels, 1, 1)
        
        # (x - mean)^2
        diff = (topo_features - mean_view) ** 2
        sum_squared_diff += torch.sum(diff, dim=[0, 2, 3])

    global_var = sum_squared_diff / total_pixels
    global_std = torch.sqrt(global_var)
    
    print(f"Global Std calculated: {global_std.tolist()}")

    return global_mean.float().tolist(), global_std.float().tolist()

def save_stats(mean, std, path):
    with open(path / 'topo_stats.json', 'w') as f:
        json.dump({'mean': mean, 'std': std}, f)

def load_stats(path):
    with open(path / 'topo_stats.json', 'r') as f:
        data = json.load(f)
    return data['mean'], data['std']
