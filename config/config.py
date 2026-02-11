import argparse
import os

def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg

def get_env_var(var_name, default):
    path = os.getenv(var_name, default=default)

    if not os.path.exists(path):
        print(f"Warning: {path} Does not exists")
    return path 

#The argparser part
argparser = argparse.ArgumentParser(fromfile_prefix_chars='@')

#env variables 
argparser.add_argument('--data_path',default = get_env_var("DATA_PATH","./data"), type=str, help="Path to the data")

argparser.add_argument("--lr", default=0.0003, type=float, help="Meta-learning rate (used on query set - potentially acoss tasks)")
argparser.add_argument("--seed", default=119, type=int, help="Seed to use")
argparser.add_argument("--model", default="TBR", type=str, help="Select model, avaliable models are ResNet, TR, TBR")
argparser.add_argument("--modelFT", default="PI_IMG", type=str, help="Select model, avaliable models are ResNet50, PI_IMG")
argparser.add_argument("--tv", default="land", type=str, help="Topological vectorization method used, methods available - check readme.txt")
argparser.add_argument("--res", default=100, type=int, help="Resolution for the Landscape vectorization method")
argparser.add_argument("--tbs", default="normal", type=str, help="Topo block size")
argparser.add_argument("--sm", default=False, action="store_true", help="Enables saving the model")
argparser.add_argument("--bw", default="cv2", type=str, help="Select the black-white transformation option")
argparser.add_argument("--topodim", default=1, type=int, help="Which dimension of the topology groups to use")
argparser.add_argument("--topodim_concat", default=False, action="store_true", help="Concatenating both dimensions of the topology features on 0 and 1 dim")
argparser.add_argument("--epochs", default=50, type=int, help="Number of epochs to train on")
argparser.add_argument("--cores", default=8, type=int, help="Number of cores to use for multiprocessing")
argparser.add_argument("--batch_size", default=64, type=int, help="Batchsize of the training")
argparser.add_argument("--val_size", default=0.2, type=float, help="Size of the validation set")
argparser.add_argument("--num_workers", default=2, type=int, help="Number of workers fo the dataloaders")
argparser.add_argument("--config", default=None, type=str, help="Path to config file, containing the Nerual architectures")
argparser.add_argument("--name", default="", type=str, help="Name of the run")
argparser.add_argument("--tb_add_x", default=False, action="store_true", help="Add the x into the topological resnet when using PIBlock")
argparser.add_argument("--tb_add_t", default=False, action="store_true", help="Add the topological out back into the topological resnet when using PIBlock")
argparser.add_argument("--aug", default=0, type=float, help="Data augmentation size")
argparser.add_argument("--aug_type", default="all", type=str, help="Data augmentation type")
argparser.add_argument("--freeze_weights", default=False, action="store_true", help="Freeze base model wieghts")
argparser.add_argument("--hidden_size", default=256, type=int, help="Size of the hidden vector for both Homology and image features")
argparser.add_argument("--run_id", default="" , type=str, help="Run id from W&B to be used for resuming or testing")
argparser.add_argument("--modelPath", default="./saveModels" , type=str, help="Path to the directory where models are stored")

args = argparser.parse_args()
