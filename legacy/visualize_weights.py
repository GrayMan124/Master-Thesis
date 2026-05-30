import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import torch
from torchvision.models import resnet50 

from config.config import args
from models.PI_finetune import PIFineTuneModel
from models.FineTuneResNet import ResNetFineTune
import matplotlib.pyplot as plt


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

    data_path = args.data_path

    base_model = resnet50(weights = "IMAGENET1K_V2")
    if args.modelFT == 'PI_IMG': 
        model = PIFineTuneModel(base_model = base_model, image_channels = 3, num_classes = 200, device= device, args= args)
        model.to(device)
    elif args.modelFT == "ResNet50":
        model = ResNetFineTune(base_model = base_model, image_channels = 3, num_classes = 200, device= device, args= args)
        model.to(device)
    else:
        raise Exception(f"Unrecognized modelFT argument: {args.modelFT}")
    # model.compile() 
    state_dict = torch.load(os.path.join(args.modelPath, f"{args.name}.pkl"), weights_only=True)

    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)

    for key in new_state_dict:
        print(key)

    print()
    print(new_state_dict['fc.0.weight'].shape)
