from torchvision import transforms
import numpy as np
from topology.pi import process_PI


class AugmentAndCalculateFeatures:
    def __init__(self, args, train=True, pi_mean=None, pi_std=None):
        self.train = train
        self.args = args
        self.base_augmentations = transforms.Compose(
            [
                transforms.RandomResizedCrop(64, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),  # Color adjustments
                transforms.RandomRotation(15),  # Randomly rotate the image
                transforms.GaussianBlur((3, 3)),
                transforms.RandomPerspective(),
            ]
        )

        self.final_image_transform_train = transforms.Compose(
            [
                transforms.ToTensor(),  # Converts (H, W, C) NumPy to (C, H, W) Tensor
                transforms.RandomErasing(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.final_image_transform_val = transforms.Compose(
            [
                transforms.ToTensor(),  # Converts (H, W, C) NumPy to (C, H, W) Tensor
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        if pi_mean is not None and pi_std is not None:
            self.pi_normalize = transforms.Normalize(mean=pi_mean, std=pi_std)
        else:
            self.pi_normalize = transforms.Normalize(mean=[0.0], std=[1.0])

    def __call__(self, pil_image):

        if self.train:
            pil_image = self.base_augmentations(pil_image)
        else:
            # pil_image = transforms.Resize((224, 224))(pil_image)
            pass

        pil_image = pil_image.convert("RGB")
        image_np = np.array(pil_image)

        topo_features = self.pi_normalize(process_PI(input=pil_image, args=self.args))
        if self.train:
            image_tensor = self.final_image_transform_train(image_np)
        else:
            image_tensor = self.final_image_transform_val(image_np)

        return (image_tensor, topo_features)
