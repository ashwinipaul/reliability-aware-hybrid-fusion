import torch
from torchvision import transforms

class AddGaussianNoise:
    def __init__(self, std=0.02, p=0.25):
        self.std = std
        self.p = p

    def __call__(self, x):
        if torch.rand(1).item() < self.p:
            return (x + torch.randn_like(x) * self.std).clamp(0, 1)
        return x

class StainLikeAugmentation:
    """
    Lightweight stain perturbation. The manuscript specifies stain
    augmentation but does not define its exact algorithm/parameters.
    Replace this transform with the validated historical implementation
    when those details are available.
    """
    def __init__(self, p=0.25):
        self.p = p

    def __call__(self, x):
        if torch.rand(1).item() >= self.p:
            return x
        scale = torch.empty(3, 1, 1).uniform_(0.85, 1.15)
        bias = torch.empty(3, 1, 1).uniform_(-0.05, 0.05)
        gamma = torch.empty(1).uniform_(0.90, 1.10).item()
        return ((x * scale + bias).clamp(0, 1) ** gamma).clamp(0, 1)

def build_transforms(cfg, train=True):
    size = int(cfg["dataset"]["image_size"])
    aug = cfg.get("augmentation", {})
    if train:
        ops = [
            transforms.RandomResizedCrop(
                size,
                scale=(
                    float(aug.get("random_crop_scale_min", 0.80)),
                    float(aug.get("random_crop_scale_max", 1.00)),
                ),
            ),
            transforms.RandomHorizontalFlip(
                p=float(aug.get("horizontal_flip_p", 0.50))
            ),
            transforms.RandomApply(
                [transforms.RandomRotation(float(aug.get("rotation_degrees", 20)))],
                p=float(aug.get("rotation_p", 0.50)),
            ),
            transforms.ColorJitter(
                brightness=0.20, contrast=0.20, saturation=0.20, hue=0.05
            ),
            StainLikeAugmentation(
                p=float(aug.get("stain_augmentation_p", 0.25))
            ),
            transforms.ToTensor(),
            AddGaussianNoise(
                std=float(aug.get("gaussian_noise_std", 0.02)),
                p=float(aug.get("gaussian_noise_p", 0.25)),
            ),
        ]
    else:
        ops = [transforms.Resize((size, size)), transforms.ToTensor()]
    return transforms.Compose(ops)
