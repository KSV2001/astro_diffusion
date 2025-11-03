from torchvision import transforms


def train_transforms(size: int):
    """
    Standard training transforms: resize, center crop, convert to tensor,
    and normalize to [-1, 1].
    """
    return transforms.Compose(
        [
            transforms.Resize(size, transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
