from torchvision import transforms


def get_tinyimagenet_transforms(image_size=224):
    """
    Preprocessing and augmentation for Tiny ImageNet using pretrained ResNet-18.

    Returns:
        train_tfms: transforms for training data
        val_tfms: transforms for validation data
    """

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_tfms = transforms.Compose([
        transforms.RandomResizedCrop(
            size=image_size,
            scale=(0.6, 1.0),
            ratio=(0.75, 1.333)
        ),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=imagenet_mean,
            std=imagenet_std
        ),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=imagenet_mean,
            std=imagenet_std
        ),
    ])

    return train_tfms, val_tfms
