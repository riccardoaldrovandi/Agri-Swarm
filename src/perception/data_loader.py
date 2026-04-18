import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from .data_utils import load_stats  # Importing load_stats to get the mean and std for normalization

def get_data_loaders(data_dir, batch_size=32):
    # Loading dataset statistics (mean and std) for normalization from the JSON file created by prepare_data.py
    stats = load_stats()
    
    if stats is None:
        print("⚠️ Stats not found! I'm using the default values ​​(ImageNet).")
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        img_size = (100, 100)
    else:
        mean, std = stats['mean'], stats['std']
        img_size = stats['img_size']

    # Transformations (Normalizing both with the same stats from the training set)
    train_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    # Loading datasets and creating data loaders
    full_train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=train_transforms)
    train_size = int(0.9 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    # Split casuale
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size]
    )
    test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=test_transforms)

    # shuffle=True shuffles data AUTOMATICALLY at the beginning of each epoch

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, full_train_dataset.classes, img_size