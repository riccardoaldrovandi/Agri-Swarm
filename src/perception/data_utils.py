import torch
import os
import json
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image

def calculate_and_save_stats(data_dir, output_path="data/processed/dataset_stats.json"):
    """
    Calculates mean and standard deviation + img size, saves them to a JSON file, and returns them.
    """

    # Looking for the first image to determine the number of channels and confirm the image size
    first_class = os.listdir(os.path.join(data_dir, 'train'))[0]
    first_img_name = os.listdir(os.path.join(data_dir, 'train', first_class))[0]
    first_img_path = os.path.join(data_dir, 'train', first_class, first_img_name)

    with Image.open(first_img_path) as img:
        width, height = img.size

    # Calculating size of the input images for the model (after resizing)
    img_size = (height, width)

    temp_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor()
    ])
    
    dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=temp_transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=2)

    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0

    print("📊 Calculating dataset statistics...")
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images

    stats = {
        "mean": mean.tolist(),
        "std": std.tolist(),
        "img_size": img_size
    }

    # Salvataggio su JSON
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=4)
    
    print(f"✅ Dataset statistics saved to: {output_path}")
    return stats

def load_stats(path="data/processed/dataset_stats.json"):
    """Loads the dataset statistics from a JSON file."""
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)