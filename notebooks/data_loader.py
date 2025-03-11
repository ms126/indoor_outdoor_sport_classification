# data_loader.py

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def load_data(batch_size=5, 
              train_dir="/projects/dsci410_510/Final_proj_ms/train", 
              valid_dir="/projects/dsci410_510/Final_proj_ms/valid", 
              test_dir="/projects/dsci410_510/Final_proj_ms/test"):
    
    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load datasets
    train_data = datasets.ImageFolder(root=train_dir, transform=transform)
    valid_data = datasets.ImageFolder(root=valid_dir, transform=transform)
    test_data = datasets.ImageFolder(root=test_dir, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader
