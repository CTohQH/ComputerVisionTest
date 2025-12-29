import torch
from torchvision.datasets import Flowers102
from torch.utils.data import DataLoader, Dataset
from .transforms import get_transforms

def create_dataloaders(data_dir='./src/data', batch_size=32, num_workers=2):
    """
    Creates DataLoaders for train, validation, and test sets.
    Uses the official Flowers102 splits but ensures transforms are applied.
    """
    
    # Define transforms
    train_transform = get_transforms('train')
    val_test_transform = get_transforms('val')

    # Load datasets
    # Note: Flowers102 'train' set is smaller (10 images per class), 
    # 'test' is larger (min 20 per class). 
    # Usually we merge and re-split if we want a standard larger train set, 
    # but strictly adhering to the dataset often means using its defined splits.
    # However, for better deep learning training, it's common to use 'train' + 'val' for training 
    # if the official 'train' is too small, OR just use the official splits.
    # Let's stick to official splits first as per "Data Integrity" rule implies using the dataset via torchvision.
    
    train_dataset = Flowers102(root=data_dir, split='train', download=True, transform=train_transform)
    val_dataset = Flowers102(root=data_dir, split='val', download=True, transform=val_test_transform)
    test_dataset = Flowers102(root=data_dir, split='test', download=True, transform=val_test_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader
