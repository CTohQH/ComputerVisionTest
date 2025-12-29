import sys
import os
import torch

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.loader import create_dataloaders

def test_loading():
    print("Testing data loading...")
    try:
        train_loader, val_loader, test_loader = create_dataloaders(data_dir='./data', batch_size=4, num_workers=0)
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Check one batch
        images, labels = next(iter(train_loader))
        print(f"Batch shape: {images.shape}")
        print(f"Labels: {labels}")
        
        if images.shape == (4, 3, 224, 224):
            print("Shape check passed!")
        else:
            print(f"Shape check failed! Expected (4, 3, 224, 224), got {images.shape}")
            
    except Exception as e:
        print(f"Error during loading: {e}")

if __name__ == "__main__":
    test_loading()
