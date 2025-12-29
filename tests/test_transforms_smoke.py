import sys
import os
import torch
from PIL import Image
import numpy as np

# Add src to path
sys.path.append(os.path.abspath('src')) # Running from root

from data.transforms import get_transforms

def test_transforms_shape():
    print("Testing Transforms Shape...")
    # Create a dummy image (e.g. 100x200, non-square)
    img = Image.fromarray(np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8))
    
    # Get transforms
    transform = get_transforms('train', image_size=640)
    
    # Apply
    out_tensor = transform(img)
    
    print(f"Output Tensor Shape: {out_tensor.shape}")
    
    if out_tensor.shape == (3, 640, 640):
        print("SUCCESS: Shape is correct (3, 640, 640)")
    else:
        print(f"FAILURE: Shape is {out_tensor.shape}")
        sys.exit(1)

    # Test Val transform too
    transform_val = get_transforms('val', image_size=640)
    out_val = transform_val(img)
    if out_val.shape == (3, 640, 640):
         print("SUCCESS: Val Shape is correct")
    else:
         print(f"FAILURE: Val Shape is {out_val.shape}")
         sys.exit(1)

if __name__ == "__main__":
    test_transforms_shape()
