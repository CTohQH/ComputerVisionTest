from torchvision import transforms

from PIL import Image, ImageOps

class ResizeWithPad:
    """
    Resizes the image to the target size by maintaining aspect ratio
    and adding white padding (letterboxing) to fit the square.
    """
    def __init__(self, size, fill=(255, 255, 255)):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        target_size = (self.size, self.size)
        
        # Copy image to avoid in-place modification issues
        img = img.copy()
        
        # Resize maintaining aspect ratio (thumbnail max dimension is size)
        img.thumbnail(target_size, Image.Resampling.LANCZOS)
        
        # Calculate padding
        delta_w = self.size - img.size[0]
        delta_h = self.size - img.size[1]
        
        # (left, top, right, bottom)
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        
        return ImageOps.expand(img, padding, fill=self.fill)

def get_transforms(phase='train', image_size=640):
    """
    Returns the data transforms for the specified phase.
    
    Args:
        phase (str): 'train' or 'val'/'test'
        image_size (int): Target input size (default 640)
    """
    # ImageNet normalization stats
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if phase == 'train':
        return transforms.Compose([
            ResizeWithPad(image_size),
            transforms.RandomRotation(30),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    else:
        return transforms.Compose([
            ResizeWithPad(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
