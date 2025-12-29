import torch.nn as nn
from torchvision import models

def get_model(num_classes=102, fine_tune=True):
    """
    Loads a pre-trained ResNet50 and replaces the final layer.
    
    Args:
        num_classes (int): Number of output classes (102 for flowers).
        fine_tune (bool): If True, unfreezes specific layers for fine-tuning.
    """
    # Load pre-trained ResNet50
    # Note: 'pretrained=True' is deprecated in newer versions, use 'weights=ResNet50_Weights.IMAGENET1K_V1'
    # But for compatibility with older envs or strict adherence to "pretrained weights" wording:
    try:
        from torchvision.models import ResNet50_Weights
        weights = ResNet50_Weights.IMAGENET1K_V1
        model = models.resnet50(weights=weights)
    except ImportError:
        # Fallback for older torch versions
        model = models.resnet50(pretrained=True)

    # Freeze basic parameters
    for param in model.parameters():
        param.requires_grad = False

    # Replace the final fully connected layer
    # ResNet50's fc layer input features is 2048
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # If fine-tuning is enabled, we might want to unfreeze the last block or just train the head initially
    # For now, let's keep the feature extractor frozen and only train the head (model.fc) which IS unfrozen by default when created.
    # If we want deeper fine-tuning (e.g. layer4), we can unfreeze it later.
    
    return model
