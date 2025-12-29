import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torchvision.transforms.functional as TF
import torch.nn.functional as F


def evaluate_model(model, dataloader, device):
    """
    Runs inference and returns true labels and predictions.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return np.array(all_labels), np.array(all_preds)

def predict_tta(model, dataloader, device, num_augs=5):
    """
    Runs inference with Test-Time Augmentation (TTA).
    Averages predictions across multiple augmented versions of each image.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Store probabilities for this batch
            batch_probs = torch.zeros(inputs.size(0), 102).to(device)
            
            for _ in range(num_augs):
                # Apply random augmentations manually or via model forward if it had dropout enabled (but here we want inputs)
                # Since dataloader gives tensor, we can apply simple flips/crops on tensor
                
                # Standard predictions (no aug)
                if _ == 0:
                    aug_inputs = inputs
                else:
                    # Random Flip
                    if torch.rand(1).item() > 0.5:
                        aug_inputs = TF.hflip(inputs)
                    else:
                        aug_inputs = inputs
                        
                    # Could add more like crops, but keeping it simple for tensor inputs
                
                outputs = model(aug_inputs)
                probs = F.softmax(outputs, dim=1)
                batch_probs += probs
            
            # Average
            batch_probs /= num_augs
            _, preds = torch.max(batch_probs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(y_true, y_pred, class_names=None):
    """
    Plots the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

def print_classification_report(y_true, y_pred, target_names=None):
    """
    Prints the classification report.
    """
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))

def visualize_misclassifications(model, dataloader, device, num_images=5):
    """
    Visualizes misclassified images.
    """
    model.eval()
    count = 0
    fig = plt.figure(figsize=(15, 6))
    
    # ImageNet un-normalization for display
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Identify mistakes in batch
            mistakes = preds != labels
            if not mistakes.any():
                continue
                
            mistake_indices = torch.where(mistakes)[0]
            
            for idx in mistake_indices:
                if count >= num_images:
                    break
                    
                img = inputs[idx].cpu().permute(1, 2, 0).numpy()
                img = std * img + mean
                img = np.clip(img, 0, 1)
                
                true_label = labels[idx].item()
                pred_label = preds[idx].item()
                
                ax = fig.add_subplot(1, num_images, count + 1)
                ax.imshow(img)
                ax.set_title(f"True: {true_label}\nPred: {pred_label}", color='red')
                ax.axis('off')
                
                count += 1
            
            if count >= num_images:
                break
    plt.show()
