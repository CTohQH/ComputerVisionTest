import matplotlib.pyplot as plt
import numpy as np
import torch
from collections import Counter

def plot_class_distribution(dataset, title="Class Distribution"):
    """
    Plots the distribution of classes in the dataset.
    """
    labels = [y for _, y in dataset]
    counts = Counter(labels)
    
    plt.figure(figsize=(15, 6))
    plt.bar(counts.keys(), counts.values())
    plt.title(title)
    plt.xlabel("Class ID")
    plt.ylabel("Count")
    plt.show()

def show_samples(dataset, num_samples=9, cols=3):
    """
    Displays a grid of sample images from the dataset.
    """
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    rows = (num_samples + cols - 1) // cols
    plt.figure(figsize=(15, 5 * rows))
    
    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image)
        plt.title(f"Class: {label}")
        plt.axis('off')
    plt.show()

def calculate_stats(dataset):
    """
    Calculates the mean and standard deviation of the dataset.
    Note: This can be slow for large datasets.
    """
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    
    mean = 0.0
    std = 0.0
    total_images_count = 0
    
    for images, _ in loader:
        # Assuming images are already tensors (0-1)
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images_count += batch_samples

    mean /= total_images_count
    std /= total_images_count

    return mean, std
