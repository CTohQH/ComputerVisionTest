# Flower Classification with ResNet50

This project implements an end-to-end pipeline for fine-grained image classification using the **Oxford 102 Flowers** dataset. It leverages Transfer Learning with a pre-trained **ResNet50** backbone to achieve precision in classifying 102 different flower categories.

## 🚀 Key Features

*   **Hybrid Architecture**: Code is structured modularly in `src/` but **inlined** within notebooks. This ensures 100% portability, allowing notebooks to run standalone in **Google Colab** (without mounting Drive) or on a local machine.
*   **Advanced Preprocessing**: Implements a custom **Letterboxing** (padding) strategy to resize images to **640x640** while preserving aspect ratios, preventing distortion.
*   **Transfer Learning**: Uses a **ResNet50** model pre-trained on ImageNet. The backbone is frozen to extract robust features, training only the final classification head.
*   **Test-Time Augmentation (TTA)**: (Available in Evaluation) Boosts reliability by aggregating predictions from multiple augmented views of the same image.
*   **Interactive Demo**: Deployment-ready demonstration using **Gradio**, allowing real-time inference on custom images.

## 📂 Project Structure

```
ComputerVisionTest/
├── notebooks/                  # Standalone Jupyter Notebooks
│   ├── 01_EDA.ipynb            # Data download & exploration
│   ├── 02_Training_Combined.ipynb  # End-to-end Model Training
│   ├── 03_Evaluation_Combined.ipynb # Comprehensive Evaluation
│   └── 04_Demo.ipynb           # Interactive Gradio App
├── src/                        # Source Code (Reference Implementation)
│   ├── data/                   # Dataset & Transforms
│   ├── models/                 # ResNet definition
│   ├── training/               # Loop & Callbacks
│   └── utils/                  # Evaluation metrics & Seeding
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```

## 🛠️ Setup & Installation

### Option 1: Google Colab (Recommended)
1.  **Upload** the `.ipynb` files from the `notebooks/` directory to Colab.
2.  **Run** the cells. The notebooks are self-contained:
    *   They automatically detect the environment.
    *   They download the dataset directly to the Colab instance.
    *   They define all necessary model and training logic internally.
    *   **No separate setup or Drive mounting is required.**

### Option 2: Local Environment
1.  **Clone the repository**.
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run Jupyter**:
    ```bash
    jupyter lab
    ```

## 🚦 Usage Guide

Execute the notebooks in the following sequence:

1.  **`01_EDA.ipynb`**: Downloads the Oxford 102 Flowers dataset and visualizes the class distribution and image samples.
2.  **`02_Training_Combined.ipynb`**: Trains the ResNet50 model.
    *   Initializes the model with ImageNet weights.
    *   Freezes the backbone.
    *   Trains the custom head for 102 classes.
    *   Saves the best weights to `best_model.pt`.
3.  **`03_Evaluation_Combined.ipynb`**: Loads `best_model.pt` and performs detailed analysis.
    *   Calculates Accuracy, F1-Score, and Confusion Matrix.
    *   Visualizes misclassified examples for error analysis.
4.  **`04_Demo.ipynb`**: Starts a local web server (Gradio) to upload and test images interactively.

## 📊 Technical Details

*   **Model**: ResNet50 (Frozen Backbone + Linear Head)
*   **Input Resolution**: 640x640 (Custom ResizeWithPad)
*   **Optimization**: Adam Optimizer with ReduceLROnPlateau scheduler.
*   **Regularization**: Early Stopping based on validation loss.
