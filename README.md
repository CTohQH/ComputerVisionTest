# Flower Classification with ResNet50

This project implements an end-to-end pipeline for fine-grained image classification using the **Oxford 102 Flowers** dataset. It leverages Transfer Learning with a pre-trained **ResNet50** backbone to achieve high accuracy.

## 🚀 Key Features

*   **Advanced Preprocessing**: Implements **Letterboxing** (padding) to resize images to **640x640** without distortion, preserving aspect ratios.
*   **Transfer Learning**: Fine-tunes a ResNet50 model pre-trained on ImageNet.
*   **Test-Time Augmentation (TTA)**: Boosts inference accuracy by averaging predictions across multiple augmented views of the same image.
*   **Google Colab Ready**: Notebooks are pre-configured to run seamlessly in Google Colab with Drive mounting.
*   **Interactive Demo**: Includes a **Gradio** web interface for real-time inference on user-uploaded images.
*   **Reproducibility**: Global seeding ensures consistent results across runs.

## 📂 Project Structure

```
ComputerVisionTest/
├── configs/
│   └── config.yaml             # Configuration for hyperparameters and paths
├── notebooks/                  # Jupyter Notebooks for each stage
│   ├── 01_EDA.ipynb            # Exploratory Data Analysis
│   ├── 02_Training.ipynb       # Initial Training (Frozen Backbone)
│   ├── 03_Evaluation...ipynb   # Evaluation of Initial Model
│   ├── 04_Fine_Tuning.ipynb    # Fine-Tuning (Unfrozen Backbone)
│   ├── 05_Evaluation_Fine...   # Final Evaluation with TTA
│   └── 06_Demo.ipynb           # Interactive Gradio App
├── src/                        # Source Code
│   ├── data/
│   │   ├── loader.py           # Data loading logic
│   │   └── transforms.py       # Custom transforms (ResizeWithPad)
│   ├── models/
│   │   └── base_model.py       # ResNet50 Model definition
│   ├── training/
│   │   ├── trainer.py          # Training loop
│   │   └── callbacks.py        # Early Stopping
│   └── utils/
│       ├── evaluation.py       # Metrics and Visualization tools
│       └── seeds.py            # Reproducibility utilities
├── requirements.txt            # Python dependencies
└── README.md                   # Project Documentation
```

## 🛠️ Setup & Installation

### Option 1: Local Environment

1.  **Clone the repository** (or unzip the project folder).
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run Notebooks**: Start Jupyter Lab or Jupyter Notebook:
    ```bash
    jupyter lab
    ```

### Option 2: Google Colab

1.  **Upload**: Upload the entire `ComputerVisionTest` folder to your Google Drive.
2.  **Open Notebooks**: Open any notebook (e.g., `notebooks/04_Fine_Tuning.ipynb`) in Colab.
3.  **Mount Drive**: Follow the instructions in the first cell of the notebook to mount your drive and set the project root path.

## 🚦 Usage Guide

Run the notebooks in the following order to reproduce the results:

1.  **`01_EDA.ipynb`**: Explore the dataset statistics and visualize sample images.
2.  **`02_Training.ipynb`**: Train the head of the model while keeping the backbone frozen. This creates `best_model.pt`.
3.  **`04_Fine_Tuning.ipynb`**: Load `best_model.pt`, unfreeze the backbone, and fine-tune with a lower learning rate and 640x640 resolution. This saves `best_model_finetuned.pt`.
4.  **`05_Evaluation_FineTuned.ipynb`**: Evaluate the fine-tuned model using standard metrics and **Test-Time Augmentation (TTA)**.
5.  **`06_Demo.ipynb`**: Launch the Gradio app to test the model with your own flower images.

## 📊 Performance Notes

*   **Resolution**: 640x640 (Squared with White Padding)
*   **Batch Size**: Reduced to 8 for fine-tuning to prevent GPU OOM errors on standard instances (e.g., Tesla T4).
