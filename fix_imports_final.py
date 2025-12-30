import json
import os

target_file = r"f:\ComputerVisionTestNew\notebooks\03_Evaluation_Combined.ipynb"

# --- Reconstruct Source Block (WITHOUT imports from src) ---
import_block = [
    "import sys\n",
    "import os\n",
    "import torch\n",
    "import torch.nn as nn\n",
    "import torch.optim as optim\n",
    "import numpy as np\n",
    "import random\n",
    "import time\n",
    "from torchvision import models, transforms\n",
    "from PIL import Image, ImageOps\n",
    "from torch.utils.data import DataLoader, Dataset\n",
    "from torchvision.datasets import Flowers102\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.metrics import confusion_matrix, classification_report\n",
    "import torchvision.transforms.functional as TF\n",
    "import torch.nn.functional as F\n",
    "\n",
    "# Add src to path (still useful)\n",
    "sys.path.append(os.path.abspath('../'))\n",
    "\n"
]

seeds_code = [
    "# --- Inlined src/utils/seeds.py ---\n",
    "def set_seeds(seed=42):\n",
    "    \"\"\"\n",
    "    Sets the random seed for reproducibility.\n",
    "    \"\"\"\n",
    "    random.seed(seed)\n",
    "    os.environ['PYTHONHASHSEED'] = str(seed)\n",
    "    np.random.seed(seed)\n",
    "    torch.manual_seed(seed)\n",
    "    torch.cuda.manual_seed(seed)\n",
    "    torch.cuda.manual_seed_all(seed)\n",
    "    torch.backends.cudnn.deterministic = True\n",
    "    torch.backends.cudnn.benchmark = False\n",
    "    print(f\"Global seed set to {seed}\")\n",
    "\n"
]

model_code = [
    "# --- Inlined src/models/base_model.py ---\n",
    "def get_model(num_classes=102, fine_tune=True):\n",
    "    \"\"\"\n",
    "    Loads a pre-trained ResNet50 and replaces the final layer.\n",
    "    \"\"\"\n",
    "    try:\n",
    "        from torchvision.models import ResNet50_Weights\n",
    "        weights = ResNet50_Weights.IMAGENET1K_V1\n",
    "        model = models.resnet50(weights=weights)\n",
    "    except ImportError:\n",
    "        model = models.resnet50(pretrained=True)\n",
    "\n",
    "    for param in model.parameters():\n",
    "        param.requires_grad = False\n",
    "\n",
    "    num_ftrs = model.fc.in_features\n",
    "    model.fc = nn.Linear(num_ftrs, num_classes)\n",
    "    return model\n",
    "\n"
]

transforms_code = [
    "# --- Inlined src/data/transforms.py ---\n",
    "class ResizeWithPad:\n",
    "    def __init__(self, size, fill=(255, 255, 255)):\n",
    "        self.size = size\n",
    "        self.fill = fill\n",
    "    def __call__(self, img):\n",
    "        target_size = (self.size, self.size)\n",
    "        img = img.copy()\n",
    "        img.thumbnail(target_size, Image.Resampling.LANCZOS)\n",
    "        delta_w = self.size - img.size[0]\n",
    "        delta_h = self.size - img.size[1]\n",
    "        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))\n",
    "        return ImageOps.expand(img, padding, fill=self.fill)\n",
    "\n",
    "def get_transforms(phase='train', image_size=640):\n",
    "    mean = [0.485, 0.456, 0.406]\n",
    "    std = [0.229, 0.224, 0.225]\n",
    "    if phase == 'train':\n",
    "        return transforms.Compose([\n",
    "            ResizeWithPad(image_size),\n",
    "            transforms.RandomRotation(30),\n",
    "            transforms.RandomHorizontalFlip(),\n",
    "            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),\n",
    "            transforms.ToTensor(),\n",
    "            transforms.Normalize(mean=mean, std=std)\n",
    "        ])\n",
    "    else:\n",
    "        return transforms.Compose([\n",
    "            ResizeWithPad(image_size),\n",
    "            transforms.ToTensor(),\n",
    "            transforms.Normalize(mean=mean, std=std)\n",
    "        ])\n",
    "\n"
]

loader_code = [
    "# --- Inlined src/data/loader.py ---\n",
    "def create_dataloaders(data_dir='./src/data', batch_size=32, num_workers=0 if os.name == 'nt' else 2):\n",
    "    train_transform = get_transforms('train')\n",
    "    val_test_transform = get_transforms('val')\n",
    "    train_dataset = Flowers102(root=data_dir, split='train', download=True, transform=train_transform)\n",
    "    val_dataset = Flowers102(root=data_dir, split='val', download=True, transform=val_test_transform)\n",
    "    test_dataset = Flowers102(root=data_dir, split='test', download=True, transform=val_test_transform)\n",
    "    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)\n",
    "    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)\n",
    "    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)\n",
    "    return train_loader, val_loader, test_loader\n",
    "\n"
]

eval_code = [
    "# --- Inlined src/utils/evaluation.py ---\n",
    "def evaluate_model(model, dataloader, device):\n",
    "    model.eval()\n",
    "    all_preds, all_labels = [], []\n",
    "    with torch.no_grad():\n",
    "        for inputs, labels in dataloader:\n",
    "            inputs = inputs.to(device)\n",
    "            labels = labels.to(device)\n",
    "            outputs = model(inputs)\n",
    "            _, preds = torch.max(outputs, 1)\n",
    "            all_preds.extend(preds.cpu().numpy())\n",
    "            all_labels.extend(labels.cpu().numpy())\n",
    "    return np.array(all_labels), np.array(all_preds)\n",
    "\n",
    "def plot_confusion_matrix(y_true, y_pred, class_names=None):\n",
    "    cm = confusion_matrix(y_true, y_pred)\n",
    "    plt.figure(figsize=(12, 10))\n",
    "    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', cbar=False)\n",
    "    plt.xlabel('Predicted Label')\n",
    "    plt.ylabel('True Label')\n",
    "    plt.title('Confusion Matrix')\n",
    "    plt.show()\n",
    "\n",
    "def print_classification_report(y_true, y_pred, target_names=None):\n",
    "    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))\n",
    "\n",
    "def visualize_misclassifications(model, dataloader, device, num_images=5):\n",
    "    model.eval()\n",
    "    count = 0\n",
    "    fig = plt.figure(figsize=(15, 6))\n",
    "    mean = np.array([0.485, 0.456, 0.406])\n",
    "    std = np.array([0.229, 0.224, 0.225])\n",
    "    with torch.no_grad():\n",
    "        for inputs, labels in dataloader:\n",
    "            inputs = inputs.to(device)\n",
    "            labels = labels.to(device)\n",
    "            outputs = model(inputs)\n",
    "            _, preds = torch.max(outputs, 1)\n",
    "            mistakes = preds != labels\n",
    "            if not mistakes.any():\n",
    "                continue\n",
    "            mistake_indices = torch.where(mistakes)[0]\n",
    "            for idx in mistake_indices:\n",
    "                if count >= num_images:\n",
    "                    break\n",
    "                img = inputs[idx].cpu().permute(1, 2, 0).numpy()\n",
    "                img = std * img + mean\n",
    "                img = np.clip(img, 0, 1)\n",
    "                true_label = labels[idx].item()\n",
    "                pred_label = preds[idx].item()\n",
    "                ax = fig.add_subplot(1, num_images, count + 1)\n",
    "                ax.imshow(img)\n",
    "                ax.set_title(f\"True: {true_label}\\nPred: {pred_label}\", color='red')\n",
    "                ax.axis('off')\n",
    "                count += 1\n",
    "            if count >= num_images:\n",
    "                break\n",
    "    plt.show()\n",
    "\n",
    "def predict_tta(model, dataloader, device, num_augs=5):\n",
    "    model.eval()\n",
    "    all_preds, all_labels = [], []\n",
    "    with torch.no_grad():\n",
    "        for inputs, labels in dataloader:\n",
    "            inputs = inputs.to(device)\n",
    "            labels = labels.to(device)\n",
    "            batch_probs = torch.zeros(inputs.size(0), 102).to(device)\n",
    "            for _ in range(num_augs):\n",
    "                if _ == 0: aug_inputs = inputs\n",
    "                else: \n",
    "                    if torch.rand(1).item() > 0.5: aug_inputs = TF.hflip(inputs)\n",
    "                    else: aug_inputs = inputs\n",
    "                outputs = model(aug_inputs)\n",
    "                probs = F.softmax(outputs, dim=1)\n",
    "                batch_probs += probs\n",
    "            batch_probs /= num_augs\n",
    "            _, preds = torch.max(batch_probs, 1)\n",
    "            all_preds.extend(preds.cpu().numpy())\n",
    "            all_labels.extend(labels.cpu().numpy())\n",
    "    return np.array(all_labels), np.array(all_preds)\n",
    "\n"
]

full_source = import_block + seeds_code + model_code + transforms_code + loader_code + eval_code + ["%matplotlib inline\n", "set_seeds(42)"]

print(f"Processing {target_file}...")
try:
    with open(target_file, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    updated = False
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = cell['source']
            source_text = "".join(source)
            
            # Check if this cell contains imports we want to ban
            has_imports = "from src.data.loader import" in source_text or "from src.utils.evaluation import" in source_text
            
            # If it has imports, regardless of whether it has inline, we OVERWRITE it to be safe.
            # This handles both (Import Only) and (Import + Inline) cases.
            if has_imports:
                print("Found cell with imports. Overwriting with full clean inline code.")
                cell['source'] = full_source
                updated = True
                # We stop after one update because usually there is only one setup cell.
                # If there are multiple, handling the first one usually fixes the 'redefinition' conflict if any.
                break 
    
    if updated:
        with open(target_file, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        print(f"Successfully fixed {target_file}")
    else:
        print("No cells with src imports found.")

except Exception as e:
    print(f"Error processing {target_file}: {e}")
