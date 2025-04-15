import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import random

# Import our NuScenes dataset module
from nuscenes_dataset import NuScenesDataset, download_nuscenes_mini, NUSCENES_AVAILABLE

# Define a vision transformer model for autonomous driving
class AutonomousDrivingTransformer(nn.Module):
    def __init__(self, num_classes=5):  # 5 classes for scene type classification
        super(AutonomousDrivingTransformer, self).__init__()
        
        # Use a pretrained ViT model as the backbone
        self.backbone = torch.hub.load('facebookresearch/deit:main', 
                                      'deit_tiny_patch16_224', 
                                      pretrained=True)
        
        # Replace the classification head
        hidden_size = self.backbone.head.in_features
        
        # Add a more complex classification head with dropout for better generalization
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden_size, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Replace the original head with identity to use our own head
        self.backbone.head = nn.Identity()
        
    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        features = F.relu(self.fc1(features))
        features = self.dropout(features)
        output = self.fc2(features)
        return output

# Custom dataset class for autonomous driving
class AutonomousDrivingDataset(Dataset):
    def __init__(self, data_dir, labels_file, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Load labels from file
        # Format: image_name,label_class
        import csv
        self.samples = []
        with open(labels_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 2:  # Make sure row has image path and label
                    img_path = self.data_dir / row[0]
                    label = int(row[1])
                    self.samples.append((img_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Enhanced augmentations for better generalization
def get_training_augmentations():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
        # Add more aggressive augmentations for intersection scenes specifically 
        transforms.RandomApply([
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Small translations
        ], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # Add random erasing to simulate occlusions
        transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))
    ])

# Analyze the class distribution in the dataset
def analyze_dataset(dataset):
    class_counts = {}
    for _, label in dataset:
        class_counts[label] = class_counts.get(label, 0) + 1
    
    total = sum(class_counts.values())
    
    if total == 0:
        print("\nWARNING: Dataset is empty!")
        return class_counts
    
    print("\nClass distribution in dataset:")
    for class_id, count in sorted(class_counts.items()):
        class_name = dataset.class_names[class_id] if hasattr(dataset, 'class_names') and class_id < len(dataset.class_names) else f"Class {class_id}"
        percentage = (count/total*100) if total > 0 else 0
        print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
    
    # Check for class imbalance
    non_zero_counts = [count for count in class_counts.values() if count > 0]
    if not non_zero_counts:  # All classes have zero samples
        print("\nWARNING: All classes have zero samples!")
        return class_counts
        
    min_count = min(non_zero_counts)
    max_count = max(non_zero_counts)
    
    # Report imbalance
    if min_count > 0 and max_count / min_count > 1.5:  # Changed threshold to 1.5
        print(f"\nWARNING: Class imbalance detected!")
        print(f"Max/Min ratio: {max_count/min_count:.2f}")
        print("Using weighted sampling to address this issue.\n")
    
    return class_counts

# Create sampler to balance classes
def create_weighted_sampler(dataset):
    # Count each class
    class_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    labels = []
    
    # Iterate through the dataset
    for i in range(len(dataset)):
        _, label = dataset[i]
        class_counts[label] = class_counts.get(label, 0) + 1
        labels.append(label)
    
    # Calculate weights, making sure to handle zero counts
    weights = []
    for label in labels:
        # Default weight is 1.0
        class_weight = 1.0
        
        # If there are samples in this class, calculate weight
        if class_counts[label] > 0:
            # Find max count of any class to scale weights appropriately
            max_count = max(class_counts.values())
            # Set weight higher for classes with fewer samples
            class_weight = max_count / class_counts[label]
            
        weights.append(class_weight)
    
    # Convert to tensor
    weights = torch.FloatTensor(weights)
    
    # Create sampler
    sampler = WeightedRandomSampler(weights, len(weights))
    
    return sampler

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Focal Loss to better handle class imbalance
    class FocalLoss(nn.Module):
        def __init__(self, alpha=1, gamma=2, reduction='mean'):
            super(FocalLoss, self).__init__()
            self.alpha = alpha
            self.gamma = gamma
            self.reduction = reduction
            
        def forward(self, inputs, targets):
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
            
            if self.reduction == 'mean':
                return focal_loss.mean()
            elif self.reduction == 'sum':
                return focal_loss.sum()
            else:
                return focal_loss
    
    # Combine Cross Entropy and Focal Loss
    criterion = FocalLoss(gamma=2.0)
    
    # AdamW optimizer with weight decay and OneCycleLR scheduler for faster convergence
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=learning_rate*10,
        steps_per_epoch=len(train_loader),
        epochs=num_epochs
    )
    
    best_val_acc = 0.0
    best_val_epoch = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    class_accuracies = {0: [], 1: [], 2: [], 3: [], 4: []}  # Track per-class accuracies
    
    # Early stopping parameters
    patience = 5
    early_stop_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        class_correct = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        class_total = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Track per-class accuracy
            for i in range(len(labels)):
                label = labels[i].item()
                pred = predicted[i].item()
                class_total[label] = class_total.get(label, 0) + 1
                if label == pred:
                    class_correct[label] = class_correct.get(label, 0) + 1
            
            progress_bar.set_postfix({"loss": loss.item(), "acc": 100 * correct / total})
            
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Calculate per-class training accuracy
        train_class_acc = {}
        for c in class_total:
            if class_total[c] > 0:
                acc = 100 * class_correct[c] / class_total[c]
                train_class_acc[c] = acc
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_class_correct = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        val_class_total = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            for inputs, labels in progress_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Track per-class accuracy on val set
                for i in range(len(labels)):
                    label = labels[i].item()
                    pred = predicted[i].item()
                    val_class_total[label] = val_class_total.get(label, 0) + 1
                    if label == pred:
                        val_class_correct[label] = val_class_correct.get(label, 0) + 1
                
                progress_bar.set_postfix({"loss": loss.item(), "acc": 100 * val_correct / val_total})
                
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Calculate per-class validation accuracy
        val_class_acc = {}
        for c in val_class_total:
            if val_class_total[c] > 0:
                acc = 100 * val_class_correct[c] / val_class_total[c]
                val_class_acc[c] = acc
                class_accuracies[c].append(acc)
        
        # Print overall and per-class accuracy
        print(f"\nEpoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        print("Per-class validation accuracy:")
        lowest_class = None
        lowest_acc = 100.0
        for c in sorted(val_class_acc.keys()):
            class_name = "Intersection" if c == 0 else "Highway" if c == 1 else "Urban Road" if c == 2 else "Parking Lot" if c == 3 else "Tunnel"
            print(f"  Class {c} ({class_name}): {val_class_acc[c]:.2f}%")
            
            # Track the lowest performing class
            if val_class_acc[c] < lowest_acc:
                lowest_acc = val_class_acc[c]
                lowest_class = c
        
        # Save best model (based on overall validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_epoch = epoch + 1
            save_model(model, 'best_driving_transformer.pth')
            early_stop_counter = 0
            print(f"New best model saved! Validation accuracy: {val_acc:.2f}%")
        else:
            early_stop_counter += 1
            
        # Early stopping
        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
            
        # If class 0 (Intersection) accuracy is low, increase its importance
        if 0 in val_class_acc and val_class_acc[0] < 70 and epoch > 5:
            print("Intersection class accuracy is low. Adjusting the focal loss gamma parameter...")
            criterion = FocalLoss(gamma=3.0)  # Increase gamma for harder examples
            
    # Plot training history
    plt.figure(figsize=(15, 10))
    
    # Plot overall accuracy and loss
    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    # Plot per-class accuracy
    plt.subplot(2, 2, 3)
    class_names = ["Intersection", "Highway", "Urban Road", "Parking Lot", "Tunnel"]
    for c in class_accuracies:
        if class_accuracies[c]:
            plt.plot(class_accuracies[c], label=f'Class {c} ({class_names[c]})')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy (%)')
    plt.legend()
    
    # Add confusion matrix from the last validation run
    plt.subplot(2, 2, 4)
    plt.text(0.1, 0.5, f"Best model at epoch {best_val_epoch}\nBest validation accuracy: {best_val_acc:.2f}%", 
             fontsize=12)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('training_history_detailed.png')
    
    print(f"\nTraining completed. Best validation accuracy: {best_val_acc:.2f}% at epoch {best_val_epoch}")
    
    # Load the best model for final evaluation
    try:
        best_model_path = 'best_driving_transformer.pth'
        state_dict = torch.load(best_model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded best model from {best_model_path}")
    except:
        print("Couldn't load best model. Using the current model.")
    
    return model

def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

if __name__ == "__main__":
    print("Starting autonomous driving transformer training...")
    
    # Point to where the NuScenes dataset is located
    nuscenes_dir = './data/nuscenes'
    
    if not os.path.exists(nuscenes_dir):
        print(f"NuScenes directory {nuscenes_dir} not found!")
        print("Please specify the correct path to your downloaded NuScenes data.")
        nuscenes_dir = input("Enter NuScenes data directory path: ")
        if not nuscenes_dir or not os.path.exists(nuscenes_dir):
            print("Invalid path. Exiting.")
            exit(1)
    
    # Check for required NuScenes subdirectories
    required_dirs = ['v1.0-mini', 'v1.0-mini/samples', 'v1.0-mini/maps']
    missing_dirs = []
    for req_dir in required_dirs:
        if not os.path.exists(os.path.join(nuscenes_dir, req_dir)):
            missing_dirs.append(req_dir)
    
    if missing_dirs:
        print(f"Warning: The following required directories are missing: {', '.join(missing_dirs)}")
        print("The NuScenes dataset might not be properly extracted or structured.")
        print("Continue anyway? (y/n)")
        choice = input().lower()
        if choice != 'y':
            print("Exiting.")
            exit(1)
    
    print(f"Using NuScenes dataset from {nuscenes_dir}")
    
    # Data transforms with enhanced augmentation for better generalization
    train_transform = get_training_augmentations()
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    try:
        print("Loading NuScenes dataset...")
        # Create NuScenes datasets with real data only
        train_dataset = NuScenesDataset(nuscenes_dir, version='v1.0-mini', 
                                      split='train', transform=train_transform)
        
        print(f"Loaded {len(train_dataset)} training samples")
        
        # Get number of classes from the dataset
        num_classes = len(train_dataset.class_names)
        print(f"Detected {num_classes} classes from dataset")
        
        # Analyze the class distribution
        class_counts = analyze_dataset(train_dataset)
        
        # Always use weighted sampler for real data as it typically has imbalance
        train_sampler = create_weighted_sampler(train_dataset)
        shuffle = False  # Don't shuffle when using a sampler
        
        # Create validation dataset
        try:
            val_dataset = NuScenesDataset(nuscenes_dir, version='v1.0-mini',
                                        split='val', transform=val_transform)
            print(f"Loaded {len(val_dataset)} validation samples")
        except:
            # If validation split isn't available, use a portion of the training data
            print("Validation split not found. Using a portion of training data for validation.")
            dataset_size = len(train_dataset)
            
            # Use stratified split to maintain class distribution
            labels = [sample['class_id'] for sample in train_dataset.samples]
            train_indices, val_indices = train_test_split(
                range(dataset_size),
                test_size=0.2,
                stratify=labels,
                random_state=42
            )
            
            # Create train and validation datasets
            train_dataset_split = torch.utils.data.Subset(train_dataset, train_indices)
            val_dataset = torch.utils.data.Subset(train_dataset, val_indices)
            
            # Update train_dataset to the split version
            train_dataset = train_dataset_split
            
            print(f"Split dataset: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
        
        # Create data loaders - use the weighted sampler
        batch_size = 16  # Adjust based on your GPU memory
        train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                               sampler=train_sampler, shuffle=False, 
                               num_workers=4)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                             shuffle=False, num_workers=4)
        
        # Create model with architecture suited to the number of classes
        model = AutonomousDrivingTransformer(num_classes=num_classes)
        
        print("Starting model training...")
        # Train model for more epochs with lower learning rate for better fine-tuning
        train_model(model, train_loader, val_loader, num_epochs=30, learning_rate=1e-4)
        
        # Save final model
        save_model(model, 'final_driving_transformer.pth')
        
        print("Training completed successfully!")
        
    except Exception as e:
        print(f"Error training with NuScenes dataset: {e}")
        print("Please check if your NuScenes dataset is correctly downloaded and extracted.")
        print("The expected structure is:")
        print("  data/nuscenes/v1.0-mini/")
        print("    ├── samples/")
        print("    │   ├── CAM_FRONT/")
        print("    │   └── ...")
        print("    ├── maps/")
        print("    ├── scene.json")
        print("    ├── sample.json")
        print("    └── ...")
        
        import traceback
        traceback.print_exc()