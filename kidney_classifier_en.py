#!/usr/bin/env python3
"""
CT Kidney Image Classifier
Deep learning-based kidney disease diagnosis model

Core Functions:
- Multiple CNN architecture support
- Automated training and validation
- Model performance evaluation
- Prediction and inference functionality
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights, efficientnet_b0, EfficientNet_B0_Weights
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import json
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class KidneyDataset(Dataset):
    """Kidney CT image dataset"""
    
    def __init__(self, csv_file, data_dir, transform=None, max_samples=None):
        """
        Args:
            csv_file: Path to CSV annotation file
            data_dir: Image data directory
            transform: Image transformations
            max_samples: Maximum number of samples limit
        """
        self.df = pd.read_csv(csv_file)
        if max_samples:
            self.df = self.df.sample(n=min(max_samples, len(self.df)), random_state=42)
        
        self.data_dir = Path(data_dir)
        self.transform = transform
        
        # Label encoding
        self.label_encoder = LabelEncoder()
        self.df['label'] = self.label_encoder.fit_transform(self.df['diag'])
        self.class_names = self.label_encoder.classes_
        
        print(f"Dataset created: {len(self.df)} samples, {len(self.class_names)} classes")
        print(f"Classes: {list(self.class_names)}")
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Build image path
        image_name = row['image_id'] + '.jpg'
        diagnosis = row['diag']
        
        # Select correct folder based on diagnosis
        if diagnosis == 'Cyst':
            image_path = self.data_dir / 'Cyst' / image_name
        elif diagnosis == 'Normal':
            image_path = self.data_dir / 'Normal' / image_name
        elif diagnosis == 'Tumor':
            image_path = self.data_dir / 'Tumor' / image_name
        elif diagnosis == 'Stone':
            image_path = self.data_dir / 'Stone' / image_name
        else:
            image_path = self.data_dir / 'Cyst' / image_name  # Default path
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            label = row['label']
            return image, label
            
        except Exception as e:
            # If image loading fails, return black image
            print(f"Warning: Unable to load image {image_path}: {e}")
            image = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                image = self.transform(image)
            return image, row['label']

class ResNetClassifier(nn.Module):
    """ResNet-based classifier"""
    
    def __init__(self, num_classes=4, pretrained=True, dropout_rate=0.5):
        super(ResNetClassifier, self).__init__()
        
        # Load pretrained model
        if pretrained:
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            self.backbone = resnet50(weights=None)
        
        # Replace classification head
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate * 0.25),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

class SimpleClassifier(nn.Module):
    """Simple CNN classifier"""
    
    def __init__(self, num_classes=4):
        super(SimpleClassifier, self).__init__()
        
        self.features = nn.Sequential(
            # First convolutional layer
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second convolutional layer
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third convolutional layer
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Fourth convolutional layer
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class KidneyClassifierTrainer:
    """Kidney classifier trainer"""
    
    def __init__(self, model_type='simple', num_classes=4, batch_size=32, 
                 learning_rate=1e-4, num_epochs=20, device=None):
        
        self.model_type = model_type
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Data transformations
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rate': []
        }
        
        print(f"Trainer initialized:")
        print(f"  Model type: {model_type}")
        print(f"  Device: {self.device}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Epochs: {num_epochs}")
    
    def build_model(self):
        """Build model"""
        print(f"🏗️ Building {self.model_type} model...")
        
        if self.model_type == 'resnet':
            model = ResNetClassifier(num_classes=self.num_classes)
        elif self.model_type == 'simple':
            model = SimpleClassifier(num_classes=self.num_classes)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        model = model.to(self.device)
        
        # Optimizer and loss function
        self.optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=3, factor=0.5, verbose=True
        )
        
        # Calculate parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        return model
    
    def prepare_data(self, csv_path, data_dir, train_split=0.8, max_samples=2000):
        """Prepare data"""
        print("📊 Preparing training data...")
        
        # Create full dataset
        full_dataset = KidneyDataset(csv_path, data_dir, self.val_transform, max_samples)
        self.class_names = full_dataset.class_names
        
        # Split train and validation sets
        train_size = int(train_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create separate datasets with proper transforms
        # This is crucial - we need separate dataset instances for different transforms
        train_dataset_with_aug = KidneyDataset(csv_path, data_dir, self.train_transform, max_samples)
        val_dataset_clean = KidneyDataset(csv_path, data_dir, self.val_transform, max_samples)
        
        # Get the indices from the split
        train_indices = train_dataset.indices
        val_indices = val_dataset.indices
        
        # Create subset datasets
        train_subset = torch.utils.data.Subset(train_dataset_with_aug, train_indices)
        val_subset = torch.utils.data.Subset(val_dataset_clean, val_indices)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_subset, batch_size=self.batch_size, 
            shuffle=True, num_workers=0, pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_subset, batch_size=self.batch_size, 
            shuffle=False, num_workers=0, pin_memory=True
        )
        
        print(f"  Training set: {len(train_subset)} samples")
        print(f"  Validation set: {len(val_subset)} samples")
        print(f"  Classes: {list(self.class_names)}")
        
        return train_subset, val_subset
    
    def train_epoch(self, model):
        """Train one epoch"""
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = model(data)
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.1f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, model):
        """Validate model"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100.*correct/total:.1f}%'
                })
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def train_model(self, model):
        """Train model"""
        print("🚀 Starting model training...")
        
        best_val_acc = 0
        best_model_state = None
        patience_counter = 0
        max_patience = 5
        
        for epoch in range(self.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.num_epochs}")
            print("-" * 50)
            
            # Training
            train_loss, train_acc = self.train_epoch(model)
            
            # Validation
            val_loss, val_acc = self.validate(model)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rate'].append(current_lr)
            
            print(f"Training - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%")
            print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.2e}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                
                # Save checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'class_names': self.class_names
                }, f'best_kidney_classifier_{self.model_type}_en.pth')
                
                print(f"✅ Saved best model (validation accuracy: {val_acc:.2f}%)")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= max_patience:
                print(f"Early stopping triggered (no improvement for {max_patience} epochs)")
                break
        
        # Load best model
        if best_model_state:
            model.load_state_dict(best_model_state)
        
        print(f"\n🎉 Training completed! Best validation accuracy: {best_val_acc:.2f}%")
        return model, best_val_acc
    
    def evaluate_model(self, model):
        """Detailed model evaluation"""
        print("\n📊 Detailed model evaluation...")
        
        model.eval()
        all_preds = []
        all_targets = []
        all_probs = []
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Evaluating"):
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                probs = F.softmax(output, dim=1)
                
                _, predicted = output.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='weighted')
        
        # Classification report
        report = classification_report(
            all_targets, all_preds, 
            target_names=self.class_names,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_preds)
        
        print(f"Overall accuracy: {accuracy:.4f}")
        print(f"Weighted F1 score: {f1:.4f}")
        print("\nDetailed classification report:")
        print(classification_report(all_targets, all_preds, target_names=self.class_names))
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': all_preds,
            'targets': all_targets,
            'probabilities': all_probs
        }
    
    def create_evaluation_plots(self, results):
        """Create evaluation visualizations"""
        print("📈 Generating evaluation visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{self.model_type.upper()} Kidney Disease Classifier Evaluation Report', fontsize=16, fontweight='bold')
        
        # 1. Training history - Loss
        epochs = range(1, len(self.history['train_loss']) + 1)
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Training History - Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Training history - Accuracy
        axes[0, 1].plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
        axes[0, 1].plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
        axes[0, 1].set_title('Training History - Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy (%)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Learning rate schedule
        axes[0, 2].plot(epochs, self.history['learning_rate'], 'g-', linewidth=2)
        axes[0, 2].set_title('Learning Rate Schedule')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Learning Rate')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Confusion matrix
        sns.heatmap(results['confusion_matrix'], 
                   annot=True, fmt='d', 
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   ax=axes[1, 0], cmap='Blues')
        axes[1, 0].set_title('Confusion Matrix')
        axes[1, 0].set_xlabel('Predicted Label')
        axes[1, 0].set_ylabel('True Label')
        
        # 5. Per-class performance metrics
        metrics_data = []
        for class_name in self.class_names:
            if class_name in results['classification_report']:
                metrics_data.append([
                    results['classification_report'][class_name]['precision'],
                    results['classification_report'][class_name]['recall'],
                    results['classification_report'][class_name]['f1-score']
                ])
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data, 
                                    columns=['Precision', 'Recall', 'F1-score'],
                                    index=self.class_names)
            
            x = np.arange(len(self.class_names))
            width = 0.25
            
            axes[1, 1].bar(x - width, metrics_df['Precision'], width, label='Precision', alpha=0.8)
            axes[1, 1].bar(x, metrics_df['Recall'], width, label='Recall', alpha=0.8)
            axes[1, 1].bar(x + width, metrics_df['F1-score'], width, label='F1-score', alpha=0.8)
            
            axes[1, 1].set_title('Per-class Performance Metrics')
            axes[1, 1].set_xlabel('Class')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(self.class_names, rotation=45)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Model summary info
        summary_info = [
            f"Model type: {self.model_type.upper()}",
            f"Final accuracy: {results['accuracy']:.4f}",
            f"Weighted F1 score: {results['f1_score']:.4f}",
            f"Training samples: {len(self.train_loader.dataset)}",
            f"Validation samples: {len(self.val_loader.dataset)}",
            f"Number of classes: {self.num_classes}",
            f"Training epochs: {len(self.history['train_loss'])}",
            f"Best validation accuracy: {max(self.history['val_acc']):.2f}%"
        ]
        
        axes[1, 2].text(0.1, 0.5, '\n'.join(summary_info),
                       fontsize=11, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].set_title('Model Summary')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'kidney_classifier_{self.model_type}_evaluation_en.png', dpi=300, bbox_inches='tight')
        print(f"  ✅ Evaluation report saved as 'kidney_classifier_{self.model_type}_evaluation_en.png'")
        
        return fig
    
    def save_training_report(self, results, best_val_acc):
        """Save training report"""
        print("💾 Saving training report...")
        
        report = {
            'model_info': {
                'model_type': self.model_type,
                'num_classes': self.num_classes,
                'class_names': list(self.class_names),
                'total_params': 'N/A'  # Simplified handling
            },
            'training_config': {
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'num_epochs': self.num_epochs,
                'device': str(self.device)
            },
            'results': {
                'best_validation_accuracy': best_val_acc,
                'final_accuracy': results['accuracy'],
                'f1_score': results['f1_score'],
                'training_history': self.history
            },
            'performance_metrics': results['classification_report'],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(f'kidney_classifier_{self.model_type}_report_en.json', 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"  ✅ Training report saved as 'kidney_classifier_{self.model_type}_report_en.json'")
        return report

def main():
    """Main function"""
    print("=" * 60)
    print("🧠 CT Kidney Image Classifier")
    print("   Children's Hospital of Philadelphia")
    print("   Imaging Data Analyst Internship Demo")
    print("=" * 60)
    
    # Configure paths
    csv_path = "G:/Shawn/kidneyData.csv"
    data_dir = "G:/Shawn/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
    
    # Model selection
    model_types = ['simple']  # Start with simple model for demo
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Training {model_type.upper()} Model")
        print(f"{'='*60}")
        
        # Create trainer
        trainer = KidneyClassifierTrainer(
            model_type=model_type,
            batch_size=16,  # Smaller batch size
            learning_rate=1e-4,
            num_epochs=10,  # Fewer epochs for demo
        )
        
        try:
            # Prepare data
            trainer.prepare_data(csv_path, data_dir, max_samples=1000)  # Limit sample size
            
            # Build model
            model = trainer.build_model()
            
            # Train model
            trained_model, best_val_acc = trainer.train_model(model)
            
            # Evaluate model
            results = trainer.evaluate_model(trained_model)
            
            # Create visualizations
            trainer.create_evaluation_plots(results)
            
            # Save report
            trainer.save_training_report(results, best_val_acc)
            
            print(f"\n✅ {model_type.upper()} model training completed!")
            print(f"Best validation accuracy: {best_val_acc:.2f}%")
            print(f"Final test accuracy: {results['accuracy']:.4f}")
            
        except Exception as e:
            print(f"❌ {model_type.upper()} model training failed: {e}")
            continue
    
    print("\n🎉 All model training completed!")
    print("\n💡 Generated files:")
    print("  🤖 best_kidney_classifier_*_en.pth - Trained models")
    print("  📊 kidney_classifier_*_evaluation_en.png - Model evaluation reports")
    print("  📄 kidney_classifier_*_report_en.json - Training reports")

if __name__ == "__main__":
    main()
