#!/usr/bin/env python3
"""
CT Kidney Image Analysis - Realistic Demo
Focused demonstration of data preprocessing and machine learning core functions

The previous 100% accuracy was unrealistic due to:
1. Using image ID hash as features (data leakage)
2. Small sample size with perfect separation
3. Target variable included in features

This version provides realistic results with proper validation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def extract_realistic_features(df, data_dir):
    """
    Extract realistic features from medical images
    
    This function extracts actual image-based features instead of using
    metadata that could cause data leakage.
    """
    print("🔬 Extracting realistic image features...")
    
    features = []
    labels = []
    processed_count = 0
    failed_count = 0
    
    for idx, row in df.iterrows():
        try:
            image_name = row['image_id'] + '.jpg'
            diagnosis = row['diag']
            
            # Select correct folder based on diagnosis
            if diagnosis == 'Cyst':
                image_path = data_dir / 'Cyst' / image_name
            elif diagnosis == 'Normal':
                image_path = data_dir / 'Normal' / image_name
            elif diagnosis == 'Tumor':
                image_path = data_dir / 'Tumor' / image_name
            elif diagnosis == 'Stone':
                image_path = data_dir / 'Stone' / image_name
            else:
                continue
            
            if image_path.exists():
                # Load and process image
                with Image.open(image_path) as img:
                    img = img.convert('RGB')
                    img = img.resize((64, 64))  # Smaller size for speed
                    
                    # Extract simple statistical features
                    img_array = np.array(img)
                    
                    # Color channel statistics
                    mean_r = np.mean(img_array[:,:,0])
                    mean_g = np.mean(img_array[:,:,1])
                    mean_b = np.mean(img_array[:,:,2])
                    
                    std_r = np.std(img_array[:,:,0])
                    std_g = np.std(img_array[:,:,1])
                    std_b = np.std(img_array[:,:,2])
                    
                    # Overall brightness and contrast
                    brightness = np.mean(img_array)
                    contrast = np.std(img_array)
                    
                    # Texture features (simplified)
                    gray = np.mean(img_array, axis=2)
                    grad_x = np.abs(np.diff(gray, axis=1)).mean()
                    grad_y = np.abs(np.diff(gray, axis=0)).mean()
                    
                    # Edge density
                    edges = grad_x + grad_y
                    
                    # Feature vector
                    feature_vector = [
                        mean_r, mean_g, mean_b,
                        std_r, std_g, std_b,
                        brightness, contrast,
                        grad_x, grad_y, edges
                    ]
                    
                    features.append(feature_vector)
                    labels.append(diagnosis)
                    processed_count += 1
            else:
                failed_count += 1
                
        except Exception as e:
            failed_count += 1
            continue
        
        # Show progress
        if (idx + 1) % 50 == 0:
            print(f"  Processed {idx + 1}/{len(df)} samples...")
    
    print(f"✅ Feature extraction completed:")
    print(f"  Successfully processed: {processed_count} images")
    print(f"  Failed to process: {failed_count} images")
    
    return np.array(features), np.array(labels)

def evaluate_model_properly(y_true, y_pred, class_names):
    """
    Proper model evaluation with realistic metrics
    """
    accuracy = accuracy_score(y_true, y_pred)
    
    # Classification report
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"📊 Model Performance:")
    print(f"  Overall Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
    print(f"  Macro Avg F1-Score: {report['macro avg']['f1-score']:.3f}")
    print(f"  Weighted Avg F1-Score: {report['weighted avg']['f1-score']:.3f}")
    
    print(f"\n📋 Per-class Performance:")
    for class_name in class_names:
        if class_name in report:
            precision = report[class_name]['precision']
            recall = report[class_name]['recall']
            f1 = report[class_name]['f1-score']
            print(f"  {class_name}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    
    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm
    }

def main():
    print("=" * 70)
    print("🏥 CT Kidney Image Analysis - Realistic Demo")
    print("   Children's Hospital of Philadelphia")
    print("   Imaging Data Analyst Internship Demo")
    print("=" * 70)
    
    # 1. Data Loading and Preprocessing
    print("\n📊 Step 1: Data Loading and Preprocessing")
    print("-" * 40)
    
    csv_path = "G:/Shawn/kidneyData.csv"
    data_dir = Path("G:/Shawn/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone")
    
    # Load data
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded data: {len(df)} records")
    
    # Show class distribution
    class_counts = df['diag'].value_counts()
    print(f"📈 Class Distribution:")
    for class_name, count in class_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {class_name}: {count} ({percentage:.1f}%)")
    
    # Data validation (check small sample)
    print(f"\n🔍 Validating data integrity (first 100 samples)...")
    valid_count = 0
    
    for idx, row in df.head(100).iterrows():
        image_name = row['image_id'] + '.jpg'
        diagnosis = row['diag']
        
        # Select correct folder
        subfolder = diagnosis if diagnosis in ['Cyst', 'Normal', 'Tumor', 'Stone'] else 'Cyst'
        image_path = data_dir / subfolder / image_name
        
        if image_path.exists():
            valid_count += 1
    
    print(f"✅ Data validation: {valid_count}/100 files exist ({valid_count}%)")
    
    # 2. Realistic Feature Extraction
    print(f"\n🔬 Step 2: Realistic Feature Extraction")
    print("-" * 40)
    
    # Use a reasonable sample size for demonstration
    sample_df = df.sample(n=min(400, len(df)), random_state=42)
    print(f"📦 Using sample: {len(sample_df)} records for realistic processing time")
    
    # Extract features from actual images
    X, y = extract_realistic_features(sample_df, data_dir)
    
    if len(X) == 0:
        print("❌ No features extracted. Check data paths and file availability.")
        return False
    
    print(f"📊 Feature extraction results:")
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Number of samples: {len(y)}")
    print(f"  Feature dimension: {X.shape[1] if len(X.shape) > 1 else 0}")
    
    # Label encoding
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    class_names = label_encoder.classes_
    
    print(f"🏷️ Label encoding:")
    for i, class_name in enumerate(class_names):
        count = (y_encoded == i).sum()
        print(f"  {class_name} -> {i} ({count} samples)")
    
    # 3. Dataset Splitting
    print(f"\n✂️ Step 3: Dataset Splitting")
    print("-" * 40)
    
    # Proper train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42
    )
    
    print(f"✅ Dataset splitting:")
    print(f"  Training set: {len(X_train)} samples")
    print(f"  Test set: {len(X_test)} samples")
    print(f"  Feature dimensions: {X_train.shape[1]}")
    
    # 4. Machine Learning Model Training
    print(f"\n🧠 Step 4: Machine Learning Model Training")
    print("-" * 40)
    
    # Train Random Forest classifier
    print("🌲 Training Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42,
        class_weight='balanced'  # Handle class imbalance
    )
    
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    
    # Calculate realistic accuracies
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print(f"✅ Model training completed:")
    print(f"  Training accuracy: {train_accuracy:.3f} ({train_accuracy*100:.1f}%)")
    print(f"  Test accuracy: {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")
    
    # Check for overfitting
    if train_accuracy - test_accuracy > 0.15:
        print("⚠️  Warning: Possible overfitting detected (train >> test accuracy)")
    elif test_accuracy > 0.90:
        print("⚠️  Warning: Suspiciously high accuracy - check for data leakage")
    else:
        print("✅ Model shows realistic performance")
    
    # 5. Detailed Model Evaluation
    print(f"\n📊 Step 5: Detailed Model Evaluation")
    print("-" * 40)
    
    results = evaluate_model_properly(y_test, y_pred_test, class_names)
    
    # Feature importance analysis
    feature_names = ['Mean_R', 'Mean_G', 'Mean_B', 'Std_R', 'Std_G', 'Std_B', 
                    'Brightness', 'Contrast', 'Grad_X', 'Grad_Y', 'Edge_Density']
    
    feature_importance = clf.feature_importances_
    print(f"\n🔍 Top 5 Most Important Features:")
    for idx in np.argsort(feature_importance)[-5:][::-1]:
        print(f"  {feature_names[idx]}: {feature_importance[idx]:.3f}")
    
    # 6. Results Visualization
    print(f"\n📈 Step 6: Results Visualization")
    print("-" * 40)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('CT Kidney Image Analysis - Realistic Demo Results', fontsize=14, fontweight='bold')
    
    # 1. Original data class distribution
    original_counts = df['diag'].value_counts()
    axes[0, 0].pie(original_counts.values, labels=original_counts.index, autopct='%1.1f%%')
    axes[0, 0].set_title('Original Dataset Class Distribution')
    
    # 2. Sample data class distribution
    sample_counts = pd.Series(y).value_counts()
    sample_counts.plot(kind='bar', ax=axes[0, 1])
    axes[0, 1].set_title('Processed Sample Class Distribution')
    axes[0, 1].set_xlabel('Class')
    axes[0, 1].set_ylabel('Sample Count')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. Feature importance
    top_features_idx = np.argsort(feature_importance)[-8:]
    axes[1, 0].barh(range(len(top_features_idx)), feature_importance[top_features_idx])
    axes[1, 0].set_yticks(range(len(top_features_idx)))
    axes[1, 0].set_yticklabels([feature_names[i] for i in top_features_idx])
    axes[1, 0].set_title('Feature Importance (Top 8)')
    axes[1, 0].set_xlabel('Importance')
    
    # 4. Confusion matrix
    import seaborn as sns
    sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', 
                xticklabels=class_names, yticklabels=class_names, 
                ax=axes[1, 1], cmap='Blues')
    axes[1, 1].set_title('Confusion Matrix')
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('realistic_demo_report_en.png', dpi=300, bbox_inches='tight')
    print(f"✅ Visualization report saved: realistic_demo_report_en.png")
    
    # 7. Generate Final Report
    print(f"\n📄 Step 7: Generate Project Report")
    print("-" * 40)
    
    # Create comprehensive report
    final_report = {
        'project_info': {
            'name': 'CT Kidney Image Analysis - Realistic Demo',
            'target_position': 'Children\'s Hospital of Philadelphia - Imaging Data Analyst Internship',
            'completion_time': datetime.now().isoformat(),
            'methodology': 'Image-based feature extraction with Random Forest classification'
        },
        'data_summary': {
            'total_dataset_size': len(df),
            'processed_samples': len(y),
            'classes': list(class_names),
            'class_distribution': {str(k): int(v) for k, v in pd.Series(y).value_counts().to_dict().items()},
            'feature_dimension': X.shape[1],
            'train_test_split': f"{len(X_train)}/{len(X_test)}"
        },
        'model_performance': {
            'training_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'generalization_gap': float(train_accuracy - test_accuracy),
            'classification_metrics': {
                'macro_avg_f1': float(results['classification_report']['macro avg']['f1-score']),
                'weighted_avg_f1': float(results['classification_report']['weighted avg']['f1-score'])
            },
            'per_class_performance': {
                class_name: {
                    'precision': float(results['classification_report'][class_name]['precision']),
                    'recall': float(results['classification_report'][class_name]['recall']),
                    'f1_score': float(results['classification_report'][class_name]['f1-score'])
                } for class_name in class_names if class_name in results['classification_report']
            }
        },
        'technical_methodology': {
            'feature_extraction': [
                'Color channel statistics (RGB means and standard deviations)',
                'Brightness and contrast measurements',
                'Gradient-based texture features',
                'Edge density analysis'
            ],
            'model_type': 'Random Forest Classifier',
            'model_parameters': {
                'n_estimators': 100,
                'max_depth': 10,
                'class_weight': 'balanced',
                'random_state': 42
            },
            'validation_approach': 'Stratified train-test split (70-30%)'
        },
        'quality_assurance': {
            'data_leakage_prevention': 'Image-based features only, no metadata dependencies',
            'overfitting_check': 'Training vs test accuracy comparison',
            'realistic_performance': f'Test accuracy {test_accuracy:.1%} indicates realistic results',
            'class_balance_handling': 'Balanced class weights in Random Forest'
        },
        'key_findings': [
            f'Achieved {test_accuracy:.1%} test accuracy on real medical image features',
            f'Model shows {"good generalization" if train_accuracy - test_accuracy < 0.15 else "signs of overfitting"}',
            f'Most important features: {", ".join([feature_names[i] for i in np.argsort(feature_importance)[-3:][::-1]])}',
            f'Successfully processed {len(y)} medical images with {(len(y)/len(sample_df)*100):.1f}% success rate'
        ],
        'chop_relevance': {
            'skills_demonstrated': [
                'Medical image feature extraction and analysis',
                'Machine learning model development and validation',
                'Proper experimental design and quality control',
                'Statistical analysis and performance evaluation',
                'Data visualization and reporting',
                'Python scientific computing ecosystem'
            ],
            'translational_impact': [
                'Automated kidney disease classification',
                'Scalable image processing workflows',
                'Quality-controlled machine learning pipelines',
                'Reproducible research methodology'
            ]
        }
    }
    
    with open('realistic_demo_summary_en.json', 'w', encoding='utf-8') as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)
    
    print(f"✅ Project report saved: realistic_demo_summary_en.json")
    
    # Final Summary
    print(f"\n" + "="*70)
    print("🎯 Demo Completion Summary")
    print("="*70)
    print(f"✅ Data Loading: {len(df):,} total records, {len(y)} processed")
    print(f"✅ Feature Extraction: 11-dimensional image-based features")
    print(f"✅ Data Preprocessing: Proper train-test split with stratification")
    print(f"✅ Machine Learning: Random Forest with realistic {test_accuracy:.1%} accuracy")
    print(f"✅ Model Validation: {train_accuracy-test_accuracy:+.1%} generalization gap")
    print(f"✅ Results Visualization: Professional analysis report")
    
    print(f"\n📁 Generated Files:")
    print(f"  📊 realistic_demo_report_en.png - Comprehensive analysis visualization")
    print(f"  📄 realistic_demo_summary_en.json - Detailed technical report")
    
    print(f"\n🏥 CHOP Internship Relevance:")
    for skill in final_report['chop_relevance']['skills_demonstrated']:
        print(f"  • {skill}")
    
    print(f"\n💡 Key Improvements over Previous Demo:")
    print(f"  • Realistic accuracy ({test_accuracy:.1%} vs previous 100%)")
    print(f"  • Image-based features (no data leakage)")
    print(f"  • Proper validation methodology")
    print(f"  • Comprehensive quality assurance")
    print(f"  • Professional reporting standards")
    
    print(f"\n🎉 Realistic demo completed successfully!")
    print(f"   This demonstrates genuine medical image analysis skills")
    print(f"   suitable for Children's Hospital of Philadelphia research.")
    
    return True

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("Please check data paths and dependencies")
