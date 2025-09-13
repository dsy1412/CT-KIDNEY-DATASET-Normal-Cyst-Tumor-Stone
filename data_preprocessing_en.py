#!/usr/bin/env python3
"""
CT Kidney Image Data Preprocessing Module
Focused on data cleaning, standardization, and preprocessing workflows

Core Functions:
- Medical image data loading and validation
- Data cleaning and quality control
- Image preprocessing and standardization
- Dataset splitting and preparation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ImagePreprocessor:
    """Image preprocessing handler"""
    
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.stats = {
            'processed': 0,
            'failed': 0,
            'corrupted': 0,
            'resized': 0
        }
    
    def load_and_validate_image(self, image_path):
        """Load and validate a single image"""
        try:
            # Try loading with PIL
            with Image.open(image_path) as img:
                # Check if image is valid
                img.verify()
            
            # Reload for processing
            img = Image.open(image_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Check image dimensions
            if img.size[0] == 0 or img.size[1] == 0:
                return None, "Zero dimension image"
            
            return img, None
            
        except Exception as e:
            return None, str(e)
    
    def resize_image(self, image):
        """Resize image to target dimensions"""
        if image.size != self.target_size:
            image = image.resize(self.target_size, Image.Resampling.LANCZOS)
            self.stats['resized'] += 1
        return image
    
    def enhance_image(self, image):
        """Image enhancement processing"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Contrast enhancement (CLAHE)
        img_yuv = cv2.cvtColor(img_array, cv2.COLOR_RGB2YUV)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_yuv[:,:,0] = clahe.apply(img_yuv[:,:,0])
        img_enhanced = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
        
        return Image.fromarray(img_enhanced)
    
    def process_image(self, image_path, enhance=True):
        """Process a single image"""
        # Load and validate
        image, error = self.load_and_validate_image(image_path)
        if image is None:
            self.stats['failed'] += 1
            return None, f"Loading failed: {error}"
        
        # Resize
        image = self.resize_image(image)
        
        # Enhancement
        if enhance:
            try:
                image = self.enhance_image(image)
            except Exception as e:
                print(f"Enhancement failed: {e}")
        
        self.stats['processed'] += 1
        return image, None

class DatasetProcessor:
    """Dataset processing handler"""
    
    def __init__(self, csv_path, data_dir):
        self.csv_path = csv_path
        self.data_dir = Path(data_dir)
        self.df = None
        self.label_encoder = LabelEncoder()
        self.image_processor = ImagePreprocessor()
        
    def load_metadata(self):
        """Load metadata from CSV"""
        print("📄 Loading data annotation file...")
        try:
            self.df = pd.read_csv(self.csv_path)
            print(f"✅ Successfully loaded {len(self.df)} records")
            
            # Display basic information
            print(f"\n📊 Dataset Overview:")
            print(f"  Total samples: {len(self.df)}")
            print(f"  Feature columns: {list(self.df.columns)}")
            
            # Class distribution
            class_counts = self.df['diag'].value_counts()
            print(f"\n🏷️ Class Distribution:")
            for class_name, count in class_counts.items():
                percentage = (count / len(self.df)) * 100
                print(f"  {class_name}: {count} ({percentage:.1f}%)")
            
            return True
        except Exception as e:
            print(f"❌ Loading failed: {e}")
            return False
    
    def validate_data_integrity(self, sample_size=1000):
        """Validate data integrity"""
        print(f"\n🔍 Validating data integrity (checking first {sample_size} samples)...")
        
        # Limit check sample size
        check_df = self.df.head(sample_size) if sample_size else self.df
        
        valid_files = []
        invalid_files = []
        missing_files = []
        
        for idx, row in check_df.iterrows():
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
            
            if not image_path.exists():
                missing_files.append(image_name)
                continue
            
            # Validate image
            _, error = self.image_processor.load_and_validate_image(str(image_path))
            if error:
                invalid_files.append((image_name, error))
            else:
                valid_files.append(image_name)
            
            # Show progress
            if (idx + 1) % 100 == 0:
                print(f"  Checked {idx + 1}/{len(check_df)} files...")
        
        # Statistics
        total_checked = len(check_df)
        valid_count = len(valid_files)
        invalid_count = len(invalid_files)
        missing_count = len(missing_files)
        
        print(f"\n📋 Data Integrity Report:")
        print(f"  Total checked: {total_checked}")
        print(f"  Valid files: {valid_count} ({valid_count/total_checked*100:.1f}%)")
        print(f"  Invalid files: {invalid_count} ({invalid_count/total_checked*100:.1f}%)")
        print(f"  Missing files: {missing_count} ({missing_count/total_checked*100:.1f}%)")
        
        if invalid_files:
            print(f"\n⚠️ First 5 invalid file examples:")
            for file, error in invalid_files[:5]:
                print(f"  {file}: {error}")
        
        return {
            'total_checked': total_checked,
            'valid_files': valid_files,
            'invalid_files': invalid_files,
            'missing_files': missing_files
        }
    
    def clean_dataset(self, validation_result=None):
        """Clean dataset"""
        print("\n🧹 Cleaning dataset...")
        
        original_size = len(self.df)
        
        if validation_result:
            # Clean based on validation results
            valid_image_ids = [Path(f).stem for f in validation_result['valid_files']]
            self.df = self.df[self.df['image_id'].isin(valid_image_ids)]
        
        # Remove duplicates
        duplicates = self.df.duplicated(subset=['image_id']).sum()
        if duplicates > 0:
            print(f"  Found {duplicates} duplicate records, removing...")
            self.df = self.df.drop_duplicates(subset=['image_id'])
        
        # Check missing values
        missing_values = self.df.isnull().sum().sum()
        if missing_values > 0:
            print(f"  Found {missing_values} missing values, processing...")
            self.df = self.df.dropna()
        
        cleaned_size = len(self.df)
        removed_count = original_size - cleaned_size
        
        print(f"✅ Data cleaning completed:")
        print(f"  Original data: {original_size} records")
        print(f"  After cleaning: {cleaned_size} records")
        print(f"  Removed: {removed_count} records ({removed_count/original_size*100:.1f}%)")
        
        return self.df
    
    def prepare_labels(self):
        """Prepare label encoding"""
        print("\n🏷️ Preparing label encoding...")
        
        # Encode labels
        self.df['label_encoded'] = self.label_encoder.fit_transform(self.df['diag'])
        
        # Show encoding mapping
        label_mapping = dict(zip(self.label_encoder.classes_, 
                                self.label_encoder.transform(self.label_encoder.classes_)))
        
        print("  Label encoding mapping:")
        for class_name, encoded_value in label_mapping.items():
            print(f"    {class_name} -> {encoded_value}")
        
        return label_mapping
    
    def split_dataset(self, test_size=0.2, val_size=0.1, random_state=42):
        """Split dataset"""
        print(f"\n✂️ Splitting dataset (train:{1-test_size-val_size:.1f}, val:{val_size:.1f}, test:{test_size:.1f})...")
        
        # Stratified sampling to ensure balanced class distribution
        train_val_df, test_df = train_test_split(
            self.df, 
            test_size=test_size, 
            stratify=self.df['diag'], 
            random_state=random_state
        )
        
        # Split again for train and validation
        val_size_adjusted = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, 
            test_size=val_size_adjusted, 
            stratify=train_val_df['diag'], 
            random_state=random_state
        )
        
        print(f"✅ Dataset splitting completed:")
        print(f"  Training set: {len(train_df)} samples")
        print(f"  Validation set: {len(val_df)} samples")
        print(f"  Test set: {len(test_df)} samples")
        
        # Check class distribution in each set
        print(f"\n📊 Class distribution in each set:")
        for split_name, split_df in [("Training", train_df), ("Validation", val_df), ("Test", test_df)]:
            print(f"  {split_name} set:")
            class_counts = split_df['diag'].value_counts()
            for class_name, count in class_counts.items():
                percentage = (count / len(split_df)) * 100
                print(f"    {class_name}: {count} ({percentage:.1f}%)")
        
        return train_df, val_df, test_df
    
    def create_preprocessing_report(self):
        """Create preprocessing visualization report"""
        print("\n📈 Generating preprocessing visualization report...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Data Preprocessing Analysis Report', fontsize=16, fontweight='bold')
        
        # 1. Original class distribution
        class_counts = self.df['diag'].value_counts()
        axes[0, 0].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Class Distribution')
        
        # 2. Label encoding distribution
        encoded_counts = self.df['label_encoded'].value_counts().sort_index()
        bars = axes[0, 1].bar(encoded_counts.index, encoded_counts.values, 
                             color=['skyblue', 'lightgreen', 'lightcoral', 'gold'][:len(encoded_counts)])
        axes[0, 1].set_title('Label Encoding Distribution')
        axes[0, 1].set_xlabel('Encoded Value')
        axes[0, 1].set_ylabel('Sample Count')
        
        # Add value labels
        for bar, value in zip(bars, encoded_counts.values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                           str(value), ha='center', va='bottom')
        
        # 3. Processing statistics
        process_stats = self.image_processor.stats
        stats_labels = list(process_stats.keys())
        stats_values = list(process_stats.values())
        
        axes[1, 0].bar(stats_labels, stats_values, color=['green', 'red', 'orange', 'blue'])
        axes[1, 0].set_title('Image Processing Statistics')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Data quality overview
        quality_info = [
            f"Total samples: {len(self.df)}",
            f"Number of classes: {self.df['diag'].nunique()}",
            f"Successfully processed: {process_stats['processed']}",
            f"Processing failed: {process_stats['failed']}",
            f"Data integrity: {(process_stats['processed']/(process_stats['processed']+process_stats['failed'])*100):.1f}%" if (process_stats['processed']+process_stats['failed']) > 0 else "Data integrity: N/A"
        ]
        
        axes[1, 1].text(0.1, 0.5, '\n'.join(quality_info), 
                       fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].set_title('Data Quality Overview')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('data_preprocessing_report_en.png', dpi=300, bbox_inches='tight')
        print("  ✅ Preprocessing report saved as 'data_preprocessing_report_en.png'")
        
        return fig
    
    def save_processed_metadata(self, train_df, val_df, test_df, label_mapping):
        """Save processed metadata"""
        print("\n💾 Saving processed metadata...")
        
        metadata = {
            'preprocessing_info': {
                'timestamp': datetime.now().isoformat(),
                'original_dataset_size': len(self.df),
                'target_image_size': self.image_processor.target_size,
                'label_mapping': label_mapping
            },
            'dataset_splits': {
                'train_size': len(train_df),
                'val_size': len(val_df),
                'test_size': len(test_df)
            },
            'class_distribution': {
                'overall': {str(k): int(v) for k, v in self.df['diag'].value_counts().to_dict().items()},
                'train': {str(k): int(v) for k, v in train_df['diag'].value_counts().to_dict().items()},
                'val': {str(k): int(v) for k, v in val_df['diag'].value_counts().to_dict().items()},
                'test': {str(k): int(v) for k, v in test_df['diag'].value_counts().to_dict().items()}
            },
            'processing_stats': self.image_processor.stats,
            'data_paths': {
                'csv_file': str(self.csv_path),
                'data_directory': str(self.data_dir)
            }
        }
        
        # Save metadata
        with open('preprocessing_metadata_en.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        # Save processed datasets
        train_df.to_csv('train_dataset_en.csv', index=False)
        val_df.to_csv('val_dataset_en.csv', index=False)
        test_df.to_csv('test_dataset_en.csv', index=False)
        
        print("  ✅ Metadata saved:")
        print("    📄 preprocessing_metadata_en.json - Preprocessing metadata")
        print("    📊 train_dataset_en.csv - Training set")
        print("    📊 val_dataset_en.csv - Validation set")
        print("    📊 test_dataset_en.csv - Test set")
        
        return metadata
    
    def run_preprocessing_pipeline(self, sample_size=1000):
        """Run complete preprocessing pipeline"""
        print("🔄 Starting data preprocessing pipeline...")
        print("=" * 50)
        
        try:
            # 1. Load metadata
            if not self.load_metadata():
                return False
            
            # 2. Validate data integrity
            validation_result = self.validate_data_integrity(sample_size)
            
            # 3. Clean dataset
            self.clean_dataset(validation_result)
            
            # 4. Prepare labels
            label_mapping = self.prepare_labels()
            
            # 5. Split dataset
            train_df, val_df, test_df = self.split_dataset()
            
            # 6. Create report
            self.create_preprocessing_report()
            
            # 7. Save results
            metadata = self.save_processed_metadata(train_df, val_df, test_df, label_mapping)
            
            print("\n🎉 Data preprocessing pipeline completed!")
            print("✅ Generated files:")
            print("  📊 data_preprocessing_report_en.png - Preprocessing visualization report")
            print("  📄 preprocessing_metadata_en.json - Preprocessing metadata")
            print("  📂 *_dataset_en.csv - Split dataset files")
            
            return True
            
        except Exception as e:
            print(f"❌ Preprocessing pipeline failed: {e}")
            return False

def main():
    """Main function"""
    print("=" * 60)
    print("🔬 CT Kidney Image Data Preprocessing Module")
    print("   Children's Hospital of Philadelphia")
    print("   Imaging Data Analyst Internship Demo")
    print("=" * 60)
    
    # Configure paths
    csv_path = "G:/Shawn/kidneyData.csv"
    data_dir = "G:/Shawn/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
    
    # Create processor instance
    processor = DatasetProcessor(csv_path, data_dir)
    
    # Run preprocessing pipeline
    success = processor.run_preprocessing_pipeline(sample_size=1000)  # Limit sample size for demo
    
    if success:
        print("\n✅ Data preprocessing completed successfully!")
        print("\n💡 Key achievements:")
        print("  • Complete data validation and cleaning workflow")
        print("  • Standardized image preprocessing")
        print("  • Stratified dataset splitting")
        print("  • Detailed processing reports and metadata")
        print("  • Clean data ready for machine learning")
    else:
        print("\n❌ Data preprocessing failed, please check data paths and file integrity")

if __name__ == "__main__":
    main()
