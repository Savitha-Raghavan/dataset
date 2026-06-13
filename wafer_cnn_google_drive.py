"""
Wafer Defect Detection using CNN - Google Drive Integration
============================================================

This notebook integrates with Google Drive to:
1. Mount Google Drive
2. Load wafer_images folder from Drive
3. Preprocess images
4. Train CNN model
5. Evaluate and visualize results
6. Save model back to Drive

Dataset Structure (in Google Drive):
    wafer_images/
    ├── good/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── defective/
        ├── defect1.jpg
        ├── defect2.jpg
        └── ...

Author: Savitha Raghavan
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import os
import cv2
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: MOUNT GOOGLE DRIVE
# ============================================================================

from google.colab import drive
drive.mount('/content/drive', force_remount=True)
print("✓ Google Drive mounted successfully!")

# ============================================================================
# STEP 2: VERIFY DATASET FOLDER
# ============================================================================

print("\n" + "=" * 80)
print("VERIFYING DATASET FOLDER")
print("=" * 80)

drive_dataset_path = '/content/drive/MyDrive/wafer_images'

if os.path.exists(drive_dataset_path):
    print(f"✓ Found wafer_images folder at: {drive_dataset_path}")
    
    # Check subdirectories
    good_path = os.path.join(drive_dataset_path, 'good')
    defective_path = os.path.join(drive_dataset_path, 'defective')
    
    if os.path.exists(good_path):
        good_count = len([f for f in os.listdir(good_path) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
        print(f"✓ Found 'good' folder with {good_count} images")
    else:
        print("❌ 'good' folder not found")
    
    if os.path.exists(defective_path):
        defective_count = len([f for f in os.listdir(defective_path) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
        print(f"✓ Found 'defective' folder with {defective_count} images")
    else:
        print("❌ 'defective' folder not found")
else:
    print(f"❌ wafer_images folder not found at {drive_dataset_path}")
    print("\nAvailable folders in MyDrive:")
    for item in os.listdir('/content/drive/MyDrive'):
        print(f"  - {item}")

# ============================================================================
# STEP 3: DEFINE CNN MODEL CLASS
# ============================================================================

class WaferDefectDetectorCNN:
    """CNN for Wafer Defect Detection"""
    
    def __init__(self, image_size=256, batch_size=32):
        self.image_size = image_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        
    def load_images_from_drive(self, dataset_path, verbose=True):
        """
        Load images from Google Drive folder
        
        Args:
            dataset_path: Path to wafer_images folder in Drive
            verbose: Print loading progress
            
        Returns:
            X: Image array (N, 256, 256, 1)
            y: Labels (0=good, 1=defective)
        """
        images = []
        labels = []
        
        class_map = {'good': 0, 'defective': 1}
        class_counts = {'good': 0, 'defective': 0}
        
        if verbose:
            print("\n" + "=" * 80)
            print("LOADING IMAGES FROM GOOGLE DRIVE")
            print("=" * 80)
        
        for class_name, label in class_map.items():
            class_path = os.path.join(dataset_path, class_name)
            
            if not os.path.exists(class_path):
                if verbose:
                    print(f"❌ {class_path} not found")
                continue
            
            if verbose:
                print(f"\n📂 Loading {class_name.upper()} images...")
            
            # Get all image files
            image_files = [f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'))]
            
            total_files = len(image_files)
            if verbose:
                print(f"   Found {total_files} images")
            
            for idx, image_file in enumerate(image_files):
                img_path = os.path.join(class_path, image_file)
                
                try:
                    # Read image
                    img = cv2.imread(img_path)
                    
                    if img is None:
                        continue
                    
                    # Convert to grayscale
                    if len(img.shape) == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    # Resize to standard size
                    img = cv2.resize(img, (self.image_size, self.image_size))
                    
                    # Normalize to [0, 1]
                    img = img.astype(np.float32) / 255.0
                    
                    images.append(img)
                    labels.append(label)
                    class_counts[class_name] += 1
                    
                    # Progress indicator
                    if verbose and (idx + 1) % max(1, total_files // 5) == 0:
                        print(f"   ✓ Loaded {idx + 1}/{total_files}")
                    
                except Exception as e:
                    if verbose:
                        print(f"   ⚠️  Error loading {image_file}: {str(e)[:50]}")
            
            if verbose:
                print(f"   ✓ Total {class_name} images: {class_counts[class_name]}")
        
        # Convert to numpy arrays
        X = np.array(images, dtype=np.float32)
        y = np.array(labels, dtype=np.int32)
        
        # Add channel dimension
        X = np.expand_dims(X, axis=-1)
        
        if verbose:
            print("\n" + "=" * 80)
            print("DATASET SUMMARY")
            print("=" * 80)
            print(f"Total images:        {len(images)}")
            print(f"Good wafers:         {class_counts['good']} ({100*class_counts['good']/max(1,len(images)):.1f}%)")
            print(f"Defective wafers:    {class_counts['defective']} ({100*class_counts['defective']/max(1,len(images)):.1f}%)")
            print(f"Image shape:         {X.shape}")
            print(f"Data type:           {X.dtype}")
            print("=" * 80 + "\n")
        
        return X, y
    
    def build_model(self):
        """Build CNN Architecture"""
        model = models.Sequential([
            # Input
            layers.Input(shape=(self.image_size, self.image_size, 1)),
            
            # ===== BLOCK 1 =====
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # ===== BLOCK 2 =====
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # ===== BLOCK 3 =====
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # ===== BLOCK 4 =====
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # ===== BLOCK 5 =====
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            
            # ===== DENSE LAYERS =====
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            
            # ===== OUTPUT =====
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.Precision(), 
                    keras.metrics.Recall(), keras.metrics.AUC(name='auc')]
        )
        
        self.model = model
        return model
    
    def train_model(self, X_train, y_train, X_val, y_val, epochs=100):
        """Train the model"""
        
        # Calculate class weights
        class_weight = {
            0: len(y_train) / (2 * max(1, (y_train == 0).sum())),
            1: len(y_train) / (2 * max(1, (y_train == 1).sum()))
        }
        
        print("\n" + "=" * 80)
        print("TRAINING CONFIGURATION")
        print("=" * 80)
        print(f"Training samples:     {len(X_train)}")
        print(f"Validation samples:   {len(X_val)}")
        print(f"Batch size:           {self.batch_size}")
        print(f"Epochs:               {epochs}")
        print(f"Class weights:        {class_weight}")
        print("=" * 80 + "\n")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=25, restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss', factor=0.5, patience=12, min_lr=1e-7,
                verbose=1
            ),
            keras.callbacks.ModelCheckpoint(
                '/content/drive/MyDrive/wafer_cnn_best.h5',
                monitor='val_auc', save_best_only=True, verbose=1
            )
        ]
        
        # Train
        self.history = self.model.fit(
            X_train, y_train,
            batch_size=self.batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate model"""
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        metrics = {
            'loss': results[0],
            'accuracy': results[1],
            'precision': results[2],
            'recall': results[3],
            'auc': results[4]
        }
        
        print("\n" + "=" * 80)
        print("TEST SET METRICS")
        print("=" * 80)
        print(f"Loss:        {metrics['loss']:.4f}")
        print(f"Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Precision:   {metrics['precision']:.4f}")
        print(f"Recall:      {metrics['recall']:.4f}")
        print(f"AUC:         {metrics['auc']:.4f}")
        print("=" * 80 + "\n")
        
        return metrics
    
    def plot_results(self, X_test, y_test):
        """Plot training history and predictions"""
        
        # 1. Training History
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        axes[0, 0].plot(self.history.history['accuracy'], label='Train', linewidth=2)
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation', linewidth=2)
        axes[0, 0].set_title('Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)
        
        axes[0, 1].plot(self.history.history['loss'], label='Train', linewidth=2)
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation', linewidth=2)
        axes[0, 1].set_title('Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)
        
        axes[1, 0].plot(self.history.history['auc'], label='Train', linewidth=2)
        axes[1, 0].plot(self.history.history['val_auc'], label='Validation', linewidth=2)
        axes[1, 0].set_title('AUC', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('AUC')
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)
        
        axes[1, 1].plot(self.history.history['precision'], label='Train Precision', linewidth=2)
        axes[1, 1].plot(self.history.history['val_precision'], label='Val Precision', linewidth=2)
        axes[1, 1].plot(self.history.history['recall'], label='Train Recall', linewidth=2)
        axes[1, 1].plot(self.history.history['val_recall'], label='Val Recall', linewidth=2)
        axes[1, 1].set_title('Precision & Recall', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/content/drive/MyDrive/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Training history saved to Drive")
        
        # 2. Confusion Matrix
        y_pred = (self.model.predict(X_test) > 0.5).astype(int).flatten()
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Good', 'Defective'],
                   yticklabels=['Good', 'Defective'],
                   cbar_kws={'label': 'Count'}, annot_kws={'fontsize': 14})
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('/content/drive/MyDrive/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Confusion matrix saved to Drive")
        
        # 3. Sample Predictions
        y_pred_proba = self.model.predict(X_test)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i in range(8):
            img = X_test[i].squeeze()
            prob = y_pred_proba[i][0]
            pred_label = "Defective" if prob > 0.5 else "Good"
            true_label = "Defective" if y_test[i] == 1 else "Good"
            
            axes[i].imshow(img, cmap='gray')
            color = 'green' if pred_label == true_label else 'red'
            axes[i].set_title(f"True: {true_label}\nPred: {pred_label}\nConf: {prob:.3f}",
                            fontsize=10, color=color, fontweight='bold')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('/content/drive/MyDrive/sample_predictions.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("✓ Sample predictions saved to Drive")
    
    def save_model(self):
        """Save model to Drive"""
        save_path = '/content/drive/MyDrive/wafer_cnn.h5'
        self.model.save(save_path)
        print(f"✓ Model saved to: {save_path}")


# ============================================================================
# STEP 4: LOAD DATA
# ============================================================================

print("\nStep 1: Loading Data from Google Drive")
print("-" * 80)

detector = WaferDefectDetectorCNN(image_size=256, batch_size=32)
X, y = detector.load_images_from_drive(drive_dataset_path)

# ============================================================================
# STEP 5: SPLIT DATA
# ============================================================================

print("\nStep 2: Splitting Dataset")
print("-" * 80)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Training set:   {X_train.shape} - Good: {(y_train == 0).sum()}, Defective: {(y_train == 1).sum()}")
print(f"Validation set: {X_val.shape} - Good: {(y_val == 0).sum()}, Defective: {(y_val == 1).sum()}")
print(f"Test set:       {X_test.shape} - Good: {(y_test == 0).sum()}, Defective: {(y_test == 1).sum()}")

# ============================================================================
# STEP 6: BUILD MODEL
# ============================================================================

print("\nStep 3: Building CNN Model")
print("-" * 80)

detector.build_model()
detector.model.summary()

# ============================================================================
# STEP 7: TRAIN MODEL
# ============================================================================

print("\nStep 4: Training Model")
print("-" * 80)

detector.train_model(X_train, y_train, X_val, y_val, epochs=100)

# ============================================================================
# STEP 8: EVALUATE MODEL
# ============================================================================

print("\nStep 5: Evaluating Model")
print("-" * 80)

metrics = detector.evaluate_model(X_test, y_test)

# ============================================================================
# STEP 9: VISUALIZE RESULTS
# ============================================================================

print("\nStep 6: Generating Visualizations")
print("-" * 80)

detector.plot_results(X_test, y_test)

# ============================================================================
# STEP 10: SAVE MODEL
# ============================================================================

print("\nStep 7: Saving Model")
print("-" * 80)

detector.save_model()

# ============================================================================
# STEP 11: CLASSIFICATION REPORT
# ============================================================================

print("\nStep 8: Classification Report")
print("-" * 80)

y_pred = (detector.model.predict(X_test) > 0.5).astype(int).flatten()
print(classification_report(y_test, y_pred, target_names=['Good', 'Defective']))

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("✓ TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"\n📊 Final Results:")
print(f"   Accuracy:  {metrics['accuracy']:.4f}")
print(f"   Precision: {metrics['precision']:.4f}")
print(f"   Recall:    {metrics['recall']:.4f}")
print(f"   AUC:       {metrics['auc']:.4f}")
print(f"\n💾 Files saved to Google Drive:")
print(f"   ✓ wafer_cnn.h5")
print(f"   ✓ wafer_cnn_best.h5")
print(f"   ✓ training_history.png")
print(f"   ✓ confusion_matrix.png")
print(f"   ✓ sample_predictions.png")
print("=" * 80 + "\n")
