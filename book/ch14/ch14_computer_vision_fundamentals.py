#!/usr/bin/env python3
"""
Chapter 14: Computer Vision Fundamentals
Data Voyage: Understanding and Processing Visual Data

This script covers essential computer vision concepts and techniques.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image, ImageFilter, ImageEnhance
import cv2
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore")

def main():
    print("=" * 80)
    print("CHAPTER 14: COMPUTER VISION FUNDAMENTALS")
    print("=" * 80)
    print()
    
    # Section 14.1: Computer Vision Overview
    print("14.1 COMPUTER VISION OVERVIEW")
    print("-" * 35)
    demonstrate_cv_overview()
    
    # Section 14.2: Image Processing Fundamentals
    print("\n14.2 IMAGE PROCESSING FUNDAMENTALS")
    print("-" * 40)
    demonstrate_image_processing()
    
    # Section 14.3: Feature Extraction
    print("\n14.3 FEATURE EXTRACTION")
    print("-" * 30)
    demonstrate_feature_extraction()
    
    # Section 14.4: Computer Vision Applications
    print("\n14.4 COMPUTER VISION APPLICATIONS")
    print("-" * 40)
    demonstrate_cv_applications()
    
    print("\n" + "=" * 80)
    print("CHAPTER SUMMARY")
    print("=" * 80)
    print("✅ Computer vision overview and applications")
    print("✅ Image processing techniques")
    print("✅ Feature extraction methods")
    print("✅ Practical CV applications")
    print()
    print("Next: Chapter 15 - Time Series Analysis")
    print("=" * 80)

def demonstrate_cv_overview():
    """Demonstrate computer vision overview and concepts."""
    print("Computer Vision Overview:")
    print("-" * 40)
    
    print("Computer Vision is a field of artificial intelligence")
    print("that enables computers to interpret and understand")
    print("visual information from the world.")
    print()
    
    # 1. What is Computer Vision?
    print("1. WHAT IS COMPUTER VISION?")
    print("-" * 30)
    
    cv_concepts = {
        "Definition": "Field of AI focused on visual data understanding",
        "Goal": "Enable computers to see and interpret images/videos",
        "Input": "Images, videos, 3D data, depth maps",
        "Output": "Understanding, analysis, decision making"
    }
    
    for concept, description in cv_concepts.items():
        print(f"  {concept}: {description}")
    print()
    
    # 2. Computer Vision Tasks
    print("2. COMPUTER VISION TASKS:")
    print("-" * 30)
    
    cv_tasks = {
        "Image Classification": ["Object recognition", "Scene understanding", "Image categorization"],
        "Object Detection": ["Bounding box detection", "Instance segmentation", "Object tracking"],
        "Image Segmentation": ["Semantic segmentation", "Instance segmentation", "Panoptic segmentation"],
        "Image Generation": ["Image synthesis", "Style transfer", "Image-to-image translation"],
        "3D Vision": ["Depth estimation", "3D reconstruction", "Point cloud processing"]
    }
    
    for task, examples in cv_tasks.items():
        print(f"  {task}:")
        for example in examples:
            print(f"    • {example}")
        print()
    
    # 3. Computer Vision Pipeline
    print("3. COMPUTER VISION PIPELINE:")
    print("-" * 35)
    
    pipeline_steps = [
        "1. Image Acquisition - Capture or load images",
        "2. Preprocessing - Clean and normalize images",
        "3. Feature Extraction - Extract relevant features",
        "4. Model Processing - Apply ML/DL models",
        "5. Post-processing - Refine and interpret results",
        "6. Output Generation - Generate final results"
    ]
    
    for step in pipeline_steps:
        print(f"  {step}")
    print()
    
    # 4. Applications
    print("4. COMPUTER VISION APPLICATIONS:")
    print("-" * 35)
    
    applications = {
        "Healthcare": "Medical imaging, disease detection, surgery assistance",
        "Autonomous Vehicles": "Lane detection, obstacle avoidance, traffic sign recognition",
        "Security": "Face recognition, surveillance, biometric authentication",
        "Retail": "Product recognition, inventory management, customer analytics",
        "Manufacturing": "Quality control, defect detection, process monitoring",
        "Entertainment": "Augmented reality, gaming, content creation"
    }
    
    for domain, examples in applications.items():
        print(f"  {domain}: {examples}")
    print()

def demonstrate_image_processing():
    """Demonstrate image processing fundamentals."""
    print("Image Processing Fundamentals:")
    print("-" * 40)
    
    print("Image processing involves manipulating and analyzing")
    print("digital images to extract useful information.")
    print()
    
    # 1. Create Synthetic Images
    print("1. CREATING SYNTHETIC IMAGES:")
    print("-" * 30)
    
    # Create a simple synthetic image
    size = 100
    image = np.zeros((size, size, 3), dtype=np.uint8)
    
    # Add different colored regions
    image[0:50, 0:50] = [255, 0, 0]      # Red
    image[0:50, 50:100] = [0, 255, 0]    # Green
    image[50:100, 0:50] = [0, 0, 255]    # Blue
    image[50:100, 50:100] = [255, 255, 0] # Yellow
    
    # Add some noise
    noise = np.random.randint(0, 50, (size, size, 3), dtype=np.uint8)
    image = np.clip(image + noise, 0, 255).astype(np.uint8)
    
    print(f"✅ Created synthetic image: {image.shape}")
    print(f"   Data type: {image.dtype}")
    print(f"   Value range: {image.min()} to {image.max()}")
    print()
    
    # 2. Basic Image Operations
    print("2. BASIC IMAGE OPERATIONS:")
    print("-" * 30)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Resize image
    resized = cv2.resize(image, (50, 50))
    
    # Rotate image
    rows, cols = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
    rotated = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    
    print("✅ Applied basic operations:")
    print(f"   Grayscale: {gray.shape}")
    print(f"   Resized: {resized.shape}")
    print(f"   Rotated: {rotated.shape}")
    print()
    
    # 3. Image Filtering
    print("3. IMAGE FILTERING:")
    print("-" * 25)
    
    # Apply different filters
    blurred = cv2.GaussianBlur(image, (15, 15), 0)
    sharpened = cv2.filter2D(image, -1, np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]))
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    print("✅ Applied filters:")
    print(f"   Gaussian blur: {blurred.shape}")
    print(f"   Sharpening: {sharpened.shape}")
    print(f"   Edge detection: {edges.shape}")
    print()
    
    # 4. Image Enhancement
    print("4. IMAGE ENHANCEMENT:")
    print("-" * 30)
    
    # Convert to PIL for enhancement
    pil_image = Image.fromarray(image)
    
    # Brightness adjustment
    enhancer = ImageEnhance.Brightness(pil_image)
    brightened = enhancer.enhance(1.5)
    
    # Contrast adjustment
    enhancer = ImageEnhance.Contrast(pil_image)
    contrasted = enhancer.enhance(1.3)
    
    # Color adjustment
    enhancer = ImageEnhance.Color(pil_image)
    saturated = enhancer.enhance(1.2)
    
    print("✅ Applied enhancements:")
    print(f"   Brightness: {np.array(brightened).shape}")
    print(f"   Contrast: {np.array(contrasted).shape}")
    print(f"   Saturation: {np.array(saturated).shape}")
    print()
    
    # 5. Visualization
    print("5. VISUALIZATION:")
    print("-" * 20)
    
    plt.figure(figsize=(15, 10))
    
    # Original and grayscale
    plt.subplot(3, 3, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.axis("off")
    
    plt.subplot(3, 3, 2)
    plt.imshow(gray, cmap="gray")
    plt.title("Grayscale")
    plt.axis("off")
    
    plt.subplot(3, 3, 3)
    plt.imshow(resized)
    plt.title("Resized")
    plt.axis("off")
    
    # Filtered images
    plt.subplot(3, 3, 4)
    plt.imshow(blurred)
    plt.title("Gaussian Blur")
    plt.axis("off")
    
    plt.subplot(3, 3, 5)
    plt.imshow(sharpened)
    plt.title("Sharpened")
    plt.axis("off")
    
    plt.subplot(3, 3, 6)
    plt.imshow(edges, cmap="gray")
    plt.title("Edge Detection")
    plt.axis("off")
    
    # Enhanced images
    plt.subplot(3, 3, 7)
    plt.imshow(brightened)
    plt.title("Brightened")
    plt.axis("off")
    
    plt.subplot(3, 3, 8)
    plt.imshow(contrasted)
    plt.title("Contrasted")
    plt.axis("off")
    
    plt.subplot(3, 3, 9)
    plt.imshow(saturated)
    plt.title("Saturated")
    plt.axis("off")
    
    plt.tight_layout()
    plt.savefig("image_processing.png", dpi=300, bbox_inches="tight")
    print("✅ Image processing visualization saved as 'image_processing.png'")
    plt.close()

def demonstrate_feature_extraction():
    """Demonstrate feature extraction techniques."""
    print("Feature Extraction:")
    print("-" * 40)
    
    print("Feature extraction identifies and extracts")
    print("meaningful information from images.")
    print()
    
    # 1. Create Feature-Rich Images
    print("1. CREATING FEATURE-RICH IMAGES:")
    print("-" * 35)
    
    # Create images with different patterns
    size = 64
    
    # Image 1: Horizontal lines
    img1 = np.zeros((size, size), dtype=np.uint8)
    for i in range(0, size, 8):
        img1[i:i+4, :] = 255
    
    # Image 2: Vertical lines
    img2 = np.zeros((size, size), dtype=np.uint8)
    for i in range(0, size, 8):
        img2[:, i:i+4] = 255
    
    # Image 3: Diagonal lines
    img3 = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            if (i + j) % 16 < 8:
                img3[i, j] = 255
    
    # Image 4: Checkerboard
    img4 = np.zeros((size, size), dtype=np.uint8)
    for i in range(size):
        for j in range(size):
            if (i // 8 + j // 8) % 2 == 0:
                img4[i, j] = 255
    
    # Image 5: Random noise
    img5 = np.random.randint(0, 256, (size, size), dtype=np.uint8)
    
    images = [img1, img2, img3, img4, img5]
    image_names = ["Horizontal Lines", "Vertical Lines", "Diagonal Lines", "Checkerboard", "Random Noise"]
    
    print("✅ Created feature-rich images:")
    for i, (img, name) in enumerate(zip(images, image_names)):
        print(f"   {i+1}. {name}: {img.shape}")
    print()
    
    # 2. Statistical Features
    print("2. STATISTICAL FEATURES:")
    print("-" * 30)
    
    def extract_statistical_features(image):
        """Extract basic statistical features from image."""
        return {
            "mean": np.mean(image),
            "std": np.std(image),
            "min": np.min(image),
            "max": np.max(image),
            "median": np.median(image),
            "variance": np.var(image)
        }
    
    print("Statistical features for each image:")
    for i, (img, name) in enumerate(zip(images, image_names)):
        features = extract_statistical_features(img)
        print(f"  {i+1}. {name}:")
        for feature, value in features.items():
            print(f"     {feature}: {value:.2f}")
        print()
    
    # 3. Texture Features
    print("3. TEXTURE FEATURES:")
    print("-" * 25)
    
    def extract_texture_features(image):
        """Extract texture features using GLCM-like approach."""
        # Simple texture features
        features = {}
        
        # Local binary pattern approximation
        center = image[1:-1, 1:-1]
        neighbors = [
            image[:-2, 1:-1],   # top
            image[1:-1, 2:],    # right
            image[2:, 1:-1],    # bottom
            image[1:-1, :-2]    # left
        ]
        
        # Calculate local patterns
        patterns = np.zeros_like(center, dtype=np.uint8)
        for i, neighbor in enumerate(neighbors):
            patterns += (neighbor > center).astype(np.uint8) * (2**i)
        
        features["pattern_entropy"] = -np.sum(np.bincount(patterns.flatten()) / patterns.size * 
                                            np.log2(np.bincount(patterns.flatten()) / patterns.size + 1e-10))
        features["pattern_variance"] = np.var(patterns)
        
        return features
    
    print("Texture features for each image:")
    for i, (img, name) in enumerate(zip(images, image_names)):
        features = extract_texture_features(img)
        print(f"  {i+1}. {name}:")
        for feature, value in features.items():
            print(f"     {feature}: {value:.2f}")
        print()
    
    # 4. Edge and Corner Features
    print("4. EDGE AND CORNER FEATURES:")
    print("-" * 35)
    
    def extract_edge_features(image):
        """Extract edge-related features."""
        # Sobel edge detection
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        
        # Edge magnitude and direction
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        direction = np.arctan2(sobely, sobelx)
        
        return {
            "edge_magnitude_mean": np.mean(magnitude),
            "edge_magnitude_std": np.std(magnitude),
            "edge_direction_mean": np.mean(direction),
            "edge_direction_std": np.std(direction)
        }
    
    print("Edge features for each image:")
    for i, (img, name) in enumerate(zip(images, image_names)):
        features = extract_edge_features(img)
        print(f"  {i+1}. {name}:")
        for feature, value in features.items():
            print(f"     {feature}: {value:.2f}")
        print()
    
    # 5. Feature Matrix and Visualization
    print("5. FEATURE MATRIX AND VISUALIZATION:")
    print("-" * 40)
    
    # Combine all features
    all_features = []
    feature_names = []
    
    for img in images:
        stats = extract_statistical_features(img)
        texture = extract_texture_features(img)
        edges = extract_edge_features(img)
        
        # Combine all features
        combined = {**stats, **texture, **edges}
        all_features.append(list(combined.values()))
        
        if not feature_names:
            feature_names = list(combined.keys())
    
    # Convert to numpy array
    feature_matrix = np.array(all_features)
    
    print(f"✅ Feature matrix shape: {feature_matrix.shape}")
    print(f"   Features: {len(feature_names)}")
    print(f"   Images: {len(images)}")
    print()
    
    # Feature importance using variance
    feature_variance = np.var(feature_matrix, axis=0)
    important_features_idx = np.argsort(feature_variance)[::-1][:5]
    
    print("Top 5 most discriminative features:")
    for i, idx in enumerate(important_features_idx):
        print(f"  {i+1}. {feature_names[idx]}: {feature_variance[idx]:.2f}")
    print()
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Feature heatmap
    plt.subplot(1, 3, 1)
    sns.heatmap(feature_matrix, 
                xticklabels=[f"F{i+1}" for i in range(len(feature_names))],
                yticklabels=image_names,
                cmap="viridis", annot=True, fmt=".2f")
    plt.title("Feature Matrix")
    plt.xlabel("Features")
    plt.ylabel("Images")
    
    # Feature variance
    plt.subplot(1, 3, 2)
    plt.bar(range(len(feature_names)), feature_variance)
    plt.xlabel("Features")
    plt.ylabel("Variance")
    plt.title("Feature Variance")
    plt.xticks(range(len(feature_names)), [f"F{i+1}" for i in range(len(feature_names))], rotation=45)
    
    # PCA visualization
    plt.subplot(1, 3, 3)
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(feature_matrix)
    
    plt.scatter(features_2d[:, 0], features_2d[:, 1], s=100, alpha=0.7)
    for i, name in enumerate(image_names):
        plt.annotate(name, (features_2d[i, 0], features_2d[i, 1]), 
                    xytext=(5, 5), textcoords="offset points")
    
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
    plt.title("PCA of Features")
    
    plt.tight_layout()
    plt.savefig("feature_extraction.png", dpi=300, bbox_inches="tight")
    print("✅ Feature extraction visualization saved as 'feature_extraction.png'")
    plt.close()

def demonstrate_cv_applications():
    """Demonstrate practical computer vision applications."""
    print("Computer Vision Applications:")
    print("-" * 40)
    
    print("Let's implement some practical computer vision")
    print("applications to demonstrate real-world usage.")
    print()
    
    # 1. Image Classification
    print("1. IMAGE CLASSIFICATION:")
    print("-" * 30)
    
    # Create a simple image classification dataset
    print("Creating synthetic image classification dataset...")
    
    # Generate different types of synthetic images
    n_samples = 200
    image_size = 32
    
    X = []
    y = []
    
    # Class 0: Horizontal lines
    for _ in range(n_samples // 4):
        img = np.zeros((image_size, image_size), dtype=np.uint8)
        for i in range(0, image_size, 4):
            img[i:i+2, :] = np.random.randint(200, 256)
        X.append(img.flatten())
        y.append(0)
    
    # Class 1: Vertical lines
    for _ in range(n_samples // 4):
        img = np.zeros((image_size, image_size), dtype=np.uint8)
        for i in range(0, image_size, 4):
            img[:, i:i+2] = np.random.randint(200, 256)
        X.append(img.flatten())
        y.append(1)
    
    # Class 2: Checkerboard
    for _ in range(n_samples // 4):
        img = np.zeros((image_size, image_size), dtype=np.uint8)
        for i in range(image_size):
            for j in range(image_size):
                if (i // 4 + j // 4) % 2 == 0:
                    img[i, j] = np.random.randint(200, 256)
        X.append(img.flatten())
        y.append(2)
    
    # Class 3: Random noise
    for _ in range(n_samples // 4):
        img = np.random.randint(0, 256, (image_size, image_size), dtype=np.uint8)
        X.append(img.flatten())
        y.append(3)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"✅ Dataset created: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"   Classes: {len(np.unique(y))}")
    print(f"   Class distribution: {np.bincount(y)}")
    print()
    
    # 2. Train Classification Model
    print("2. TRAIN CLASSIFICATION MODEL:")
    print("-" * 35)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Train Random Forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test)
    
    print("Training Results:")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print()
    
    # 3. Model Evaluation
    print("3. MODEL EVALUATION:")
    print("-" * 25)
    
    class_names = ["Horizontal Lines", "Vertical Lines", "Checkerboard", "Random Noise"]
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    
    print("Confusion Matrix:")
    print("                Predicted")
    print("                ", end="")
    for name in class_names:
        print(f"{name[:8]:>8}", end="")
    print()
    
    for i, actual in enumerate(class_names):
        print(f"{actual[:8]:>8}", end="")
        for j in range(len(class_names)):
            print(f"{cm[i,j]:>8}", end="")
        print()
    print()
    
    # 4. Feature Importance
    print("4. FEATURE IMPORTANCE:")
    print("-" * 25)
    
    # Get feature importance
    feature_importance = clf.feature_importances_
    
    # Reshape to image dimensions
    importance_image = feature_importance.reshape(image_size, image_size)
    
    print("Feature importance analysis:")
    print(f"   Most important pixel: {np.argmax(feature_importance)}")
    print(f"   Average importance: {np.mean(feature_importance):.4f}")
    print(f"   Importance std: {np.std(feature_importance):.4f}")
    print()
    
    # 5. New Image Classification
    print("5. NEW IMAGE CLASSIFICATION:")
    print("-" * 35)
    
    # Create new test images
    new_images = []
    new_image_names = []
    
    # New horizontal lines
    img = np.zeros((image_size, image_size), dtype=np.uint8)
    for i in range(0, image_size, 6):
        img[i:i+3, :] = 255
    new_images.append(img.flatten())
    new_image_names.append("New Horizontal Lines")
    
    # New vertical lines
    img = np.zeros((image_size, image_size), dtype=np.uint8)
    for i in range(0, image_size, 6):
        img[:, i:i+3] = 255
    new_images.append(img.flatten())
    new_image_names.append("New Vertical Lines")
    
    # New checkerboard
    img = np.zeros((image_size, image_size), dtype=np.uint8)
    for i in range(image_size):
        for j in range(image_size):
            if (i // 6 + j // 6) % 2 == 0:
                img[i, j] = 255
    new_images.append(img.flatten())
    new_image_names.append("New Checkerboard")
    
    # Predict on new images
    new_predictions = clf.predict(new_images)
    new_probabilities = clf.predict_proba(new_images)
    
    print("Predicting on new images:")
    for i, (name, pred, prob) in enumerate(zip(new_image_names, new_predictions, new_probabilities)):
        predicted_class = class_names[pred]
        confidence = max(prob)
        print(f"  {i+1}. {name}")
        print(f"     Prediction: {predicted_class} (confidence: {confidence:.3f})")
        print()
    
    # 6. Visualization
    print("6. VISUALIZATION:")
    print("-" * 20)
    
    plt.figure(figsize=(15, 5))
    
    # Confusion Matrix
    plt.subplot(1, 3, 1)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[name[:8] for name in class_names],
                yticklabels=[name[:8] for name in class_names])
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    
    # Feature Importance Heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(importance_image, cmap="hot")
    plt.title("Feature Importance Heatmap")
    plt.colorbar(label="Importance")
    plt.axis("off")
    
    # New Images and Predictions
    plt.subplot(1, 3, 3)
    for i, (img, name, pred) in enumerate(zip(new_images, new_image_names, new_predictions)):
        img_reshaped = img.reshape(image_size, image_size)
        plt.subplot(3, 1, i+1)
        plt.imshow(img_reshaped, cmap="gray")
        plt.title(f"{name[:15]} → {class_names[pred][:8]}")
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig("cv_applications.png", dpi=300, bbox_inches="tight")
    print("✅ Computer vision applications visualization saved as 'cv_applications.png'")
    plt.close()
    
    print("Computer Vision Applications Summary:")
    print("✅ Created synthetic image dataset")
    print("✅ Trained image classification model")
    print("✅ Evaluated model performance")
    print("✅ Analyzed feature importance")
    print("✅ Applied model to new images")

if __name__ == "__main__":
    main()
