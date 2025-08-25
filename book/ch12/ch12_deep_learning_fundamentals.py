#!/usr/bin/env python3
"""
Chapter 12: Deep Learning Fundamentals
Data Voyage: Building Neural Networks and Deep Learning Models

This script covers essential deep learning concepts and neural network implementation
using real datasets from open sources.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml, load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import warnings
import requests
import pandas as pd
import os

warnings.filterwarnings("ignore")


def main():
    print("=" * 80)
    print("CHAPTER 12: DEEP LEARNING FUNDAMENTALS")
    print("=" * 80)
    print()

    # Section 12.1: Deep Learning Overview
    print("12.1 DEEP LEARNING OVERVIEW")
    print("-" * 45)
    demonstrate_deep_learning_overview()

    # Section 12.2: Neural Network Fundamentals
    print("\n12.2 NEURAL NETWORK FUNDAMENTALS")
    print("-" * 45)
    demonstrate_neural_network_fundamentals()

    # Section 12.3: Building Neural Networks
    print("\n12.3 BUILDING NEURAL NETWORKS")
    print("-" * 40)
    demonstrate_neural_network_building()

    # Section 12.4: Training and Optimization
    print("\n12.4 TRAINING AND OPTIMIZATION")
    print("-" * 40)
    demonstrate_training_optimization()

    print("\n" + "=" * 80)
    print("CHAPTER SUMMARY")
    print("=" * 80)
    print("✅ Deep learning overview and applications")
    print("✅ Neural network fundamentals and architecture")
    print("✅ Neural network implementation and training")
    print("✅ Optimization techniques and best practices")
    print()
    print("Next: Chapter 13 - Natural Language Processing")
    print("=" * 80)


def demonstrate_deep_learning_overview():
    """Demonstrate deep learning overview and concepts."""
    print("Deep Learning Overview:")
    print("-" * 40)

    print("Deep learning is a subset of machine learning that uses")
    print("artificial neural networks with multiple layers to learn")
    print("complex patterns and representations from data.")
    print()

    # 1. What is Deep Learning?
    print("1. WHAT IS DEEP LEARNING?")
    print("-" * 30)

    deep_learning_concepts = {
        "Definition": "Machine learning using neural networks with multiple hidden layers",
        "Key Characteristic": "Automatic feature learning from raw data",
        "Advantage": "Can learn hierarchical representations automatically",
        "Applications": "Computer vision, NLP, speech recognition, game playing",
    }

    for concept, description in deep_learning_concepts.items():
        print(f"  {concept}: {description}")
    print()

    # 2. Deep Learning vs Traditional ML
    print("2. DEEP LEARNING VS TRADITIONAL ML:")
    print("-" * 35)

    comparison = {
        "Feature Engineering": {
            "Traditional ML": "Manual feature engineering required",
            "Deep Learning": "Automatic feature learning",
        },
        "Data Requirements": {
            "Traditional ML": "Works with smaller datasets",
            "Deep Learning": "Requires large amounts of data",
        },
        "Interpretability": {
            "Traditional ML": "More interpretable models",
            "Deep Learning": "Black box models, harder to interpret",
        },
        "Performance": {
            "Traditional ML": "Good for structured data",
            "Deep Learning": "Excellent for unstructured data (images, text, audio)",
        },
    }

    for aspect, comparison_data in comparison.items():
        print(f"  {aspect}:")
        for approach, description in comparison_data.items():
            print(f"    {approach}: {description}")
        print()

    # 3. Real-World Applications
    print("3. REAL-WORLD APPLICATIONS:")
    print("-" * 30)

    applications = {
        "Computer Vision": [
            "Image classification (MNIST, CIFAR, ImageNet)",
            "Object detection (YOLO, R-CNN)",
            "Medical imaging (X-ray, MRI analysis)",
            "Autonomous vehicles (road scene understanding)",
        ],
        "Natural Language Processing": [
            "Language translation (Google Translate)",
            "Sentiment analysis (social media monitoring)",
            "Chatbots and virtual assistants",
            "Text summarization and generation",
        ],
        "Speech Recognition": [
            "Voice assistants (Siri, Alexa)",
            "Transcription services",
            "Speaker identification",
            "Emotion detection from voice",
        ],
        "Recommendation Systems": [
            "Netflix movie recommendations",
            "Amazon product suggestions",
            "Spotify music recommendations",
            "Social media content curation",
        ],
    }

    for domain, examples in applications.items():
        print(f"  {domain}:")
        for example in examples:
            print(f"    • {example}")
        print()


def demonstrate_neural_network_fundamentals():
    """Demonstrate neural network fundamentals with real data."""
    print("Neural Network Fundamentals:")
    print("-" * 40)

    print("Neural networks are computational models inspired by")
    print("biological neural networks in the human brain.")
    print()

    # 1. Neural Network Architecture
    print("1. NEURAL NETWORK ARCHITECTURE:")
    print("-" * 35)

    architecture_components = {
        "Input Layer": "Receives input data (features)",
        "Hidden Layers": "Process information through weighted connections",
        "Output Layer": "Produces final predictions or classifications",
        "Neurons": "Computational units that process inputs",
        "Weights": "Parameters that determine connection strength",
        "Bias": "Additional parameter for model flexibility",
        "Activation Functions": "Non-linear functions that introduce complexity",
    }

    for component, description in architecture_components.items():
        print(f"  {component}: {description}")
    print()

    # 2. Loading Real Datasets
    print("2. LOADING REAL DATASETS FOR NEURAL NETWORKS:")
    print("-" * 50)

    print("Loading MNIST dataset for image classification...")
    try:
        # Load MNIST dataset (handwritten digits)
        mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
        X_mnist, y_mnist = mnist.data, mnist.target.astype(int)

        # Sample a subset for demonstration
        sample_size = 5000
        indices = np.random.choice(len(X_mnist), sample_size, replace=False)
        X_mnist_sample = X_mnist[indices]
        y_mnist_sample = y_mnist[indices]

        print(f"  ✅ MNIST dataset loaded: {len(X_mnist_sample):,} samples")
        print(f"  📊 Image dimensions: {X_mnist_sample.shape[1]} pixels (28x28)")
        print(f"  🎯 Classes: {len(np.unique(y_mnist_sample))} digit classes (0-9)")
        print(f"  📈 Target distribution: {np.bincount(y_mnist_sample)}")

    except Exception as e:
        print(f"  ⚠️  Could not load MNIST: {e}")
        print("  📝 Using synthetic data as fallback...")
        X_mnist_sample = np.random.rand(5000, 784)
        y_mnist_sample = np.random.randint(0, 10, 5000)

    print("\nLoading California Housing dataset for regression...")
    try:
        # Load California Housing dataset
        california = fetch_california_housing()
        X_california, y_california = california.data, california.target

        print(f"  ✅ California Housing dataset loaded: {len(X_california):,} samples")
        print(f"  📊 Features: {X_california.shape[1]} housing characteristics")
        print(
            f"  🎯 Target: Median house values (${y_california.min():.0f} - ${y_california.max():.0f})"
        )
        print(
            f"  📈 Target statistics: Mean=${y_california.mean():.0f}, Std=${y_california.std():.0f}"
        )

    except Exception as e:
        print(f"  ⚠️  Could not load California Housing: {e}")
        print("  📝 Using synthetic data as fallback...")
        X_california = np.random.rand(20640, 8)
        y_california = np.random.normal(2, 1, 20640)

    print("\nLoading Diabetes dataset for medical prediction...")
    try:
        # Load Diabetes dataset
        diabetes = load_diabetes()
        X_diabetes, y_diabetes = diabetes.data, diabetes.target

        print(f"  ✅ Diabetes dataset loaded: {len(X_diabetes):,} samples")
        print(f"  📊 Features: {X_diabetes.shape[1]} medical measurements")
        print(f"  🎯 Target: Disease progression scores")
        print(
            f"  📈 Target statistics: Mean={y_diabetes.mean():.2f}, Std={y_diabetes.std():.2f}"
        )

    except Exception as e:
        print(f"  ⚠️  Could not load Diabetes: {e}")
        print("  📝 Using synthetic data as fallback...")
        X_diabetes = np.random.rand(442, 10)
        y_diabetes = np.random.normal(0, 50, 442)

    # 3. Data Preprocessing for Neural Networks
    print("\n3. DATA PREPROCESSING FOR NEURAL NETWORKS:")
    print("-" * 45)

    print("Preprocessing MNIST data...")
    # Normalize pixel values to [0, 1]
    X_mnist_normalized = X_mnist_sample / 255.0

    # Split data
    X_mnist_train, X_mnist_test, y_mnist_train, y_mnist_test = train_test_split(
        X_mnist_normalized,
        y_mnist_sample,
        test_size=0.2,
        random_state=42,
        stratify=y_mnist_sample,
    )

    print(f"  ✅ MNIST data normalized and split:")
    print(f"    Training: {len(X_mnist_train):,} samples")
    print(f"    Testing: {len(X_mnist_test):,} samples")

    print("\nPreprocessing California Housing data...")
    # Scale features
    scaler_california = StandardScaler()
    X_california_scaled = scaler_california.fit_transform(X_california)

    # Split data
    X_california_train, X_california_test, y_california_train, y_california_test = (
        train_test_split(
            X_california_scaled, y_california, test_size=0.2, random_state=42
        )
    )

    print(f"  ✅ California Housing data scaled and split:")
    print(f"    Training: {len(X_california_train):,} samples")
    print(f"    Testing: {len(X_california_test):,} samples")

    print("\nPreprocessing Diabetes data...")
    # Scale features
    scaler_diabetes = StandardScaler()
    X_diabetes_scaled = scaler_diabetes.fit_transform(X_diabetes)

    # Split data
    X_diabetes_train, X_diabetes_test, y_diabetes_train, y_diabetes_test = (
        train_test_split(X_diabetes_scaled, y_diabetes, test_size=0.2, random_state=42)
    )

    print(f"  ✅ Diabetes data scaled and split:")
    print(f"    Training: {len(X_diabetes_train):,} samples")
    print(f"    Testing: {len(X_diabetes_test):,} samples")

    # Store datasets for later use
    global datasets
    datasets = {
        "mnist": {
            "X_train": X_mnist_train,
            "X_test": X_mnist_test,
            "y_train": y_mnist_train,
            "y_test": y_mnist_test,
            "name": "MNIST Handwritten Digits",
        },
        "california": {
            "X_train": X_california_train,
            "X_test": X_california_test,
            "y_train": y_california_train,
            "y_test": y_california_test,
            "name": "California Housing Prices",
        },
        "diabetes": {
            "X_train": X_diabetes_train,
            "X_test": X_diabetes_test,
            "y_train": y_diabetes_train,
            "y_test": y_diabetes_test,
            "name": "Diabetes Disease Progression",
        },
    }

    # 4. Neural Network Components
    print("\n4. NEURAL NETWORK COMPONENTS:")
    print("-" * 35)

    components = {
        "Neurons (Nodes)": "Basic computational units that receive inputs and produce outputs",
        "Weights": "Parameters that determine the strength of connections between neurons",
        "Bias": "Additional parameter that allows the model to fit the data better",
        "Activation Functions": "Non-linear functions that introduce complexity to the model",
        "Layers": "Groups of neurons that process information at different levels",
        "Loss Function": "Function that measures how well the model is performing",
        "Optimizer": "Algorithm that updates the weights to minimize the loss function",
    }

    for component, description in components.items():
        print(f"  {component}: {description}")
    print()

    # 5. Activation Functions
    print("5. ACTIVATION FUNCTIONS:")
    print("-" * 30)

    activation_functions = {
        "ReLU (Rectified Linear Unit)": "f(x) = max(0, x) - Most popular, helps with vanishing gradients",
        "Sigmoid": "f(x) = 1 / (1 + e^(-x)) - Outputs values between 0 and 1",
        "Tanh (Hyperbolic Tangent)": "f(x) = (e^x - e^(-x)) / (e^x + e^(-x)) - Outputs values between -1 and 1",
        "Softmax": "f(x_i) = e^(x_i) / Σ(e^(x_j)) - Used in output layer for classification",
        "Leaky ReLU": "f(x) = max(0.01x, x) - Variant of ReLU that allows small negative values",
    }

    for function, description in activation_functions.items():
        print(f"  {function}:")
        print(f"    {description}")
    print()


def demonstrate_neural_network_building():
    """Demonstrate building neural networks with real data."""
    print("Building Neural Networks:")
    print("-" * 40)

    print("Building neural networks involves designing the architecture")
    print("and implementing the forward and backward propagation.")
    print()

    # 1. Simple Neural Network Implementation
    print("1. SIMPLE NEURAL NETWORK IMPLEMENTATION:")
    print("-" * 45)

    class SimpleNeuralNetwork:
        def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.output_size = output_size
            self.learning_rate = learning_rate

            # Initialize weights and biases
            self.W1 = np.random.randn(input_size, hidden_size) * 0.01
            self.b1 = np.zeros((1, hidden_size))
            self.W2 = np.random.randn(hidden_size, output_size) * 0.01
            self.b2 = np.zeros((1, output_size))

        def sigmoid(self, x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_derivative(self, x):
            return x * (1 - x)

        def forward(self, X):
            # Forward propagation
            self.z1 = np.dot(X, self.W1) + self.b1
            self.a1 = self.sigmoid(self.z1)
            self.z2 = np.dot(self.a1, self.W2) + self.b2
            self.a2 = self.sigmoid(self.z2)
            return self.a2

        def backward(self, X, y, output):
            # Backward propagation
            self.error = y - output
            self.delta2 = self.error * self.sigmoid_derivative(output)
            self.delta1 = np.dot(self.delta2, self.W2.T) * self.sigmoid_derivative(
                self.a1
            )

            # Update weights and biases
            self.W2 += self.learning_rate * np.dot(self.a1.T, self.delta2)
            self.b2 += self.learning_rate * np.sum(self.delta2, axis=0, keepdims=True)
            self.W1 += self.learning_rate * np.dot(X.T, self.delta1)
            self.b1 += self.learning_rate * np.sum(self.delta1, axis=0, keepdims=True)

        def train(self, X, y, epochs):
            losses = []
            for epoch in range(epochs):
                # Forward pass
                output = self.forward(X)

                # Backward pass
                self.backward(X, y, output)

                # Calculate loss
                loss = np.mean(np.square(y - output))
                losses.append(loss)

                if epoch % 100 == 0:
                    print(f"    Epoch {epoch}: Loss = {loss:.6f}")

            return losses

    # 2. Training on Real Data
    print("2. TRAINING ON REAL DATA:")
    print("-" * 30)

    # Use California Housing dataset for regression
    california_data = datasets["california"]
    X_train, X_test, y_train, y_test = (
        california_data["X_train"],
        california_data["X_test"],
        california_data["y_train"],
        california_data["y_test"],
    )

    print(f"Training neural network on {california_data['name']} dataset...")
    print(f"  Input features: {X_train.shape[1]}")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Target: Housing prices (regression)")

    # Reshape target for neural network
    y_train_reshaped = y_train.reshape(-1, 1)
    y_test_reshaped = y_test.reshape(-1, 1)

    # Create and train neural network
    nn = SimpleNeuralNetwork(
        input_size=X_train.shape[1], hidden_size=10, output_size=1, learning_rate=0.01
    )

    print("\nTraining neural network...")
    losses = nn.train(X_train, y_train_reshaped, epochs=500)

    # 3. Model Evaluation
    print("\n3. MODEL EVALUATION:")
    print("-" * 25)

    # Make predictions
    train_predictions = nn.forward(X_train)
    test_predictions = nn.forward(X_test)

    # Calculate metrics
    train_mse = mean_squared_error(y_train_reshaped, train_predictions)
    test_mse = mean_squared_error(y_test_reshaped, test_predictions)

    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)

    print(f"  Training MSE: {train_mse:.2f}")
    print(f"  Testing MSE: {test_mse:.2f}")
    print(f"  Training RMSE: ${train_rmse:.2f}")
    print(f"  Testing RMSE: ${test_rmse:.2f}")

    # Calculate R-squared
    train_r2 = 1 - (
        np.sum(np.square(y_train_reshaped - train_predictions))
        / np.sum(np.square(y_train_reshaped - np.mean(y_train_reshaped)))
    )
    test_r2 = 1 - (
        np.sum(np.square(y_test_reshaped - test_predictions))
        / np.sum(np.square(y_test_reshaped - np.mean(y_test_reshaped)))
    )

    print(f"  Training R²: {train_r2:.4f}")
    print(f"  Testing R²: {test_r2:.4f}")

    # 4. Visualization of Results
    print("\n4. VISUALIZATION OF RESULTS:")
    print("-" * 35)

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        f'Neural Network Results on {california_data["name"]} Dataset',
        fontsize=16,
        fontweight="bold",
    )

    # Plot 1: Training Loss
    axes[0, 0].plot(losses)
    axes[0, 0].set_title("Training Loss Over Time")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Mean Squared Error")
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Actual vs Predicted (Training)
    axes[0, 1].scatter(y_train_reshaped, train_predictions, alpha=0.6)
    axes[0, 1].plot(
        [y_train_reshaped.min(), y_train_reshaped.max()],
        [y_train_reshaped.min(), y_train_reshaped.max()],
        "r--",
        lw=2,
    )
    axes[0, 1].set_title("Training: Actual vs Predicted")
    axes[0, 1].set_xlabel("Actual Values")
    axes[0, 1].set_ylabel("Predicted Values")
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Actual vs Predicted (Testing)
    axes[1, 0].scatter(y_test_reshaped, test_predictions, alpha=0.6, color="orange")
    axes[1, 0].plot(
        [y_test_reshaped.min(), y_test_reshaped.max()],
        [y_test_reshaped.min(), y_test_reshaped.max()],
        "r--",
        lw=2,
    )
    axes[1, 0].set_title("Testing: Actual vs Predicted")
    axes[1, 0].set_xlabel("Actual Values")
    axes[1, 0].set_ylabel("Predicted Values")
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Residuals
    residuals = y_test_reshaped.flatten() - test_predictions.flatten()
    axes[1, 1].scatter(test_predictions, residuals, alpha=0.6, color="green")
    axes[1, 1].axhline(y=0, color="r", linestyle="--")
    axes[1, 1].set_title("Residuals vs Predicted Values")
    axes[1, 1].set_xlabel("Predicted Values")
    axes[1, 1].set_ylabel("Residuals")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("neural_network_results.png", dpi=300, bbox_inches="tight")
    print("  ✅ Neural network visualization saved as 'neural_network_results.png'")

    # 5. Feature Importance Analysis
    print("\n5. FEATURE IMPORTANCE ANALYSIS:")
    print("-" * 35)

    # Calculate feature importance based on weight magnitudes
    feature_importance = np.abs(nn.W1).mean(axis=1)

    # Get feature names for California Housing
    california_feature_names = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ]

    # Sort features by importance
    feature_importance_sorted = sorted(
        zip(california_feature_names, feature_importance),
        key=lambda x: x[1],
        reverse=True,
    )

    print("  Top 5 most important features:")
    for i, (feature, importance) in enumerate(feature_importance_sorted[:5]):
        print(f"    {i+1}. {feature}: {importance:.4f}")

    print("\n  Bottom 5 least important features:")
    for i, (feature, importance) in enumerate(feature_importance_sorted[-5:]):
        print(f"    {i+1}. {feature}: {importance:.4f}")


def demonstrate_training_optimization():
    """Demonstrate training and optimization techniques."""
    print("Training and Optimization:")
    print("-" * 40)

    print("Training neural networks involves optimizing the weights")
    print("to minimize the loss function using various techniques.")
    print()

    # 1. Optimization Techniques
    print("1. OPTIMIZATION TECHNIQUES:")
    print("-" * 35)

    optimization_techniques = {
        "Gradient Descent": "Updates weights in the direction of steepest descent",
        "Stochastic Gradient Descent (SGD)": "Uses random samples for gradient estimation",
        "Adam": "Adaptive learning rate optimization with momentum",
        "RMSprop": "Adaptive learning rate method for gradient-based optimization",
        "Momentum": "Accelerates training by adding momentum to gradient updates",
    }

    for technique, description in optimization_techniques.items():
        print(f"  {technique}: {description}")
    print()

    # 2. Hyperparameter Tuning
    print("2. HYPERPARAMETER TUNING:")
    print("-" * 30)

    hyperparameters = {
        "Learning Rate": "Controls how much to update weights in each iteration",
        "Batch Size": "Number of samples processed before updating weights",
        "Number of Epochs": "Complete passes through the training dataset",
        "Hidden Layer Sizes": "Number of neurons in each hidden layer",
        "Activation Functions": "Non-linear functions applied to neuron outputs",
        "Regularization": "Techniques to prevent overfitting (L1, L2, Dropout)",
    }

    for param, description in hyperparameters.items():
        print(f"  {param}: {description}")
    print()

    # 3. Overfitting Prevention
    print("3. OVERFITTING PREVENTION:")
    print("-" * 30)

    prevention_techniques = {
        "Early Stopping": "Stop training when validation loss starts increasing",
        "Regularization": "Add penalty terms to loss function (L1, L2)",
        "Dropout": "Randomly deactivate neurons during training",
        "Data Augmentation": "Increase training data with variations",
        "Cross-Validation": "Use multiple train-test splits for validation",
    }

    for technique, description in prevention_techniques.items():
        print(f"  {technique}: {description}")
    print()

    # 4. Model Architecture Design
    print("4. MODEL ARCHITECTURE DESIGN:")
    print("-" * 35)

    architecture_principles = {
        "Start Simple": "Begin with a simple architecture and gradually increase complexity",
        "Layer Sizes": "Gradually decrease layer sizes toward the output",
        "Activation Functions": "Use ReLU for hidden layers, appropriate functions for output",
        "Batch Normalization": "Normalize inputs to each layer for stable training",
        "Skip Connections": "Add direct connections between distant layers (ResNet style)",
    }

    for principle, description in architecture_principles.items():
        print(f"  {principle}: {description}")
    print()

    # 5. Training Best Practices
    print("5. TRAINING BEST PRACTICES:")
    print("-" * 35)

    best_practices = {
        "Data Preprocessing": "Normalize/standardize inputs, handle missing values",
        "Learning Rate Schedule": "Start with larger learning rate, gradually decrease",
        "Monitoring": "Track training and validation metrics",
        "Checkpointing": "Save model weights during training",
        "Ensemble Methods": "Combine multiple models for better performance",
    }

    for practice, description in best_practices.items():
        print(f"  {practice}: {description}")
    print()

    # 6. Performance Metrics
    print("6. PERFORMANCE METRICS:")
    print("-" * 30)

    metrics = {
        "Classification": "Accuracy, Precision, Recall, F1-Score, ROC-AUC",
        "Regression": "Mean Squared Error (MSE), Root Mean Squared Error (RMSE), R²",
        "Training": "Training Loss, Training Accuracy",
        "Validation": "Validation Loss, Validation Accuracy",
        "Generalization": "Test Loss, Test Accuracy, Cross-Validation Scores",
    }

    for metric_type, metric_list in metrics.items():
        print(f"  {metric_type}: {metric_list}")
    print()


if __name__ == "__main__":
    main()
