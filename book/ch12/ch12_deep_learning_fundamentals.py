#!/usr/bin/env python3
"""
Chapter 12: Deep Learning Fundamentals
Data Voyage: Building Neural Networks and Deep Learning Models

This script covers essential deep learning concepts and neural network implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import warnings

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
        "Computational Cost": {
            "Traditional ML": "Lower computational requirements",
            "Deep Learning": "High computational requirements (GPU/TPU)",
        },
    }

    for aspect, details in comparison.items():
        print(f"  {aspect}:")
        print(f"    Traditional ML: {details['Traditional ML']}")
        print(f"    Deep Learning: {details['Deep Learning']}")
        print()

    # 3. Deep Learning Applications
    print("3. DEEP LEARNING APPLICATIONS:")
    print("-" * 35)

    applications = {
        "Computer Vision": [
            "Image classification",
            "Object detection",
            "Image segmentation",
            "Face recognition",
        ],
        "Natural Language Processing": [
            "Text classification",
            "Machine translation",
            "Question answering",
            "Text generation",
        ],
        "Speech Recognition": [
            "Voice assistants",
            "Transcription",
            "Speaker identification",
            "Emotion detection",
        ],
        "Reinforcement Learning": [
            "Game playing",
            "Robotics",
            "Autonomous vehicles",
            "Trading systems",
        ],
    }

    for domain, examples in applications.items():
        print(f"  {domain}:")
        for example in examples:
            print(f"    • {example}")
        print()

    # 4. Neural Network Types
    print("4. NEURAL NETWORK TYPES:")
    print("-" * 30)

    network_types = {
        "Feedforward Neural Networks": "Basic neural networks with forward connections only",
        "Convolutional Neural Networks (CNNs)": "Specialized for image and spatial data",
        "Recurrent Neural Networks (RNNs)": "Designed for sequential and time series data",
        "Long Short-Term Memory (LSTM)": "Advanced RNN for long-term dependencies",
        "Generative Adversarial Networks (GANs)": "Generate new data similar to training data",
        "Transformers": "Modern architecture for NLP tasks",
    }

    for network_type, description in network_types.items():
        print(f"  {network_type}: {description}")
    print()


def demonstrate_neural_network_fundamentals():
    """Demonstrate neural network fundamentals and concepts."""
    print("Neural Network Fundamentals:")
    print("-" * 40)

    print("Neural networks are inspired by biological neurons and")
    print("consist of interconnected nodes (neurons) organized in layers.")
    print()

    # 1. Basic Components
    print("1. BASIC COMPONENTS:")
    print("-" * 25)

    components = {
        "Input Layer": "Receives input features",
        "Hidden Layers": "Process information through weighted connections",
        "Output Layer": "Produces final predictions",
        "Weights": "Connection strengths between neurons",
        "Biases": "Additional parameters for each neuron",
        "Activation Functions": "Non-linear transformations applied to neuron outputs",
    }

    for component, description in components.items():
        print(f"  {component}: {description}")
    print()

    # 2. Activation Functions
    print("2. ACTIVATION FUNCTIONS:")
    print("-" * 30)

    # Create data for activation function visualization
    x = np.linspace(-5, 5, 1000)

    # Common activation functions
    relu = np.maximum(0, x)
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)
    leaky_relu = np.where(x > 0, x, 0.01 * x)

    # Plot activation functions
    plt.figure(figsize=(15, 10))

    # ReLU
    plt.subplot(2, 3, 1)
    plt.plot(x, relu, "b-", linewidth=2)
    plt.title("ReLU (Rectified Linear Unit)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    plt.axvline(x=0, color="k", linestyle="-", alpha=0.3)

    # Sigmoid
    plt.subplot(2, 3, 2)
    plt.plot(x, sigmoid, "r-", linewidth=2)
    plt.title("Sigmoid")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    plt.axvline(x=0, color="k", linestyle="-", alpha=0.3)

    # Tanh
    plt.subplot(2, 3, 3)
    plt.plot(x, tanh, "g-", linewidth=2)
    plt.title("Tanh (Hyperbolic Tangent)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    plt.axvline(x=0, color="k", linestyle="-", alpha=0.3)

    # Leaky ReLU
    plt.subplot(2, 3, 4)
    plt.plot(x, leaky_relu, "m-", linewidth=2)
    plt.title("Leaky ReLU")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color="k", linestyle="-", alpha=0.3)
    plt.axvline(x=0, color="k", linestyle="-", alpha=0.3)

    # Activation function comparison
    plt.subplot(2, 3, 5)
    plt.plot(x, relu, "b-", linewidth=2, label="ReLU")
    plt.plot(x, sigmoid, "r-", linewidth=2, label="Sigmoid")
    plt.plot(x, tanh, "g-", linewidth=2, label="Tanh")
    plt.title("Activation Functions Comparison")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-3, 3)
    plt.ylim(-1.5, 1.5)

    # Gradient visualization (derivatives)
    plt.subplot(2, 3, 6)
    # ReLU derivative
    relu_derivative = np.where(x > 0, 1, 0)
    # Sigmoid derivative
    sigmoid_derivative = sigmoid * (1 - sigmoid)
    # Tanh derivative
    tanh_derivative = 1 - tanh**2

    plt.plot(x, relu_derivative, "b-", linewidth=2, label="ReLU'")
    plt.plot(x, sigmoid_derivative, "r-", linewidth=2, label="Sigmoid'")
    plt.plot(x, tanh_derivative, "g-", linewidth=2, label="Tanh'")
    plt.title("Activation Function Derivatives")
    plt.xlabel("x")
    plt.ylabel("f'(x)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(-3, 3)

    plt.tight_layout()
    plt.savefig("activation_functions.png", dpi=300, bbox_inches="tight")
    print("✅ Activation functions visualization saved as 'activation_functions.png'")
    plt.close()

    # 3. Forward Propagation
    print("3. FORWARD PROPAGATION:")
    print("-" * 30)

    print("Forward propagation is the process of computing outputs")
    print("from inputs through the neural network layers:")
    print()

    forward_steps = [
        "1. Input data is fed to the input layer",
        "2. Each neuron computes: output = activation(weighted_sum + bias)",
        "3. Outputs are passed to the next layer",
        "4. Process continues until the output layer",
        "5. Final output represents the network's prediction",
    ]

    for step in forward_steps:
        print(f"  {step}")
    print()

    # 4. Loss Functions
    print("4. LOSS FUNCTIONS:")
    print("-" * 25)

    loss_functions = {
        "Mean Squared Error (MSE)": "Used for regression problems",
        "Binary Cross-Entropy": "Used for binary classification",
        "Categorical Cross-Entropy": "Used for multi-class classification",
        "Huber Loss": "Robust loss function for regression",
        "Focal Loss": "Addresses class imbalance in classification",
    }

    for loss_func, description in loss_functions.items():
        print(f"  {loss_func}: {description}")
    print()


# Define SimpleNeuralNetwork class globally so it can be used across functions
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
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

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
        m = X.shape[0]

        # Output layer error
        d2 = output - y
        dW2 = np.dot(self.a1.T, d2) / m
        db2 = np.sum(d2, axis=0, keepdims=True) / m

        # Hidden layer error
        d1 = np.dot(d2, self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, d1) / m
        db1 = np.sum(d1, axis=0, keepdims=True) / m

        # Update weights and biases
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def train(self, X, y, epochs):
        losses = []
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)

            # Compute loss
            loss = np.mean((output - y) ** 2)
            losses.append(loss)

            # Backward pass
            self.backward(X, y, output)

            if epoch % 100 == 0:
                print(f"    Epoch {epoch}: Loss = {loss:.6f}")

        return losses


def demonstrate_neural_network_building():
    """Demonstrate building and implementing neural networks."""
    print("Building Neural Networks:")
    print("-" * 40)

    print("Let's build a simple neural network from scratch")
    print("to understand the fundamental concepts.")
    print()

    # 1. Create Sample Data
    print("1. CREATING SAMPLE DATA:")
    print("-" * 30)

    # Generate classification data
    X_clf, y_clf = make_classification(
        n_samples=1000,
        n_features=10,
        n_classes=2,
        n_informative=8,
        n_redundant=2,
        random_state=42,
    )

    # Generate regression data
    X_reg, y_reg = make_regression(
        n_samples=1000, n_features=8, n_targets=1, noise=0.1, random_state=42
    )

    print(f"✅ Created classification dataset: {X_clf.shape}")
    print(f"✅ Created regression dataset: {X_reg.shape}")
    print()

    # 2. Simple Neural Network Implementation
    print("2. SIMPLE NEURAL NETWORK IMPLEMENTATION:")
    print("-" * 45)

    print("✅ Using SimpleNeuralNetwork class with:")
    print("  • Forward propagation")
    print("  • Backward propagation (backpropagation)")
    print("  • Sigmoid activation function")
    print("  • Mean squared error loss")
    print("  • Gradient descent optimization")
    print()

    print("✅ Implemented SimpleNeuralNetwork class with:")
    print("  • Forward propagation")
    print("  • Backward propagation (backpropagation)")
    print("  • Sigmoid activation function")
    print("  • Mean squared error loss")
    print("  • Gradient descent optimization")
    print()

    # 3. Train Neural Network
    print("3. TRAINING NEURAL NETWORK:")
    print("-" * 30)

    # Prepare data
    X_train, X_test, y_train, y_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape targets for neural network
    y_train_reshaped = y_train.reshape(-1, 1)
    y_test_reshaped = y_test.reshape(-1, 1)

    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    print()

    # Create and train neural network
    nn = SimpleNeuralNetwork(
        input_size=10, hidden_size=8, output_size=1, learning_rate=0.01
    )

    print("Training neural network...")
    losses = nn.train(X_train_scaled, y_train_reshaped, epochs=500)
    print()

    # 4. Evaluate Performance
    print("4. EVALUATE PERFORMANCE:")
    print("-" * 30)

    # Make predictions
    train_predictions = (nn.forward(X_train_scaled) > 0.5).astype(int)
    test_predictions = (nn.forward(X_test_scaled) > 0.5).astype(int)

    # Calculate accuracy
    train_accuracy = accuracy_score(y_train, train_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)

    print(f"Training accuracy: {train_accuracy:.3f}")
    print(f"Test accuracy: {test_accuracy:.3f}")
    print()

    # 5. Visualization
    print("5. VISUALIZATION:")
    print("-" * 20)

    plt.figure(figsize=(15, 5))

    # Training loss
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.title("Training Loss Over Time")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.grid(True, alpha=0.3)

    # Predictions vs Actual
    plt.subplot(1, 3, 2)
    plt.scatter(range(len(y_test)), y_test, alpha=0.7, label="Actual", color="blue")
    plt.scatter(
        range(len(test_predictions)),
        test_predictions,
        alpha=0.7,
        label="Predicted",
        color="red",
    )
    plt.title("Predictions vs Actual (Test Set)")
    plt.xlabel("Sample Index")
    plt.ylabel("Class")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Confusion matrix visualization
    plt.subplot(1, 3, 3)
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_test, test_predictions)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Class 0", "Class 1"],
        yticklabels=["Class 0", "Class 1"],
    )
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    plt.tight_layout()
    plt.savefig("neural_network_results.png", dpi=300, bbox_inches="tight")
    print(
        "✅ Neural network results visualization saved as 'neural_network_results.png'"
    )
    plt.close()

    # Store network for later use
    global trained_nn
    trained_nn = nn


def demonstrate_training_optimization():
    """Demonstrate training and optimization techniques."""
    print("Training and Optimization:")
    print("-" * 40)

    print("Neural network training involves several optimization")
    print("techniques to improve performance and convergence.")
    print()

    # 1. Learning Rate Impact
    print("1. LEARNING RATE IMPACT:")
    print("-" * 30)

    # Test different learning rates
    learning_rates = [0.001, 0.01, 0.1, 0.5]
    lr_results = {}

    X_train, X_test, y_train, y_test = train_test_split(
        make_classification(n_samples=500, n_features=5, n_classes=2, random_state=42)[
            0
        ],
        make_classification(n_samples=500, n_features=5, n_classes=2, random_state=42)[
            1
        ],
        test_size=0.2,
        random_state=42,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    y_train_reshaped = y_train.reshape(-1, 1)

    for lr in learning_rates:
        print(f"Testing learning rate: {lr}")
        nn = SimpleNeuralNetwork(
            input_size=5, hidden_size=4, output_size=1, learning_rate=lr
        )
        losses = nn.train(X_train_scaled, y_train_reshaped, epochs=200)
        lr_results[lr] = losses

    # 2. Overfitting and Regularization
    print("\n2. OVERFITTING AND REGULARIZATION:")
    print("-" * 40)

    print("Overfitting occurs when the model learns training data too well")
    print("but fails to generalize to new data.")
    print()

    regularization_techniques = {
        "Dropout": "Randomly deactivate neurons during training",
        "L1/L2 Regularization": "Add penalty terms to loss function",
        "Early Stopping": "Stop training when validation performance degrades",
        "Data Augmentation": "Increase training data variety",
        "Cross-validation": "Use multiple train/validation splits",
    }

    for technique, description in regularization_techniques.items():
        print(f"  {technique}: {description}")
    print()

    # 3. Optimization Algorithms
    print("3. OPTIMIZATION ALGORITHMS:")
    print("-" * 35)

    optimizers = {
        "Stochastic Gradient Descent (SGD)": "Basic gradient descent with mini-batches",
        "Adam": "Adaptive learning rate with momentum and RMSprop",
        "RMSprop": "Adaptive learning rate based on moving average of squared gradients",
        "Adagrad": "Adaptive learning rate for sparse features",
        "Momentum": "Accelerates training in relevant directions",
    }

    for optimizer, description in optimizers.items():
        print(f"  {optimizer}: {description}")
    print()

    # 4. Visualization of Learning Rate Impact
    print("4. VISUALIZATION OF LEARNING RATE IMPACT:")
    print("-" * 45)

    plt.figure(figsize=(15, 5))

    # Loss curves for different learning rates
    plt.subplot(1, 3, 1)
    for lr, losses in lr_results.items():
        plt.plot(losses, label=f"LR = {lr}")
    plt.title("Training Loss vs Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Final loss comparison
    plt.subplot(1, 3, 2)
    final_losses = [losses[-1] for losses in lr_results.values()]
    plt.bar(
        learning_rates,
        final_losses,
        color=["skyblue", "lightgreen", "lightcoral", "gold"],
    )
    plt.title("Final Loss by Learning Rate")
    plt.xlabel("Learning Rate")
    plt.ylabel("Final Loss")
    plt.grid(True, alpha=0.3)

    # Convergence speed
    plt.subplot(1, 3, 3)
    convergence_epochs = []
    for lr, losses in lr_results.items():
        # Find epoch where loss drops below 0.1
        try:
            conv_epoch = np.where(np.array(losses) < 0.1)[0][0]
        except:
            conv_epoch = len(losses)
        convergence_epochs.append(conv_epoch)

    plt.bar(
        learning_rates,
        convergence_epochs,
        color=["skyblue", "lightgreen", "lightcoral", "gold"],
    )
    plt.title("Convergence Speed by Learning Rate")
    plt.xlabel("Learning Rate")
    plt.ylabel("Epochs to Converge (Loss < 0.1)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_optimization.png", dpi=300, bbox_inches="tight")
    print("✅ Training optimization visualization saved as 'training_optimization.png'")
    plt.close()

    # 5. Best Practices Summary
    print("5. BEST PRACTICES SUMMARY:")
    print("-" * 30)

    best_practices = [
        "Start with a small learning rate (0.01) and adjust based on performance",
        "Use appropriate activation functions for your problem type",
        "Initialize weights properly to avoid vanishing/exploding gradients",
        "Monitor training and validation loss to detect overfitting",
        "Use regularization techniques when overfitting occurs",
        "Normalize/standardize input features for better convergence",
        "Choose appropriate network architecture for your data complexity",
    ]

    for i, practice in enumerate(best_practices, 1):
        print(f"  {i}. {practice}")
    print()

    print("Deep Learning Fundamentals Summary:")
    print("✅ Implemented neural network from scratch")
    print("✅ Demonstrated forward and backward propagation")
    print("✅ Explored activation functions and their properties")
    print("✅ Analyzed learning rate impact on training")
    print("✅ Covered optimization techniques and best practices")


if __name__ == "__main__":
    main()
