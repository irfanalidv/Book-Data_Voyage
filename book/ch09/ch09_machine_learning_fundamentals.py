#!/usr/bin/env python3
"""
Chapter 9: Machine Learning Fundamentals
Data Voyage: Building Predictive Models and Learning from Data

This script covers essential machine learning concepts and algorithms.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


def main():
    print("=" * 80)
    print("CHAPTER 9: MACHINE LEARNING FUNDAMENTALS")
    print("=" * 80)
    print()

    # Section 9.1: Machine Learning Overview and Types
    print("9.1 MACHINE LEARNING OVERVIEW AND TYPES")
    print("-" * 50)
    demonstrate_ml_overview()

    # Section 9.2: Supervised Learning - Regression
    print("\n9.2 SUPERVISED LEARNING - REGRESSION")
    print("-" * 45)
    demonstrate_regression()

    # Section 9.3: Supervised Learning - Classification
    print("\n9.3 SUPERVISED LEARNING - CLASSIFICATION")
    print("-" * 45)
    demonstrate_classification()

    # Section 9.4: Model Evaluation and Validation
    print("\n9.4 MODEL EVALUATION AND VALIDATION")
    print("-" * 45)
    demonstrate_model_evaluation()

    print("\n" + "=" * 80)
    print("CHAPTER SUMMARY")
    print("=" * 80)
    print("✅ Machine learning overview and types")
    print("✅ Supervised learning - regression and classification")
    print("✅ Model evaluation and validation")
    print()
    print("Next: Chapter 10 - Feature Engineering and Selection")
    print("=" * 80)


def demonstrate_ml_overview():
    """Demonstrate machine learning overview and types."""
    print("Machine Learning Overview and Types:")
    print("-" * 40)

    print("Machine Learning is a subset of artificial intelligence that enables")
    print(
        "computers to learn and make decisions from data without explicit programming."
    )
    print()

    # 1. Types of Machine Learning
    print("1. TYPES OF MACHINE LEARNING:")
    print("-" * 30)

    ml_types = {
        "Supervised Learning": {
            "description": "Learning from labeled training data",
            "examples": ["Regression", "Classification"],
            "use_cases": ["Price prediction", "Spam detection", "Medical diagnosis"],
        },
        "Unsupervised Learning": {
            "description": "Finding patterns in unlabeled data",
            "examples": ["Clustering", "Dimensionality reduction", "Association"],
            "use_cases": [
                "Customer segmentation",
                "Market basket analysis",
                "Data compression",
            ],
        },
        "Reinforcement Learning": {
            "description": "Learning through interaction with environment",
            "examples": ["Q-learning", "Policy gradients", "Deep Q-networks"],
            "use_cases": ["Game playing", "Autonomous vehicles", "Robotics"],
        },
    }

    for ml_type, details in ml_types.items():
        print(f"{ml_type}:")
        print(f"  Description: {details['description']}")
        print(f"  Examples: {', '.join(details['examples'])}")
        print(f"  Use Cases: {', '.join(details['use_cases'])}")
        print()

    # 2. Machine Learning Workflow
    print("2. MACHINE LEARNING WORKFLOW:")
    print("-" * 30)

    workflow_steps = [
        "Data Collection and Understanding",
        "Data Preprocessing and Cleaning",
        "Feature Engineering and Selection",
        "Model Selection and Training",
        "Model Evaluation and Validation",
        "Model Deployment and Monitoring",
    ]

    for i, step in enumerate(workflow_steps, 1):
        print(f"  {i}. {step}")
    print()

    # 3. Key Concepts
    print("3. KEY MACHINE LEARNING CONCEPTS:")
    print("-" * 35)

    concepts = {
        "Overfitting": "Model performs well on training data but poorly on new data",
        "Underfitting": "Model is too simple to capture patterns in the data",
        "Bias-Variance Tradeoff": "Balance between model complexity and generalization",
        "Cross-validation": "Technique to assess model performance on unseen data",
        "Feature Importance": "Understanding which variables most influence predictions",
    }

    for concept, description in concepts.items():
        print(f"  {concept}: {description}")
    print()

    # 4. Create sample dataset for demonstrations
    print("4. CREATING SAMPLE DATASET:")
    print("-" * 30)

    np.random.seed(42)
    n_samples = 1000

    # Generate features
    age = np.random.normal(35, 10, n_samples)
    income = np.random.lognormal(10.5, 0.6, n_samples)
    education_years = np.random.poisson(16, n_samples)
    credit_score = (
        300 + (age * 2) + (income / 1000) + np.random.normal(0, 50, n_samples)
    )

    # Generate target variables
    # Regression target: house price
    house_price = (
        200000
        + (income * 2)
        + (age * 1000)
        + (education_years * 5000)
        + np.random.normal(0, 50000, n_samples)
    )

    # Classification target: loan approval (binary) - more balanced
    loan_approval = (credit_score > 600) & (income > 30000) & (age > 20)
    loan_approval = loan_approval.astype(int)

    # Create DataFrame
    data = {
        "age": age,
        "income": income,
        "education_years": education_years,
        "credit_score": credit_score,
        "house_price": house_price,
        "loan_approval": loan_approval,
    }

    df = pd.DataFrame(data)

    print(f"✅ Created dataset with {n_samples} samples and {len(df.columns)} features")
    print(f"Features: age, income, education_years, credit_score")
    print(f"Targets: house_price (regression), loan_approval (classification)")
    print()

    # Store dataset for later use
    global ml_dataset
    ml_dataset = df.copy()

    # Display dataset info
    print("Dataset Overview:")
    print(f"  Shape: {df.shape}")
    print(f"  Features: {list(df.columns[:-2])}")
    print(f"  Regression target: {df.columns[-2]}")
    print(f"  Classification target: {df.columns[-1]}")
    print()

    # Basic statistics
    print("Feature Statistics:")
    print(df.describe().round(2))


def demonstrate_regression():
    """Demonstrate supervised learning - regression."""
    print("Supervised Learning - Regression:")
    print("-" * 40)

    if "ml_dataset" not in globals():
        print("Dataset not available. Please run ML overview first.")
        return

    df = ml_dataset.copy()

    # Prepare data for regression
    X_reg = df[["age", "income", "education_years", "credit_score"]]
    y_reg = df["house_price"]

    print("Regression Problem: Predicting House Price")
    print(f"Features: {list(X_reg.columns)}")
    print(f"Target: {y_reg.name}")
    print(f"Dataset size: {len(df)} samples")
    print()

    # 1. Data Splitting
    print("1. DATA SPLITTING:")
    print("-" * 20)

    X_train, X_test, y_train, y_test = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )

    print(f"Training set: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
    print(f"Test set: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")
    print()

    # 2. Feature Scaling
    print("2. FEATURE SCALING:")
    print("-" * 20)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Features scaled using StandardScaler (Z-score normalization)")
    print("Training set scaled, test set transformed using training parameters")
    print()

    # 3. Model Training
    print("3. MODEL TRAINING:")
    print("-" * 20)

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)

    print("Linear Regression model trained:")
    print(f"  Intercept: ${lr_model.intercept_:,.2f}")
    print("  Feature coefficients:")
    for feature, coef in zip(X_reg.columns, lr_model.coef_):
        print(f"    {feature}: ${coef:,.2f}")
    print()

    # 4. Model Predictions
    print("4. MODEL PREDICTIONS:")
    print("-" * 20)

    # Training predictions
    y_train_pred = lr_model.predict(X_train_scaled)

    # Test predictions
    y_test_pred = lr_model.predict(X_test_scaled)

    print("Predictions generated for training and test sets")
    print(f"Sample predictions (first 5):")
    for i in range(5):
        actual = y_test.iloc[i]
        predicted = y_test_pred[i]
        error = actual - predicted
        print(
            f"  Actual: ${actual:,.0f}, Predicted: ${predicted:,.0f}, Error: ${error:,.0f}"
        )
    print()

    # 5. Model Performance
    print("5. MODEL PERFORMANCE:")
    print("-" * 20)

    # Training performance
    train_mse = mean_squared_error(y_train, y_train_pred)
    train_rmse = np.sqrt(train_mse)
    train_r2 = r2_score(y_train, y_train_pred)

    # Test performance
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_rmse = np.sqrt(test_mse)
    test_r2 = r2_score(y_test, y_test_pred)

    print("Training Performance:")
    print(f"  MSE: ${train_mse:,.0f}")
    print(f"  RMSE: ${train_rmse:,.0f}")
    print(f"  R²: {train_r2:.3f}")
    print()

    print("Test Performance:")
    print(f"  MSE: ${test_mse:,.0f}")
    print(f"  RMSE: ${test_rmse:,.0f}")
    print(f"  R²: {test_r2:.3f}")
    print()

    # 6. Cross-validation
    print("6. CROSS-VALIDATION:")
    print("-" * 20)

    cv_scores = cross_val_score(lr_model, X_train_scaled, y_train, cv=5, scoring="r2")

    print("5-Fold Cross-Validation R² scores:")
    for i, score in enumerate(cv_scores, 1):
        print(f"  Fold {i}: {score:.3f}")
    print(f"  Mean CV R²: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    print()

    # Store model for later use
    global regression_model
    regression_model = lr_model


def demonstrate_classification():
    """Demonstrate supervised learning - classification."""
    print("Supervised Learning - Classification:")
    print("-" * 40)

    if "ml_dataset" not in globals():
        print("Dataset not available. Please run ML overview first.")
        return

    df = ml_dataset.copy()

    # Prepare data for classification
    X_clf = df[["age", "income", "education_years", "credit_score"]]
    y_clf = df["loan_approval"]

    print("Classification Problem: Predicting Loan Approval")
    print(f"Features: {list(X_clf.columns)}")
    print(f"Target: {y_clf.name} (0: Rejected, 1: Approved)")
    print(f"Dataset size: {len(df)} samples")
    print(f"Class distribution:")
    print(
        f"  Rejected: {(y_clf == 0).sum()} ({((y_clf == 0).sum()/len(y_clf)*100):.1f}%)"
    )
    print(
        f"  Approved: {(y_clf == 1).sum()} ({((y_clf == 1).sum()/len(y_clf)*100):.1f}%)"
    )
    print()

    # 1. Data Splitting
    print("1. DATA SPLITTING:")
    print("-" * 20)

    X_train, X_test, y_train, y_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    print(f"Training set: {len(X_train)} samples ({len(X_train)/len(df)*100:.1f}%)")
    print(f"Test set: {len(X_test)} samples ({len(X_test)/len(df)*100:.1f}%)")
    print("Stratified sampling used to maintain class distribution")
    print()

    # 2. Feature Scaling
    print("2. FEATURE SCALING:")
    print("-" * 20)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Features scaled using StandardScaler")
    print()

    # 3. Model Training
    print("3. MODEL TRAINING:")
    print("-" * 20)

    # Logistic Regression
    lr_clf = LogisticRegression(random_state=42, max_iter=1000)
    lr_clf.fit(X_train_scaled, y_train)

    # Decision Tree
    dt_clf = DecisionTreeClassifier(random_state=42, max_depth=5)
    dt_clf.fit(X_train_scaled, y_train)

    # Random Forest
    rf_clf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=5)
    rf_clf.fit(X_train_scaled, y_train)

    models = {
        "Logistic Regression": lr_clf,
        "Decision Tree": dt_clf,
        "Random Forest": rf_clf,
    }

    print("Three classification models trained:")
    for name, model in models.items():
        print(f"  {name}: {type(model).__name__}")
    print()

    # 4. Model Predictions
    print("4. MODEL PREDICTIONS:")
    print("-" * 20)

    predictions = {}
    probabilities = {}

    for name, model in models.items():
        # Predictions
        y_pred = model.predict(X_test_scaled)
        predictions[name] = y_pred

        # Probabilities (for ROC curve)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            probabilities[name] = y_prob

        print(f"{name} predictions generated")

    print()

    # 5. Model Performance Comparison
    print("5. MODEL PERFORMANCE COMPARISON:")
    print("-" * 35)

    print("Accuracy Scores:")
    for name, y_pred in predictions.items():
        accuracy = accuracy_score(y_test, y_pred)
        print(f"  {name}: {accuracy:.3f}")
    print()

    # Store models for later use
    global classification_models, test_predictions, test_probabilities
    classification_models = models
    test_predictions = predictions
    test_probabilities = probabilities


def demonstrate_model_evaluation():
    """Demonstrate model evaluation and validation."""
    print("Model Evaluation and Validation:")
    print("-" * 40)

    if "classification_models" not in globals():
        print("Classification models not available. Please run classification first.")
        return

    # Get the training and test data from the classification function
    df = ml_dataset.copy()
    X_clf = df[["age", "income", "education_years", "credit_score"]]
    y_clf = df["loan_approval"]
    X_train, X_test, y_train, y_test = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # 1. Detailed Classification Report
    print("1. DETAILED CLASSIFICATION REPORT:")
    print("-" * 35)

    # Use Random Forest as example
    rf_model = classification_models["Random Forest"]
    y_pred = test_predictions["Random Forest"]

    print("Random Forest Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Rejected", "Approved"]))

    # 2. Confusion Matrix
    print("2. CONFUSION MATRIX:")
    print("-" * 20)

    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:")
    print("                Predicted")
    print("                Rejected  Approved")
    print(f"Actual Rejected    {cm[0,0]:>8}    {cm[0,1]:>8}")
    print(f"      Approved     {cm[1,0]:>8}    {cm[1,1]:>8}")
    print()

    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    print("Derived Metrics:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1-Score: {f1:.3f}")
    print()

    # 3. ROC Curve and AUC
    print("3. ROC CURVE AND AUC:")
    print("-" * 25)

    if "test_probabilities" in globals() and "Random Forest" in test_probabilities:
        y_prob = test_probabilities["Random Forest"]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        print(f"ROC AUC: {roc_auc:.3f}")
        print()

        # Create ROC curve visualization
        plt.figure(figsize=(15, 10))

        # ROC Curve
        plt.subplot(2, 3, 1)
        plt.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})"
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - Random Forest")
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)

        # Confusion Matrix Heatmap
        plt.subplot(2, 3, 2)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Rejected", "Approved"],
            yticklabels=["Rejected", "Approved"],
        )
        plt.title("Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")

        # Feature Importance (Random Forest)
        plt.subplot(2, 3, 3)
        feature_names = ["age", "income", "education_years", "credit_score"]
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(
            range(len(importances)), [feature_names[i] for i in indices], rotation=45
        )
        plt.title("Feature Importance - Random Forest")
        plt.ylabel("Importance")

        # Model Comparison
        plt.subplot(2, 3, 4)
        model_names = list(classification_models.keys())
        accuracies = [
            accuracy_score(y_test, test_predictions[name]) for name in model_names
        ]

        plt.bar(model_names, accuracies, color=["skyblue", "lightgreen", "lightcoral"])
        plt.title("Model Accuracy Comparison")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        for i, v in enumerate(accuracies):
            plt.text(i, v + 0.01, f"{v:.3f}", ha="center", va="bottom")

        # Prediction Distribution
        plt.subplot(2, 3, 5)
        plt.hist(
            y_test,
            alpha=0.7,
            label="Actual",
            bins=2,
            color="skyblue",
            edgecolor="black",
        )
        plt.hist(
            y_pred,
            alpha=0.7,
            label="Predicted",
            bins=2,
            color="lightgreen",
            edgecolor="black",
        )
        plt.title("Actual vs Predicted Distribution")
        plt.xlabel("Loan Approval (0: Rejected, 1: Approved)")
        plt.ylabel("Count")
        plt.legend()
        plt.xticks([0, 1], ["Rejected", "Approved"])

        # Error Analysis
        plt.subplot(2, 3, 6)
        errors = y_test != y_pred
        error_rate = errors.mean()

        plt.pie(
            [1 - error_rate, error_rate],
            labels=["Correct", "Incorrect"],
            autopct="%1.1f%%",
            colors=["lightgreen", "lightcoral"],
        )
        plt.title(f"Prediction Accuracy\nError Rate: {error_rate:.1%}")

        plt.tight_layout()
        plt.savefig("model_evaluation.png", dpi=300, bbox_inches="tight")
        print("✅ Model evaluation visualization saved as 'model_evaluation.png'")
        plt.close()

    # 4. Cross-validation for all models
    print("4. CROSS-VALIDATION RESULTS:")
    print("-" * 30)

    cv_results = {}
    for name, model in classification_models.items():
        cv_scores = cross_val_score(
            model, X_train_scaled, y_train, cv=5, scoring="accuracy"
        )
        cv_results[name] = cv_scores

        print(f"{name}:")
        print(f"  CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        print(f"  Individual CV scores: {[f'{score:.3f}' for score in cv_scores]}")
        print()

    # 5. Model Selection Recommendations
    print("5. MODEL SELECTION RECOMMENDATIONS:")
    print("-" * 35)

    print("Based on the analysis:")
    print("  - Random Forest shows best performance with feature importance insights")
    print("  - Cross-validation confirms model stability")
    print("  - Consider ensemble methods for production use")
    print("  - Feature engineering could further improve performance")
    print()

    # Final summary
    print("MODEL EVALUATION SUMMARY:")
    print("-" * 30)
    print("✅ Classification reports and confusion matrices generated")
    print("✅ ROC curves and AUC scores calculated")
    print("✅ Feature importance analysis completed")
    print("✅ Cross-validation results obtained")
    print("✅ Model comparison and recommendations provided")
    print()
    print("Machine Learning Fundamentals complete!")
    print(
        "Key concepts demonstrated: supervised learning, model training, and evaluation."
    )


if __name__ == "__main__":
    main()
