#!/usr/bin/env python3
"""
Chapter 18: Model Deployment and MLOps
======================================

This chapter covers essential practices for deploying machine learning models
to production and implementing MLOps workflows for continuous improvement.

Topics Covered:
- Model Deployment Strategies and Architectures
- MLOps and Model Lifecycle Management
- Production Infrastructure and Containerization
- Model Monitoring and Observability
- Security and Compliance Considerations
"""

import os
import sys
import time
import json
import pickle
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Any, Tuple

# Machine Learning imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# Set up plotting style
plt.style.use("default")
sns.set_palette("husl")
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
np.random.seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ModelDeploymentDemo:
    """Demonstration class for model deployment concepts."""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_version = "1.0.0"
        self.deployment_timestamp = datetime.now()
        self.performance_metrics = {}
        
    def create_training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create synthetic data for demonstration."""
        print("1. CREATING TRAINING DATASET:")
        print("-" * 35)
        
        # Generate synthetic data
        n_samples = 2000
        n_features = 20
        
        X = np.random.randn(n_samples, n_features)
        
        # Create target with complex relationships
        target = (
            0.3 * X[:, 0] ** 2 +
            0.4 * X[:, 1] * X[:, 2] +
            0.2 * np.sin(X[:, 3]) +
            0.1 * np.exp(X[:, 4]) +
            np.random.normal(0, 0.1, n_samples)
        )
        
        y = (target > np.median(target)).astype(int)
        
        print(f"  ✅ Dataset created: {n_samples:,} samples, {n_features} features")
        print(f"  📊 Target distribution: {np.bincount(y)}")
        print(f"  🎯 Binary classification task")
        
        return X, y
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train a machine learning model."""
        print("\n2. TRAINING MACHINE LEARNING MODEL:")
        print("-" * 40)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        start_time = time.time()
        self.model.fit(X_train_scaled, y_train)
        training_time = time.time() - start_time
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"  🚀 Model trained successfully!")
        print(f"  ⏱️  Training time: {training_time:.4f}s")
        print(f"  📊 Test accuracy: {accuracy:.4f}")
        print(f"  🔢 Model parameters: {self.model.get_params()}")
        
        # Store performance metrics
        self.performance_metrics = {
            "accuracy": accuracy,
            "training_time": training_time,
            "n_samples": len(X_train),
            "n_features": X_train.shape[1]
        }
    
    def demonstrate_model_packaging(self) -> None:
        """Demonstrate model packaging and serialization."""
        print("\n3. MODEL PACKAGING AND SERIALIZATION:")
        print("-" * 45)
        
        # Create model artifacts
        model_artifacts = {
            "model": self.model,
            "scaler": self.scaler,
            "metadata": {
                "version": self.model_version,
                "created_at": self.deployment_timestamp.isoformat(),
                "algorithm": "RandomForestClassifier",
                "performance": self.performance_metrics
            }
        }
        
        # Save model using different methods
        print("  📦 Saving model artifacts...")
        
        # Method 1: Pickle (simple but less secure)
        with open("model.pkl", "wb") as f:
            pickle.dump(model_artifacts, f)
        print(f"  ✅ Pickle format: model.pkl ({os.path.getsize('model.pkl')} bytes)")
        
        # Method 2: Joblib (better for large numpy arrays)
        joblib.dump(model_artifacts, "model.joblib")
        print(f"  ✅ Joblib format: model.joblib ({os.path.getsize('model.joblib')} bytes)")
        
        # Method 3: JSON metadata + separate model files
        metadata_file = "model_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(model_artifacts["metadata"], f, indent=2)
        print(f"  ✅ JSON metadata: {metadata_file}")
        
        # Method 4: ONNX format (for production deployment)
        try:
            import onnx
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
            
            # Convert to ONNX
            initial_type = [("float_input", FloatTensorType([None, 20]))]
            onx = convert_sklearn(self.model, initial_types=initial_type)
            
            with open("model.onnx", "wb") as f:
                f.write(onx.SerializeToString())
            print(f"  ✅ ONNX format: model.onnx ({os.path.getsize('model.onnx')} bytes)")
        except ImportError:
            print("  ⚠️  ONNX conversion skipped (skl2onnx not available)")
        
        print(f"  📊 Total artifacts created: {len([f for f in os.listdir('.') if f.startswith('model')])} files")
    
    def demonstrate_api_design(self) -> None:
        """Demonstrate API design for model serving."""
        print("\n4. API DESIGN AND MODEL SERVING:")
        print("-" * 40)
        
        # Simulate API endpoints
        api_endpoints = {
            "/health": "GET - Health check endpoint",
            "/predict": "POST - Single prediction endpoint",
            "/predict_batch": "POST - Batch prediction endpoint",
            "/model_info": "GET - Model metadata and performance",
            "/retrain": "POST - Trigger model retraining"
        }
        
        print("  🌐 API Endpoints Design:")
        for endpoint, description in api_endpoints.items():
            print(f"    {endpoint:20}: {description}")
        
        # Simulate prediction service
        print("\n  🔮 Prediction Service Simulation:")
        
        # Single prediction
        sample_input = np.random.randn(1, 20)
        sample_scaled = self.scaler.transform(sample_input)
        prediction = self.model.predict(sample_scaled)[0]
        probability = self.model.predict_proba(sample_scaled)[0]
        
        print(f"    Single prediction: {prediction} (confidence: {max(probability):.3f})")
        
        # Batch prediction
        batch_input = np.random.randn(5, 20)
        batch_scaled = self.scaler.transform(batch_input)
        batch_predictions = self.model.predict(batch_scaled)
        batch_probabilities = self.model.predict_proba(batch_scaled)
        
        print(f"    Batch predictions: {batch_predictions}")
        print(f"    Average confidence: {np.mean(np.max(batch_probabilities, axis=1)):.3f}")
        
        # API response format
        api_response = {
            "prediction": int(prediction),
            "confidence": float(max(probability)),
            "model_version": self.model_version,
            "timestamp": datetime.now().isoformat(),
            "input_features": sample_input.tolist()
        }
        
        print(f"    API Response: {json.dumps(api_response, indent=2)}")
    
    def demonstrate_mlops_practices(self) -> None:
        """Demonstrate MLOps practices and workflows."""
        print("\n5. MLOPS PRACTICES AND WORKFLOWS:")
        print("-" * 45)
        
        # Model versioning
        print("  📋 Model Versioning:")
        versions = [
            {"version": "1.0.0", "accuracy": 0.8234, "deployed": "2024-01-01", "status": "Production"},
            {"version": "1.1.0", "accuracy": 0.8456, "deployed": "2024-02-01", "status": "Production"},
            {"version": "1.2.0", "accuracy": 0.8567, "deployed": "2024-03-01", "status": "Production"},
            {"version": "2.0.0", "accuracy": 0.8789, "deployed": "2024-04-01", "status": "Staging"},
            {"version": "2.1.0", "accuracy": 0.8912, "deployed": "2024-05-01", "status": "Development"}
        ]
        
        for version_info in versions:
            status_icon = "🟢" if version_info["status"] == "Production" else "🟡" if version_info["status"] == "Staging" else "🔴"
            print(f"    {status_icon} v{version_info['version']:6}: Accuracy: {version_info['accuracy']:.4f}, "
                  f"Deployed: {version_info['deployed']}, Status: {version_info['status']}")
        
        # Experiment tracking
        print("\n  🔬 Experiment Tracking:")
        experiments = [
            {"id": "exp_001", "algorithm": "RandomForest", "accuracy": 0.8234, "params": {"n_estimators": 100}},
            {"id": "exp_002", "algorithm": "RandomForest", "accuracy": 0.8456, "params": {"n_estimators": 200}},
            {"id": "exp_003", "algorithm": "GradientBoosting", "accuracy": 0.8567, "params": {"n_estimators": 100}},
            {"id": "exp_004", "algorithm": "XGBoost", "accuracy": 0.8789, "params": {"max_depth": 6}}
        ]
        
        for exp in experiments:
            print(f"    {exp['id']:8}: {exp['algorithm']:15} - Accuracy: {exp['accuracy']:.4f}")
        
        # CI/CD pipeline simulation
        print("\n  🔄 CI/CD Pipeline:")
        pipeline_steps = [
            "Code commit and push",
            "Automated testing",
            "Model training and validation",
            "Performance benchmarking",
            "Security scanning",
            "Deployment to staging",
            "Integration testing",
            "Production deployment"
        ]
        
        for i, step in enumerate(pipeline_steps, 1):
            print(f"    {i:2d}. {step}")
    
    def demonstrate_monitoring_and_observability(self) -> None:
        """Demonstrate model monitoring and observability."""
        print("\n6. MODEL MONITORING AND OBSERVABILITY:")
        print("-" * 50)
        
        # Performance monitoring
        print("  📊 Performance Monitoring:")
        
        # Simulate performance over time
        time_periods = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"]
        accuracy_trend = [0.8234, 0.8256, 0.8234, 0.8212, 0.8198, 0.8189, 0.8176]
        latency_trend = [120, 118, 122, 125, 128, 130, 132]  # ms
        request_count = [1000, 1050, 1100, 1150, 1200, 1250, 1300]
        
        print(f"    Time Periods: {time_periods}")
        print(f"    Accuracy Trend: {[f'{acc:.4f}' for acc in accuracy_trend]}")
        print(f"    Latency Trend: {latency_trend} ms")
        print(f"    Request Count: {request_count}")
        
        # Drift detection
        print("\n  🚨 Drift Detection:")
        
        # Simulate feature drift
        feature_drift = {
            "feature_0": {"drift_score": 0.12, "status": "Normal"},
            "feature_1": {"drift_score": 0.34, "status": "Warning"},
            "feature_2": {"drift_score": 0.67, "status": "Alert"},
            "feature_3": {"drift_score": 0.23, "status": "Normal"},
            "feature_4": {"drift_score": 0.45, "status": "Warning"}
        }
        
        for feature, info in feature_drift.items():
            status_icon = "🟢" if info["status"] == "Normal" else "🟡" if info["status"] == "Warning" else "🔴"
            print(f"    {status_icon} {feature:10}: Drift Score: {info['drift_score']:.2f}, Status: {info['status']}")
        
        # Alerting system
        print("\n  ⚠️  Alerting System:")
        alerts = [
            {"level": "INFO", "message": "Model performance within normal range", "timestamp": "2024-05-01 10:00:00"},
            {"level": "WARNING", "message": "Feature drift detected in feature_1", "timestamp": "2024-05-01 10:15:00"},
            {"level": "ALERT", "message": "High drift detected in feature_2", "timestamp": "2024-05-01 10:30:00"},
            {"level": "WARNING", "message": "Latency increased by 10%", "timestamp": "2024-05-01 10:45:00"}
        ]
        
        for alert in alerts:
            level_icon = {"INFO": "ℹ️", "WARNING": "⚠️", "ALERT": "🚨"}[alert["level"]]
            print(f"    {level_icon} [{alert['level']:7}] {alert['message']} - {alert['timestamp']}")
    
    def demonstrate_production_infrastructure(self) -> None:
        """Demonstrate production infrastructure concepts."""
        print("\n7. PRODUCTION INFRASTRUCTURE:")
        print("-" * 40)
        
        # Containerization
        print("  🐳 Containerization:")
        
        dockerfile_content = """
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY model.pkl .
COPY app.py .

EXPOSE 8000

CMD ["python", "app.py"]
        """.strip()
        
        print("    Dockerfile created for model serving")
        print(f"    Base image: python:3.9-slim")
        print(f"    Working directory: /app")
        print(f"    Exposed port: 8000")
        
        # Kubernetes deployment
        print("\n  ☸️  Kubernetes Deployment:")
        
        k8s_deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "ml-model-deployment"},
            "spec": {
                "replicas": 3,
                "selector": {"matchLabels": {"app": "ml-model"}},
                "template": {
                    "metadata": {"labels": {"app": "ml-model"}},
                    "spec": {
                        "containers": [{
                            "name": "ml-model",
                            "image": "ml-model:latest",
                            "ports": [{"containerPort": 8000}],
                            "resources": {
                                "requests": {"memory": "512Mi", "cpu": "250m"},
                                "limits": {"memory": "1Gi", "cpu": "500m"}
                            }
                        }]
                    }
                }
            }
        }
        
        print("    Deployment configured with 3 replicas")
        print("    Resource limits: 1Gi memory, 500m CPU")
        print("    Auto-scaling enabled")
        
        # Load balancing
        print("\n  ⚖️  Load Balancing:")
        
        load_balancer_config = {
            "algorithm": "round_robin",
            "health_check": "/health",
            "session_persistence": True,
            "ssl_termination": True,
            "rate_limiting": "1000 requests/minute"
        }
        
        for key, value in load_balancer_config.items():
            print(f"    {key.replace('_', ' ').title()}: {value}")
    
    def create_deployment_visualizations(self) -> None:
        """Create comprehensive visualizations for deployment concepts."""
        print("\n8. CREATING DEPLOYMENT VISUALIZATIONS:")
        print("-" * 45)
        
        print("  Generating visualizations...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Model Deployment and MLOps: Production Systems Overview", fontsize=16, fontweight="bold")
        
        # 1. Model Performance Over Time
        ax1 = axes[0, 0]
        time_periods = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7"]
        accuracy_trend = [0.8234, 0.8256, 0.8234, 0.8212, 0.8198, 0.8189, 0.8176]
        latency_trend = [120, 118, 122, 125, 128, 130, 132]
        
        ax1_twin = ax1.twinx()
        line1 = ax1.plot(time_periods, accuracy_trend, "o-", color="#FF6B6B", linewidth=2, markersize=8, label="Accuracy")
        line2 = ax1_twin.plot(time_periods, latency_trend, "s-", color="#4ECDC4", linewidth=2, markersize=8, label="Latency (ms)")
        
        ax1.set_title("Model Performance Over Time", fontweight="bold")
        ax1.set_ylabel("Accuracy", color="#FF6B6B")
        ax1_twin.set_ylabel("Latency (ms)", color="#4ECDC4")
        ax1.grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="upper right")
        
        # 2. Feature Drift Analysis
        ax2 = axes[0, 1]
        features = [f"Feature_{i}" for i in range(10)]
        drift_scores = np.random.uniform(0.1, 0.7, 10)
        colors = ["#FF6B6B" if score > 0.5 else "#4ECDC4" if score > 0.3 else "#98D8C8" for score in drift_scores]
        
        bars = ax2.bar(features, drift_scores, color=colors, alpha=0.8)
        ax2.set_title("Feature Drift Analysis", fontweight="bold")
        ax2.set_ylabel("Drift Score")
        ax2.tick_params(axis="x", rotation=45)
        ax2.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="Alert Threshold")
        ax2.axhline(y=0.3, color="orange", linestyle="--", alpha=0.7, label="Warning Threshold")
        ax2.legend()
        
        # 3. Model Version Performance
        ax3 = axes[0, 2]
        versions = ["v1.0", "v1.1", "v1.2", "v2.0", "v2.1"]
        accuracies = [0.8234, 0.8456, 0.8567, 0.8789, 0.8912]
        deployment_dates = ["Jan", "Feb", "Mar", "Apr", "May"]
        
        bars = ax3.bar(versions, accuracies, color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFA07A"], alpha=0.8)
        ax3.set_title("Model Version Performance", fontweight="bold")
        ax3.set_ylabel("Accuracy")
        ax3.set_ylim(0.8, 0.95)
        
        # Add value labels
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f"{acc:.3f}", ha="center", va="bottom", fontweight="bold")
        
        # 4. Request Volume and Latency
        ax4 = axes[1, 0]
        time_points = ["00:00", "06:00", "12:00", "18:00", "24:00"]
        request_volume = [500, 300, 800, 1200, 600]
        avg_latency = [110, 105, 125, 140, 115]
        
        ax4_twin = ax4.twinx()
        bars = ax4.bar(time_points, request_volume, color="#98D8C8", alpha=0.7, label="Request Volume")
        line = ax4_twin.plot(time_points, avg_latency, "o-", color="#FF6B6B", linewidth=2, markersize=8, label="Avg Latency")
        
        ax4.set_title("Daily Traffic Patterns", fontweight="bold")
        ax4.set_ylabel("Request Volume", color="#98D8C8")
        ax4_twin.set_ylabel("Latency (ms)", color="#FF6B6B")
        
        # Combine legends
        ax4.legend([bars, line[0]], ["Request Volume", "Avg Latency"], loc="upper left")
        
        # 5. Resource Utilization
        ax5 = axes[1, 1]
        resources = ["CPU", "Memory", "Network", "Disk"]
        utilization = [65, 78, 45, 32]  # Percentage
        
        bars = ax5.bar(resources, utilization, color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"], alpha=0.8)
        ax5.set_title("Resource Utilization", fontweight="bold")
        ax5.set_ylabel("Utilization (%)")
        ax5.set_ylim(0, 100)
        
        # Add value labels
        for bar, util in zip(bars, utilization):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f"{util}%", ha="center", va="bottom", fontweight="bold")
        
        # 6. Deployment Pipeline
        ax6 = axes[1, 2]
        pipeline_steps = ["Code\nCommit", "Build\n& Test", "Model\nTraining", "Validation", "Deploy\nStaging", "Production\nDeploy"]
        step_status = ["✅", "✅", "✅", "✅", "🔄", "⏳"]
        
        y_pos = np.arange(len(pipeline_steps))
        bars = ax6.barh(y_pos, [1]*len(pipeline_steps), color=["#98D8C8" if status == "✅" else "#FFA07A" if status == "🔄" else "#D3D3D3" for status in step_status], alpha=0.8)
        
        ax6.set_title("Deployment Pipeline Status", fontweight="bold")
        ax6.set_yticks(y_pos)
        ax6.set_yticklabels(pipeline_steps)
        ax6.set_xlim(0, 1.2)
        
        # Add status icons
        for i, status in enumerate(step_status):
            ax6.text(0.5, i, status, ha="center", va="center", fontsize=16)
        
        plt.tight_layout()
        
        # Save visualization
        output_file = "model_deployment_mlops.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"  ✅ Visualization saved: {output_file}")
        
        plt.show()

def main():
    """Main function to run all demonstrations."""
    try:
        print("=" * 80)
        print("CHAPTER 18: MODEL DEPLOYMENT AND MLOPS")
        print("=" * 80)
        
        # Initialize demo
        demo = ModelDeploymentDemo()
        
        # Run all demonstrations
        X, y = demo.create_training_data()
        demo.train_model(X, y)
        demo.demonstrate_model_packaging()
        demo.demonstrate_api_design()
        demo.demonstrate_mlops_practices()
        demo.demonstrate_monitoring_and_observability()
        demo.demonstrate_production_infrastructure()
        demo.create_deployment_visualizations()
        
        print("\n" + "=" * 80)
        print("CHAPTER 18 COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("\n🎯 What You've Learned:")
        print("  • Model deployment strategies and architectures")
        print("  • MLOps practices and model lifecycle management")
        print("  • Production infrastructure and containerization")
        print("  • Model monitoring and observability")
        print("  • Security and compliance considerations")
        
        print("\n📊 Generated Visualizations:")
        print("  • model_deployment_mlops.png - Comprehensive deployment dashboard")
        
        print("\n🚀 Next Steps:")
        print("  • Practice deploying models to cloud platforms")
        print("  • Implement monitoring and alerting systems")
        print("  • Build CI/CD pipelines for ML")
        print("  • Continue to Chapter 19: Real-World Case Studies")
        
        # Clean up generated files
        cleanup_files = ["model.pkl", "model.joblib", "model_metadata.json"]
        for file in cleanup_files:
            if os.path.exists(file):
                os.remove(file)
                print(f"  🧹 Cleaned up: {file}")
        
    except Exception as e:
        print(f"\n❌ Error in Chapter 18: {str(e)}")
        print("Please check the error and try again.")

if __name__ == "__main__":
    main()
