#!/usr/bin/env python3
"""
Chapter 10: Feature Engineering and Selection
Data Voyage: Creating Powerful Features for Better Machine Learning Models with Real Data

This script covers essential feature engineering and selection techniques using REAL DATA.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import (
    SelectKBest, f_regression, f_classif, RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer, load_wine
import warnings

warnings.filterwarnings("ignore")

def main():
    print("=" * 80)
    print("CHAPTER 10: FEATURE ENGINEERING AND SELECTION")
    print("=" * 80)
    print()
    
    # Section 10.1: Feature Engineering Fundamentals
    print("10.1 FEATURE ENGINEERING FUNDAMENTALS")
    print("-" * 50)
    demonstrate_feature_engineering_fundamentals()
    
    # Section 10.2: Advanced Feature Engineering
    print("\n10.2 ADVANCED FEATURE ENGINEERING")
    print("-" * 45)
    demonstrate_advanced_feature_engineering()
    
    # Section 10.3: Feature Selection Methods
    print("\n10.3 FEATURE SELECTION METHODS")
    print("-" * 40)
    demonstrate_feature_selection()
    
    # Section 10.4: Dimensionality Reduction
    print("\n10.4 DIMENSIONALITY REDUCTION")
    print("-" * 40)
    demonstrate_dimensionality_reduction()
    
    print("\n" + "=" * 80)
    print("CHAPTER SUMMARY")
    print("=" * 80)
    print("✅ Feature engineering fundamentals and techniques on real data")
    print("✅ Advanced feature creation and transformation using actual datasets")
    print("✅ Feature selection methods and evaluation with real features")
    print("✅ Dimensionality reduction techniques on authentic data")
    print()
    print("Next: Chapter 11 - Unsupervised Learning")
    print("=" * 80)

def demonstrate_feature_engineering_fundamentals():
    """Demonstrate fundamental feature engineering techniques using real data."""
    print("Feature Engineering Fundamentals:")
    print("-" * 40)
    
    print("Feature engineering is the process of creating new features from")
    print("existing data to improve machine learning model performance.")
    print()
    
    # 1. Load real datasets
    print("1. LOADING REAL DATASETS:")
    print("-" * 30)
    
    iris = load_iris()
    diabetes = load_diabetes()
    breast_cancer = load_breast_cancer()
    wine = load_wine()
    
    # Create DataFrames
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target
    iris_df['species'] = [iris.target_names[i] for i in iris.target]
    
    diabetes_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    diabetes_df['target'] = diabetes.target
    
    breast_cancer_df = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    breast_cancer_df['target'] = breast_cancer.target
    breast_cancer_df['diagnosis'] = ['Malignant' if t == 1 else 'Benign' for t in breast_cancer.target]
    
    wine_df = pd.DataFrame(wine.data, columns=wine.feature_names)
    wine_df['target'] = wine.target
    wine_df['wine_type'] = [wine.target_names[i] for i in wine.target]
    
    print(f"✅ Loaded real datasets:")
    print(f"  • Iris: {iris_df.shape[0]} samples, {iris_df.shape[1]-2} features")
    print(f"  • Diabetes: {diabetes_df.shape[0]} samples, {diabetes_df.shape[1]-1} features")
    print(f"  • Breast Cancer: {breast_cancer_df.shape[0]} samples, {breast_cancer_df.shape[1]-2} features")
    print(f"  • Wine: {wine_df.shape[0]} samples, {wine_df.shape[1]-2} features")
    print()
    
    # 2. Use Iris dataset for feature engineering demonstration
    print("2. FEATURE ENGINEERING ON IRIS DATASET:")
    print("-" * 40)
    
    df = iris_df.copy()
    print(f"Working with Iris dataset: {df.shape[0]} samples, {df.shape[1]-2} original features")
    print(f"Original features: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']")
    print(f"Target: {df['target'].nunique()} classes")
    print()
    
    # 3. Basic Feature Engineering
    print("3. BASIC FEATURE ENGINEERING:")
    print("-" * 30)
    
    # Create area features
    df['sepal_area'] = df['sepal length (cm)'] * df['sepal width (cm)']
    df['petal_area'] = df['petal length (cm)'] * df['petal width (cm)']
    
    # Create ratio features
    df['sepal_length_width_ratio'] = df['sepal length (cm)'] / df['sepal width (cm)']
    df['petal_length_width_ratio'] = df['petal length (cm)'] / df['petal width (cm)']
    df['petal_to_sepal_ratio'] = df['petal_area'] / df['sepal_area']
    
    # Create perimeter features
    df['sepal_perimeter'] = 2 * (df['sepal length (cm)'] + df['sepal width (cm)'])
    df['petal_perimeter'] = 2 * (df['petal length (cm)'] + df['petal width (cm)'])
    
    # Create size categories
    df['size_category'] = pd.cut(df['sepal_area'], 
                                bins=[0, 15, 20, float('inf')], 
                                labels=['Small', 'Medium', 'Large'])
    
    print(f"✅ Created {8} new engineered features:")
    print(f"  • Area features: sepal_area, petal_area")
    print(f"  • Ratio features: sepal_length_width_ratio, petal_length_width_ratio, petal_to_sepal_ratio")
    print(f"  • Perimeter features: sepal_perimeter, petal_perimeter")
    print(f"  • Categorical features: size_category")
    print()
    
    # 4. Feature Statistics
    print("4. FEATURE STATISTICS:")
    print("-" * 25)
    
    # Show statistics for new features
    new_features = ['sepal_area', 'petal_area', 'sepal_length_width_ratio', 
                   'petal_length_width_ratio', 'petal_to_sepal_ratio']
    
    print("New Engineered Features Statistics:")
    print(df[new_features].describe().round(3))
    print()
    
    # 5. Feature Correlations
    print("5. FEATURE CORRELATIONS:")
    print("-" * 30)
    
    # Calculate correlations with target
    iris_feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    all_features = iris_feature_names + new_features
    correlations = df[all_features].corrwith(df['target']).abs().sort_values(ascending=False)
    
    print("Feature Correlations with Target (Top 10):")
    for i, (feature, corr) in enumerate(correlations.head(10).items()):
        print(f"  {i+1:2d}. {feature:25s}: {corr:.3f}")
    print()
    
    # Store dataset globally for other functions
    global engineered_dataset
    engineered_dataset = df
    
    return df

def demonstrate_advanced_feature_engineering():
    """Demonstrate advanced feature engineering techniques using real data."""
    print("Advanced Feature Engineering:")
    print("-" * 40)
    
    if 'engineered_dataset' not in globals() or engineered_dataset is None:
        print("❌ No engineered dataset available")
        return
    
    df = engineered_dataset.copy()
    
    # 1. Polynomial Features
    print("1. POLYNOMIAL FEATURES:")
    print("-" * 25)
    
    # Select numerical features for polynomial transformation
    numerical_features = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    X_poly = df[numerical_features].values
    
    # Create polynomial features (degree 2)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly_transformed = poly.fit_transform(X_poly)
    
    # Get feature names
    poly_feature_names = poly.get_feature_names_out(numerical_features)
    
    print(f"✅ Created polynomial features (degree 2)")
    print(f"  Original features: {len(numerical_features)}")
    print(f"  Polynomial features: {len(poly_feature_names)}")
    print(f"  New features include: {', '.join(poly_feature_names[4:8])}")  # Show some new features
    print()
    
    # 2. Interaction Features
    print("2. INTERACTION FEATURES:")
    print("-" * 30)
    
    # Create interaction features manually
    df['sepal_petal_length_interaction'] = df['sepal length (cm)'] * df['petal length (cm)']
    df['sepal_petal_width_interaction'] = df['sepal width (cm)'] * df['petal width (cm)']
    df['length_width_interaction'] = (df['sepal length (cm)'] + df['petal length (cm)']) * \
                                    (df['sepal width (cm)'] + df['petal width (cm)'])
    
    print(f"✅ Created {3} interaction features:")
    print(f"  • sepal_petal_length_interaction: Sepal length × Petal length")
    print(f"  • sepal_petal_width_interaction: Sepal width × Petal width")
    print(f"  • length_width_interaction: (Sepal length + Petal length) × (Sepal width + Petal width)")
    print()
    
    # 3. Statistical Features
    print("3. STATISTICAL FEATURES:")
    print("-" * 30)
    
    # Create rolling statistics (using a window approach)
    # For demonstration, we'll create features based on sorted values
    sorted_indices = df['sepal length (cm)'].sort_values().index
    
    # Create rank-based features
    df['sepal_length_rank'] = df['sepal length (cm)'].rank()
    df['petal_length_rank'] = df['petal length (cm)'].rank()
    
    # Create percentile features
    df['sepal_length_percentile'] = df['sepal length (cm)'].rank(pct=True)
    df['petal_length_percentile'] = df['petal length (cm)'].rank(pct=True)
    
    print(f"✅ Created {4} statistical features:")
    print(f"  • sepal_length_rank: Rank of sepal length")
    print(f"  • petal_length_rank: Rank of petal length")
    print(f"  • sepal_length_percentile: Percentile rank of sepal length")
    print(f"  • petal_length_percentile: Percentile rank of petal length")
    print()
    
    # 4. Domain-Specific Features
    print("4. DOMAIN-SPECIFIC FEATURES:")
    print("-" * 35)
    
    # Create features specific to botany/biology
    df['total_length'] = df['sepal length (cm)'] + df['petal length (cm)']
    df['total_width'] = df['sepal width (cm)'] + df['petal width (cm)']
    df['length_width_balance'] = df['total_length'] / df['total_width']
    
    # Create symmetry features
    df['sepal_symmetry'] = abs(df['sepal length (cm)'] - df['sepal width (cm)'])
    df['petal_symmetry'] = abs(df['petal length (cm)'] - df['petal width (cm)'])
    
    # Create compactness features
    df['sepal_compactness'] = df['sepal_area'] / (df['sepal_perimeter'] ** 2)
    df['petal_compactness'] = df['petal_area'] / (df['petal_perimeter'] ** 2)
    
    print(f"✅ Created {7} domain-specific features:")
    print(f"  • Total measurements: total_length, total_width")
    print(f"  • Balance features: length_width_balance")
    print(f"  • Symmetry features: sepal_symmetry, petal_symmetry")
    print(f"  • Compactness features: sepal_compactness, petal_compactness")
    print()
    
    # 5. Feature Summary
    print("5. FEATURE SUMMARY:")
    print("-" * 20)
    
    print(f"Final dataset shape: {df.shape}")
    print(f"Total features: {df.shape[1]}")
    print(f"Original features: 4")
    print(f"Engineered features: {df.shape[1] - 4 - 2}")  # -4 for original features, -2 for target and species
    print()
    
    # Show all feature names
    all_feature_names = [col for col in df.columns if col not in ['target', 'species']]
    print("All Features:")
    for i, feature in enumerate(all_feature_names, 1):
        print(f"  {i:2d}. {feature}")
    print()
    
    # Store advanced dataset globally
    global advanced_dataset
    advanced_dataset = df
    
    return df

def demonstrate_feature_selection():
    """Demonstrate feature selection methods using real data."""
    print("Feature Selection Methods:")
    print("-" * 40)
    
    if 'advanced_dataset' not in globals() or advanced_dataset is None:
        print("❌ No advanced dataset available for feature selection")
        return
    
    df = advanced_dataset.copy()
    
    # Prepare data for feature selection
    # Remove non-numerical features
    numerical_features = df.select_dtypes(include=[np.number]).columns
    numerical_features = [col for col in numerical_features if col != 'target']
    
    X = df[numerical_features].values
    y = df['target'].values
    
    print(f"Feature selection on {len(numerical_features)} numerical features")
    print(f"Dataset shape: {X.shape}")
    print(f"Target classes: {len(np.unique(y))}")
    print()
    
    # 1. Univariate Feature Selection
    print("1. UNIVARIATE FEATURE SELECTION:")
    print("-" * 35)
    
    # F-test for classification
    f_selector = SelectKBest(score_func=f_classif, k=10)
    X_f_selected = f_selector.fit_transform(X, y)
    
    # Get selected feature indices and scores
    selected_features_f = f_selector.get_support()
    feature_scores_f = f_selector.scores_
    
    print(f"✅ F-test selected {X_f_selected.shape[1]} features")
    print("Top 10 features by F-score:")
    
    # Create feature score DataFrame
    feature_scores_df = pd.DataFrame({
        'Feature': numerical_features,
        'F_Score': feature_scores_f,
        'Selected': selected_features_f
    }).sort_values('F_Score', ascending=False)
    
    for i, (_, row) in enumerate(feature_scores_df.head(10).iterrows()):
        status = "✓" if row['Selected'] else "✗"
        print(f"  {i+1:2d}. {status} {row['Feature']:25s}: {row['F_Score']:.3f}")
    print()
    
    # 2. Recursive Feature Elimination (RFE)
    print("2. RECURSIVE FEATURE ELIMINATION (RFE):")
    print("-" * 45)
    
    # Use Random Forest for RFE
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rfe = RFE(estimator=rf_classifier, n_features_to_select=10)
    X_rfe_selected = rfe.fit_transform(X, y)
    
    # Get selected features
    selected_features_rfe = rfe.get_support()
    feature_ranking_rfe = rfe.ranking_
    
    print(f"✅ RFE selected {X_rfe_selected.shape[1]} features")
    print("Top 10 features by RFE ranking:")
    
    # Create feature ranking DataFrame
    feature_ranking_df = pd.DataFrame({
        'Feature': numerical_features,
        'Ranking': feature_ranking_rfe,
        'Selected': selected_features_rfe
    }).sort_values('Ranking')
    
    for i, (_, row) in enumerate(feature_ranking_df.head(10).iterrows()):
        status = "✓" if row['Selected'] else "✗"
        print(f"  {i+1:2d}. {status} {row['Feature']:25s}: Rank {row['Ranking']}")
    print()
    
    # 3. Feature Selection with Lasso
    print("3. FEATURE SELECTION WITH LASSO:")
    print("-" * 35)
    
    # Use Lasso for feature selection
    lasso = Lasso(alpha=0.01, random_state=42)
    lasso.fit(X, y)
    
    # Get feature importance (absolute coefficients)
    feature_importance_lasso = np.abs(lasso.coef_)
    
    print(f"✅ Lasso feature selection completed")
    print("Top 10 features by Lasso importance:")
    
    # Create Lasso importance DataFrame
    lasso_importance_df = pd.DataFrame({
        'Feature': numerical_features,
        'Lasso_Importance': feature_importance_lasso
    }).sort_values('Lasso_Importance', ascending=False)
    
    for i, (_, row) in enumerate(lasso_importance_df.head(10).iterrows()):
        print(f"  {i+1:2d}. {row['Feature']:25s}: {row['Lasso_Importance']:.6f}")
    print()
    
    # 4. Feature Selection with Random Forest
    print("4. FEATURE SELECTION WITH RANDOM FOREST:")
    print("-" * 40)
    
    # Use Random Forest for feature importance
    rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_selector.fit(X, y)
    
    # Get feature importance
    feature_importance_rf = rf_selector.feature_importances_
    
    print(f"✅ Random Forest feature selection completed")
    print("Top 10 features by Random Forest importance:")
    
    # Create RF importance DataFrame
    rf_importance_df = pd.DataFrame({
        'Feature': numerical_features,
        'RF_Importance': feature_importance_rf
    }).sort_values('RF_Importance', ascending=False)
    
    for i, (_, row) in enumerate(rf_importance_df.head(10).iterrows()):
        print(f"  {i+1:2d}. {row['Feature']:25s}: {row['RF_Importance']:.6f}")
    print()
    
    # 5. Compare Feature Selection Methods
    print("5. COMPARING FEATURE SELECTION METHODS:")
    print("-" * 40)
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame({
        'Feature': numerical_features,
        'F_Score': feature_scores_f,
        'RFE_Ranking': feature_ranking_rfe,
        'Lasso_Importance': feature_importance_lasso,
        'RF_Importance': feature_importance_rf
    })
    
    # Normalize scores for comparison
    comparison_df['F_Score_Norm'] = comparison_df['F_Score'] / comparison_df['F_Score'].max()
    comparison_df['Lasso_Importance_Norm'] = comparison_df['Lasso_Importance'] / comparison_df['Lasso_Importance'].max()
    comparison_df['RF_Importance_Norm'] = comparison_df['RF_Importance'] / comparison_df['RF_Importance'].max()
    
    # Calculate average ranking
    comparison_df['Avg_Ranking'] = (
        comparison_df['F_Score_Norm'].rank(ascending=False) +
        comparison_df['RFE_Ranking'] +
        comparison_df['Lasso_Importance_Norm'].rank(ascending=False) +
        comparison_df['RF_Importance_Norm'].rank(ascending=False)
    ) / 4
    
    # Sort by average ranking
    comparison_df = comparison_df.sort_values('Avg_Ranking')
    
    print("Top 10 features by average ranking across all methods:")
    for i, (_, row) in enumerate(comparison_df.head(10).iterrows()):
        print(f"  {i+1:2d}. {row['Feature']:25s}: Avg Rank {row['Avg_Ranking']:.2f}")
    print()
    
    # Store selected features globally
    global selected_features
    selected_features = comparison_df.head(10)['Feature'].tolist()
    
    return selected_features

def demonstrate_dimensionality_reduction():
    """Demonstrate dimensionality reduction techniques using real data."""
    print("Dimensionality Reduction:")
    print("-" * 40)
    
    if 'advanced_dataset' not in globals() or advanced_dataset is None:
        print("❌ No advanced dataset available for dimensionality reduction")
        return
    
    df = advanced_dataset.copy()
    
    # Prepare data for dimensionality reduction
    numerical_features = df.select_dtypes(include=[np.number]).columns
    numerical_features = [col for col in numerical_features if col != 'target']
    
    X = df[numerical_features].values
    y = df['target'].values
    
    print(f"Dimensionality reduction on {len(numerical_features)} features")
    print(f"Original dataset shape: {X.shape}")
    print()
    
    # 1. Principal Component Analysis (PCA)
    print("1. PRINCIPAL COMPONENT ANALYSIS (PCA):")
    print("-" * 45)
    
    # Standardize the data first
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Get explained variance
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    print(f"✅ PCA completed")
    print(f"  Original features: {X.shape[1]}")
    print(f"  PCA components: {X_pca.shape[1]}")
    print()
    
    print("Explained variance by components:")
    for i, (var_ratio, cum_var_ratio) in enumerate(zip(explained_variance_ratio, cumulative_variance_ratio)):
        print(f"  Component {i+1:2d}: {var_ratio:.3f} ({var_ratio*100:.1f}%) - Cumulative: {cum_var_ratio:.3f} ({cum_var_ratio*100:.1f}%)")
        if cum_var_ratio >= 0.95:
            print(f"    → 95% variance explained with {i+1} components")
            break
    print()
    
    # 2. Feature Importance in PCA
    print("2. FEATURE IMPORTANCE IN PCA:")
    print("-" * 35)
    
    # Get feature importance (absolute loadings) for first 3 components
    print("Top 5 features contributing to first 3 principal components:")
    
    for comp_idx in range(min(3, len(pca.components_))):
        print(f"\nPrincipal Component {comp_idx + 1}:")
        
        # Get feature loadings for this component
        loadings = np.abs(pca.components_[comp_idx])
        feature_importance = pd.DataFrame({
            'Feature': numerical_features,
            'Loading': loadings
        }).sort_values('Loading', ascending=False)
        
        for i, (_, row) in enumerate(feature_importance.head(5).iterrows()):
            print(f"  {i+1:2d}. {row['Feature']:25s}: {row['Loading']:.3f}")
    print()
    
    # 3. Dimensionality Reduction with Different Numbers of Components
    print("3. DIMENSIONALITY REDUCTION WITH DIFFERENT COMPONENTS:")
    print("-" * 55)
    
    # Test different numbers of components
    n_components_list = [2, 3, 5, 10, 15, 20]
    
    print("Variance explained with different numbers of components:")
    for n_comp in n_components_list:
        if n_comp <= len(explained_variance_ratio):
            var_explained = cumulative_variance_ratio[n_comp - 1]
            print(f"  {n_comp:2d} components: {var_explained:.3f} ({var_explained*100:.1f}%)")
    print()
    
    # 4. Create dimensionality reduction visualization
    print("4. CREATING DIMENSIONALITY REDUCTION VISUALIZATION:")
    print("-" * 50)
    
    try:
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Dimensionality Reduction Results - Real Iris Data', fontsize=16, fontweight='bold')
        
        # 1. Explained variance plot
        n_components = min(20, len(explained_variance_ratio))
        axes[0, 0].plot(range(1, n_components + 1), explained_variance_ratio[:n_components], 'bo-', linewidth=2, markersize=6)
        axes[0, 0].set_title('Explained Variance by Component')
        axes[0, 0].set_xlabel('Principal Component')
        axes[0, 0].set_ylabel('Explained Variance Ratio')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Cumulative explained variance
        axes[0, 1].plot(range(1, n_components + 1), cumulative_variance_ratio[:n_components], 'ro-', linewidth=2, markersize=6)
        axes[0, 1].axhline(y=0.95, color='green', linestyle='--', label='95% Threshold')
        axes[0, 1].set_title('Cumulative Explained Variance')
        axes[0, 1].set_xlabel('Number of Components')
        axes[0, 1].set_ylabel('Cumulative Explained Variance Ratio')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. First two principal components scatter plot
        if X_pca.shape[1] >= 2:
            scatter = axes[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7)
            axes[1, 0].set_title('First Two Principal Components')
            axes[1, 0].set_xlabel('Principal Component 1')
            axes[1, 0].set_ylabel('Principal Component 2')
            axes[1, 0].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[1, 0], label='Target Class')
        
        # 4. Feature importance heatmap (first 5 components)
        n_comp_heatmap = min(5, len(pca.components_))
        if n_comp_heatmap > 0:
            # Get top 10 features for heatmap
            top_features_idx = np.argsort(np.sum(np.abs(pca.components_[:n_comp_heatmap]), axis=0))[-10:]
            top_features = [numerical_features[i] for i in top_features_idx]
            
            # Create heatmap data
            heatmap_data = pca.components_[:n_comp_heatmap, top_features_idx]
            
            sns.heatmap(heatmap_data, 
                       xticklabels=top_features, 
                       yticklabels=[f'PC{i+1}' for i in range(n_comp_heatmap)],
                       cmap='coolwarm', center=0, 
                       ax=axes[1, 1], 
                       cbar_kws={'label': 'Component Loading'})
            axes[1, 1].set_title('Feature Loadings in Top 5 Components')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save the visualization
        output_file = "feature_engineering_selection.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"  ✅ Visualization saved: {output_file}")
        
        plt.show()
        
    except Exception as e:
        print(f"  ❌ Error creating visualization: {e}")
    
    print()
    
    # 5. Summary
    print("5. DIMENSIONALITY REDUCTION SUMMARY:")
    print("-" * 40)
    
    print(f"✅ PCA successfully reduced {X.shape[1]} features to {X_pca.shape[1]} components")
    print(f"✅ 95% variance explained with {np.argmax(cumulative_variance_ratio >= 0.95) + 1} components")
    print(f"✅ Top contributing features identified for each principal component")
    print(f"✅ Visualization created showing explained variance and component relationships")
    print()
    
    return pca, X_pca

if __name__ == "__main__":
    main()
