#!/usr/bin/env python3
"""
Chapter 6: Data Cleaning and Preprocessing
Data Voyage: Preparing Data for Analysis and Machine Learning

This script covers essential data cleaning and preprocessing concepts.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

def main():
    print("=" * 80)
    print("CHAPTER 6: DATA CLEANING AND PREPROCESSING")
    print("=" * 80)
    print()
    
    # Section 6.1: Data Quality Assessment
    print("6.1 DATA QUALITY ASSESSMENT")
    print("-" * 40)
    demonstrate_data_quality_assessment()
    
    # Section 6.2: Data Cleaning Techniques
    print("\n6.2 DATA CLEANING TECHNIQUES")
    print("-" * 40)
    demonstrate_data_cleaning()
    
    # Section 6.3: Data Preprocessing Methods
    print("\n6.3 DATA PREPROCESSING METHODS")
    print("-" * 40)
    demonstrate_data_preprocessing()
    
    print("\n" + "=" * 80)
    print("CHAPTER SUMMARY")
    print("=" * 80)
    print("✅ Data quality assessment - Identifying and measuring data issues")
    print("✅ Data cleaning techniques - Handling missing values and outliers")
    print("✅ Data preprocessing methods - Scaling, encoding, and transformation")
    print()
    print("Next: Chapter 7 - Exploratory Data Analysis")
    print("=" * 80)

def demonstrate_data_quality_assessment():
    """Demonstrate data quality assessment and analysis."""
    print("Data Quality Assessment and Analysis:")
    print("-" * 40)
    
    # Create sample dataset with various quality issues
    print("Creating sample dataset with quality issues...")
    
    np.random.seed(42)
    n_records = 500
    
    # Generate clean base data
    customer_ids = list(range(1, n_records + 1))
    ages = np.random.normal(35, 12, n_records)
    incomes = np.random.lognormal(10.5, 0.5, n_records)
    education_years = np.random.poisson(16, n_records)
    
    # Introduce quality issues
    data_with_issues = []
    for i in range(n_records):
        record = {
            'customer_id': customer_ids[i],
            'age': ages[i],
            'income': incomes[i],
            'education_years': education_years[i],
            'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Boston', 'Seattle']),
            'subscription_status': np.random.choice(['Active', 'Inactive', 'Pending'])
        }
        
        # Introduce missing values
        if np.random.random() < 0.08:
            record['age'] = np.nan
        if np.random.random() < 0.12:
            record['income'] = np.nan
        if np.random.random() < 0.05:
            record['education_years'] = np.nan
        
        # Introduce invalid values
        if np.random.random() < 0.02:
            record['age'] = np.random.choice([-5, 150, 999])
        if np.random.random() < 0.01:
            record['income'] = -np.random.exponential(1000)
        
        data_with_issues.append(record)
    
    # Convert to DataFrame
    df = pd.DataFrame(data_with_issues)
    print(f"✅ Created dataset with {len(df)} records")
    print(f"Dataset shape: {df.shape}")
    print()
    
    # Data Quality Assessment
    print("1. DATA QUALITY ASSESSMENT:")
    print("-" * 30)
    
    # Completeness analysis
    print("Completeness Analysis:")
    completeness = df.notna().mean() * 100
    for col, comp in completeness.items():
        print(f"  {col}: {comp:.1f}% complete")
    
    missing_summary = df.isna().sum()
    print(f"\nMissing values summary:")
    for col, missing in missing_summary.items():
        if missing > 0:
            print(f"  {col}: {missing} missing values")
    print()
    
    # Data type analysis
    print("Data Type Analysis:")
    print(f"  Data types: {df.dtypes}")
    print()
    
    # Value range analysis
    print("Value Range Analysis:")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'customer_id':
            col_data = df[col].dropna()
            if len(col_data) > 0:
                print(f"  {col}:")
                print(f"    Range: {col_data.min():.2f} to {col_data.max():.2f}")
                print(f"    Mean: {col_data.mean():.2f}")
                print(f"    Std: {col_data.std():.2f}")
    print()
    
    # Identify outliers using IQR method
    print("Outlier Detection (IQR Method):")
    for col in numeric_cols:
        if col != 'customer_id':
            col_data = df[col].dropna()
            if len(col_data) > 0:
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                print(f"  {col}: {len(outliers)} outliers ({len(outliers)/len(col_data)*100:.1f}%)")
    print()
    
    # Store dataset for later use
    global original_df
    original_df = df.copy()

def demonstrate_data_cleaning():
    """Demonstrate data cleaning techniques."""
    print("Data Cleaning Techniques and Methods:")
    print("-" * 40)
    
    # Start with the original dataset
    df = original_df.copy()
    print(f"Starting data cleaning with dataset: {df.shape}")
    print()
    
    # 1. Handling Missing Values
    print("1. HANDLING MISSING VALUES:")
    print("-" * 25)
    
    # Imputation for numeric data
    df_cleaned = df.copy()
    
    # Age imputation (median)
    age_imputer = SimpleImputer(strategy='median')
    df_cleaned['age'] = age_imputer.fit_transform(df_cleaned[['age']])
    
    # Income imputation (mean)
    income_imputer = SimpleImputer(strategy='mean')
    df_cleaned['income'] = income_imputer.fit_transform(df_cleaned[['income']])
    
    # Education imputation (mode)
    education_imputer = SimpleImputer(strategy='most_frequent')
    df_cleaned['education_years'] = education_imputer.fit_transform(df_cleaned[['education_years']])
    
    print(f"  Missing values after imputation: {df_cleaned.isna().sum().sum()}")
    print()
    
    # 2. Handling Invalid Values
    print("2. HANDLING INVALID VALUES:")
    print("-" * 25)
    
    # Clean age values
    median_age = df_cleaned['age'].median()
    df_cleaned.loc[df_cleaned['age'] < 0, 'age'] = median_age
    df_cleaned.loc[df_cleaned['age'] > 120, 'age'] = median_age
    
    # Clean income values
    median_income = df_cleaned['income'].median()
    df_cleaned.loc[df_cleaned['income'] < 0, 'income'] = median_income
    
    # Clean education values
    mode_education = df_cleaned['education_years'].mode()[0]
    df_cleaned.loc[df_cleaned['education_years'] < 0, 'education_years'] = mode_education
    df_cleaned.loc[df_cleaned['education_years'] > 25, 'education_years'] = mode_education
    
    print(f"  Invalid ages cleaned: {len(df_cleaned[(df_cleaned['age'] < 0) | (df_cleaned['age'] > 120)])}")
    print(f"  Negative incomes cleaned: {len(df_cleaned[df_cleaned['income'] < 0])}")
    print(f"  Invalid education years cleaned: {len(df_cleaned[(df_cleaned['education_years'] < 0) | (df_cleaned['education_years'] > 25)])}")
    print()
    
    # Store cleaned dataset
    global cleaned_df
    cleaned_df = df_cleaned.copy()

def demonstrate_data_preprocessing():
    """Demonstrate data preprocessing methods."""
    print("Data Preprocessing Methods and Techniques:")
    print("-" * 40)
    
    # Start with cleaned dataset
    df = cleaned_df.copy()
    print(f"Starting preprocessing with cleaned dataset: {df.shape}")
    print()
    
    # 1. Feature Scaling
    print("1. FEATURE SCALING:")
    print("-" * 20)
    
    # Select numeric features for scaling
    numeric_features = ['age', 'income', 'education_years']
    X_numeric = df[numeric_features].copy()
    
    print("Original numeric features:")
    print(X_numeric.describe().round(2))
    print()
    
    # Standardization (Z-score normalization)
    scaler_standard = StandardScaler()
    X_standardized = scaler_standard.fit_transform(X_numeric)
    X_standardized_df = pd.DataFrame(X_standardized, columns=numeric_features)
    
    print("Standardized features (Z-score):")
    print(X_standardized_df.describe().round(2))
    print()
    
    # 2. Categorical Encoding
    print("2. CATEGORICAL ENCODING:")
    print("-" * 25)
    
    # Select categorical features
    categorical_features = ['city', 'subscription_status']
    
    # Label encoding
    label_encoders = {}
    df_encoded = df.copy()
    
    for feature in categorical_features:
        le = LabelEncoder()
        df_encoded[f'{feature}_encoded'] = le.fit_transform(df[feature])
        label_encoders[feature] = le
        
        print(f"{feature} encoding:")
        for i, label in enumerate(le.classes_):
            print(f"  {label} -> {i}")
        print()
    
    # One-hot encoding for city
    city_dummies = pd.get_dummies(df['city'], prefix='city')
    df_encoded = pd.concat([df_encoded, city_dummies], axis=1)
    
    print("One-hot encoding for cities:")
    print(city_dummies.head())
    print()
    
    # 3. Feature Engineering
    print("3. FEATURE ENGINEERING:")
    print("-" * 20)
    
    # Age groups
    df_encoded['age_group'] = pd.cut(df_encoded['age'], bins=5, labels=['Very Young', 'Young', 'Middle', 'Senior', 'Elderly'])
    
    # Income categories
    df_encoded['income_category'] = pd.cut(df_encoded['income'], 
                                          bins=[0, 50000, 100000, 200000, float('inf')],
                                          labels=['Low', 'Medium', 'High', 'Very High'])
    
    # Interaction features
    df_encoded['age_income_ratio'] = df_encoded['age'] / df_encoded['income']
    df_encoded['education_income_ratio'] = df_encoded['education_years'] / df_encoded['income']
    
    print("Engineered features created:")
    print(f"  Age groups: {df_encoded['age_group'].nunique()} categories")
    print(f"  Income categories: {df_encoded['income_category'].nunique()} categories")
    print(f"  Age-Income ratio: {df_encoded['age_income_ratio'].nunique()} unique values")
    print(f"  Education-Income ratio: {df_encoded['education_income_ratio'].nunique()} unique values")
    print()
    
    # 4. Final Dataset Summary
    print("4. FINAL DATASET SUMMARY:")
    print("-" * 25)
    
    print(f"Final dataset shape: {df_encoded.shape}")
    print(f"Features: {len(df_encoded.columns)}")
    print(f"Records: {len(df_encoded)}")
    print(f"Missing values: {df_encoded.isna().sum().sum()}")
    print(f"Data types:")
    for dtype, count in df_encoded.dtypes.value_counts().items():
        print(f"  {dtype}: {count} features")
    
    # Create preprocessing visualization
    plt.figure(figsize=(15, 10))
    
    # Before and after scaling comparison
    plt.subplot(2, 3, 1)
    plt.hist(X_numeric['income'], bins=30, alpha=0.7, label='Original', color='skyblue')
    plt.hist(X_standardized_df['income'], bins=30, alpha=0.7, label='Standardized', color='lightcoral')
    plt.title('Income Distribution: Original vs Standardized')
    plt.xlabel('Income')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Categorical encoding
    plt.subplot(2, 3, 2)
    city_counts = df['city'].value_counts()
    plt.pie(city_counts.values, labels=city_counts.index, autopct='%1.1f%%')
    plt.title('City Distribution')
    
    # Age distribution
    plt.subplot(2, 3, 3)
    plt.hist(df_encoded['age'], bins=20, alpha=0.7, color='orange', edgecolor='black')
    plt.title('Age Distribution')
    plt.xlabel('Age')
    plt.ylabel('Frequency')
    
    # Income categories
    plt.subplot(2, 3, 4)
    df_encoded['income_category'].value_counts().plot(kind='bar', color='purple')
    plt.title('Income Categories')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Age groups
    plt.subplot(2, 3, 5)
    df_encoded['age_group'].value_counts().plot(kind='bar', color='lightgreen')
    plt.title('Age Groups')
    plt.xlabel('Group')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    # Feature correlations
    plt.subplot(2, 3, 6)
    numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns
    correlations = df_encoded[numeric_cols].corrwith(df_encoded['age']).abs().sort_values(ascending=False)
    correlations.head(10).plot(kind='bar', color='pink')
    plt.title('Top 10 Feature Correlations with Age')
    plt.xlabel('Features')
    plt.ylabel('Correlation')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('data_preprocessing.png', dpi=300, bbox_inches='tight')
    print("✅ Data preprocessing visualization saved as 'data_preprocessing.png'")
    plt.close()
    
    # Final summary
    print("\nDATA PREPROCESSING SUMMARY:")
    print("-" * 30)
    print(f"Original features: {len(original_df.columns)}")
    print(f"Features after cleaning: {len(cleaned_df.columns)}")
    print(f"Features after preprocessing: {len(df_encoded.columns)}")
    print()
    print("Data preprocessing process complete!")
    print("Dataset is now ready for machine learning and analysis.")

if __name__ == "__main__":
    main()
