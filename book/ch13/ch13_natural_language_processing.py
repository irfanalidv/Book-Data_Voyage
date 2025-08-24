#!/usr/bin/env python3
"""
Chapter 13: Natural Language Processing
Data Voyage: Processing and Understanding Human Language

This script covers essential NLP concepts and techniques.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore")


def main():
    print("=" * 80)
    print("CHAPTER 13: NATURAL LANGUAGE PROCESSING")
    print("=" * 80)
    print()

    # Section 13.1: NLP Overview
    print("13.1 NLP OVERVIEW")
    print("-" * 30)
    demonstrate_nlp_overview()

    # Section 13.2: Text Preprocessing
    print("\n13.2 TEXT PREPROCESSING")
    print("-" * 35)
    demonstrate_text_preprocessing()

    # Section 13.3: Text Representation
    print("\n13.3 TEXT REPRESENTATION")
    print("-" * 35)
    demonstrate_text_representation()

    # Section 13.4: NLP Applications
    print("\n13.4 NLP APPLICATIONS")
    print("-" * 35)
    demonstrate_nlp_applications()

    print("\n" + "=" * 80)
    print("CHAPTER SUMMARY")
    print("=" * 80)
    print("✅ NLP overview and applications")
    print("✅ Text preprocessing techniques")
    print("✅ Text representation methods")
    print("✅ Practical NLP applications")
    print()
    print("Next: Chapter 14 - Computer Vision Fundamentals")
    print("=" * 80)


def demonstrate_nlp_overview():
    """Demonstrate NLP overview and concepts."""
    print("Natural Language Processing Overview:")
    print("-" * 40)

    print("Natural Language Processing (NLP) is a branch of artificial")
    print("intelligence that helps computers understand, interpret, and")
    print("manipulate human language.")
    print()

    # 1. What is NLP?
    print("1. WHAT IS NLP?")
    print("-" * 20)

    nlp_concepts = {
        "Definition": "Field of AI focused on human language understanding",
        "Goal": "Enable computers to process and analyze text/speech",
        "Challenges": "Ambiguity, context, sarcasm, multiple languages",
        "Applications": "Chatbots, translation, sentiment analysis, search engines",
    }

    for concept, description in nlp_concepts.items():
        print(f"  {concept}: {description}")
    print()

    # 2. NLP Tasks
    print("2. NLP TASKS:")
    print("-" * 20)

    nlp_tasks = {
        "Text Classification": [
            "Sentiment analysis",
            "Topic classification",
            "Spam detection",
        ],
        "Information Extraction": [
            "Named Entity Recognition (NER)",
            "Relation extraction",
            "Event extraction",
        ],
        "Text Generation": [
            "Machine translation",
            "Text summarization",
            "Question answering",
        ],
        "Language Understanding": [
            "Part-of-speech tagging",
            "Dependency parsing",
            "Semantic analysis",
        ],
    }

    for task, examples in nlp_tasks.items():
        print(f"  {task}:")
        for example in examples:
            print(f"    • {example}")
        print()

    # 3. NLP Pipeline
    print("3. NLP PIPELINE:")
    print("-" * 25)

    pipeline_steps = [
        "1. Text Collection - Gather raw text data",
        "2. Text Preprocessing - Clean and normalize text",
        "3. Text Representation - Convert text to numerical format",
        "4. Feature Engineering - Extract relevant features",
        "5. Model Training - Train NLP models",
        "6. Evaluation - Assess model performance",
        "7. Deployment - Use models in production",
    ]

    for step in pipeline_steps:
        print(f"  {step}")
    print()

    # 4. NLP Challenges
    print("4. NLP CHALLENGES:")
    print("-" * 25)

    challenges = {
        "Ambiguity": "Words with multiple meanings (bank, bark, fair)",
        "Context": "Understanding meaning based on surrounding text",
        "Sarcasm/Irony": "Detecting non-literal language use",
        "Slang/Informal": "Handling casual language and abbreviations",
        "Multilingual": "Processing text in different languages",
        "Domain Specific": "Specialized terminology in different fields",
    }

    for challenge, description in challenges.items():
        print(f"  {challenge}: {description}")
    print()


def demonstrate_text_preprocessing():
    """Demonstrate text preprocessing techniques."""
    print("Text Preprocessing:")
    print("-" * 40)

    print("Text preprocessing is crucial for converting raw text")
    print("into a format suitable for NLP models.")
    print()

    # 1. Sample Text Data
    print("1. SAMPLE TEXT DATA:")
    print("-" * 25)

    sample_texts = [
        "Hello! How are you doing today? 😊",
        "The quick brown fox jumps over the lazy dog.",
        "I can't believe it's already 2024! #amazing #year",
        "RT @user: This is a retweet with some @mentions and #hashtags",
        "Text with numbers 123 and symbols @#$%^&*()",
        "   Multiple    spaces    and    tabs\t\t\there   ",
    ]

    print("Original texts:")
    for i, text in enumerate(sample_texts, 1):
        print(f"  {i}. {repr(text)}")
    print()

    # 2. Text Cleaning Functions
    print("2. TEXT CLEANING FUNCTIONS:")
    print("-" * 30)

    def clean_text(text):
        """Clean and normalize text."""
        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)

        # Remove user mentions and hashtags
        text = re.sub(r"@\w+|#\w+", "", text)

        # Remove numbers
        text = re.sub(r"\d+", "", text)

        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Remove extra whitespace
        text = " ".join(text.split())

        return text

    def remove_stopwords(text, stopwords):
        """Remove common stopwords."""
        words = text.split()
        filtered_words = [word for word in words if word.lower() not in stopwords]
        return " ".join(filtered_words)

    def tokenize_text(text):
        """Simple tokenization by splitting on whitespace."""
        return text.split()

    # 3. Apply Preprocessing
    print("3. APPLY PREPROCESSING:")
    print("-" * 25)

    # Define common stopwords
    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "this",
        "that",
        "these",
        "those",
    }

    print("Preprocessed texts:")
    for i, text in enumerate(sample_texts, 1):
        cleaned = clean_text(text)
        no_stopwords = remove_stopwords(cleaned, stopwords)
        tokens = tokenize_text(no_stopwords)

        print(f"  {i}. Original: {repr(text)}")
        print(f"     Cleaned: {repr(cleaned)}")
        print(f"     No stopwords: {repr(no_stopwords)}")
        print(f"     Tokens: {tokens}")
        print()

    # 4. Advanced Preprocessing
    print("4. ADVANCED PREPROCESSING:")
    print("-" * 30)

    def lemmatize_text(text):
        """Simple lemmatization (basic word form reduction)."""
        # This is a simplified version - real lemmatization uses linguistic rules
        lemmatization_rules = {
            "running": "run",
            "runs": "run",
            "ran": "run",
            "jumping": "jump",
            "jumps": "jump",
            "jumped": "jump",
            "quickly": "quick",
            "slowly": "slow",
            "better": "good",
            "best": "good",
            "worse": "bad",
            "worst": "bad",
        }

        words = text.split()
        lemmatized = [lemmatization_rules.get(word, word) for word in words]
        return " ".join(lemmatized)

    def extract_features(text):
        """Extract basic text features."""
        words = text.split()
        return {
            "word_count": len(words),
            "char_count": len(text),
            "avg_word_length": np.mean([len(word) for word in words]) if words else 0,
            "unique_words": len(set(words)),
            "vocabulary_richness": len(set(words)) / len(words) if words else 0,
        }

    # Apply advanced preprocessing
    print("Advanced preprocessing example:")
    sample_text = "The quick brown foxes are running quickly through the forest."
    cleaned_sample = clean_text(sample_text)
    lemmatized_sample = lemmatize_text(cleaned_sample)
    features = extract_features(lemmatized_sample)

    print(f"  Original: {sample_text}")
    print(f"  Cleaned: {cleaned_sample}")
    print(f"  Lemmatized: {lemmatized_sample}")
    print(f"  Features: {features}")
    print()

    # 5. Preprocessing Visualization
    print("5. PREPROCESSING VISUALIZATION:")
    print("-" * 35)

    # Create sample dataset for visualization
    sample_dataset = [
        "Hello world! How are you? 😊",
        "The quick brown fox jumps over the lazy dog.",
        "I love machine learning and AI! #awesome #tech",
        "This is a sample text with some numbers 123 and symbols @#$%",
        "Natural language processing is fascinating and complex.",
    ]

    # Apply preprocessing and collect statistics
    preprocessing_stats = []
    for text in sample_dataset:
        original_length = len(text)
        cleaned = clean_text(text)
        cleaned_length = len(cleaned)
        no_stopwords = remove_stopwords(cleaned, stopwords)
        final_length = len(no_stopwords)

        preprocessing_stats.append(
            {
                "original": original_length,
                "cleaned": cleaned_length,
                "no_stopwords": final_length,
            }
        )

    # Create visualization
    plt.figure(figsize=(15, 5))

    # Text length comparison
    plt.subplot(1, 3, 1)
    x = range(len(sample_dataset))
    original_lengths = [stats["original"] for stats in preprocessing_stats]
    cleaned_lengths = [stats["cleaned"] for stats in preprocessing_stats]
    final_lengths = [stats["no_stopwords"] for stats in preprocessing_stats]

    plt.bar(
        [i - 0.2 for i in x],
        original_lengths,
        width=0.2,
        label="Original",
        color="skyblue",
    )
    plt.bar(x, cleaned_lengths, width=0.2, label="Cleaned", color="lightgreen")
    plt.bar(
        [i + 0.2 for i in x],
        final_lengths,
        width=0.2,
        label="No Stopwords",
        color="lightcoral",
    )

    plt.xlabel("Text Sample")
    plt.ylabel("Length (characters)")
    plt.title("Text Length After Preprocessing Steps")
    plt.legend()
    plt.xticks(x, [f"Text {i+1}" for i in x])

    # Preprocessing impact
    plt.subplot(1, 3, 2)
    reduction_percentages = [
        (1 - stats["cleaned"] / stats["original"]) * 100
        for stats in preprocessing_stats
    ]
    plt.bar(range(len(sample_dataset)), reduction_percentages, color="orange")
    plt.xlabel("Text Sample")
    plt.ylabel("Reduction (%)")
    plt.title("Text Reduction After Cleaning")
    plt.xticks(
        range(len(sample_dataset)), [f"Text {i+1}" for i in range(len(sample_dataset))]
    )

    # Feature distribution
    plt.subplot(1, 3, 3)
    feature_names = ["word_count", "avg_word_length", "vocabulary_richness"]
    feature_values = []

    for text in sample_dataset:
        cleaned = clean_text(text)
        features = extract_features(cleaned)
        feature_values.append([features[name] for name in feature_names])

    feature_values = np.array(feature_values)

    for i, feature in enumerate(feature_names):
        plt.plot(
            range(len(sample_dataset)),
            feature_values[:, i],
            marker="o",
            label=feature.replace("_", " ").title(),
        )

    plt.xlabel("Text Sample")
    plt.ylabel("Feature Value")
    plt.title("Text Features Distribution")
    plt.legend()
    plt.xticks(
        range(len(sample_dataset)), [f"Text {i+1}" for i in range(len(sample_dataset))]
    )

    plt.tight_layout()
    plt.savefig("text_preprocessing.png", dpi=300, bbox_inches="tight")
    print("✅ Text preprocessing visualization saved as 'text_preprocessing.png'")
    plt.close()


def demonstrate_text_representation():
    """Demonstrate text representation methods."""
    print("Text Representation:")
    print("-" * 40)

    print("Text representation converts text into numerical format")
    print("that machine learning models can process.")
    print()

    # 1. Sample Documents
    print("1. SAMPLE DOCUMENTS:")
    print("-" * 25)

    documents = [
        "machine learning is a subset of artificial intelligence",
        "deep learning uses neural networks with multiple layers",
        "natural language processing helps computers understand text",
        "computer vision processes and analyzes images",
        "data science combines statistics programming and domain knowledge",
    ]

    print("Document collection:")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc}")
    print()

    # 2. Bag of Words (BoW)
    print("2. BAG OF WORDS (BOW):")
    print("-" * 25)

    # Create BoW representation
    vectorizer = CountVectorizer(lowercase=True, stop_words="english")
    bow_matrix = vectorizer.fit_transform(documents)

    print("BoW Matrix:")
    print(f"Shape: {bow_matrix.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    print()

    # Show vocabulary
    print("Vocabulary:")
    for word, index in sorted(vectorizer.vocabulary_.items()):
        print(f"  {word}: {index}")
    print()

    # Show BoW matrix
    print("BoW Matrix (dense format):")
    bow_df = pd.DataFrame(
        bow_matrix.toarray(),
        columns=vectorizer.get_feature_names_out(),
        index=[f"Doc {i+1}" for i in range(len(documents))],
    )
    print(bow_df)
    print()

    # 3. TF-IDF Representation
    print("3. TF-IDF REPRESENTATION:")
    print("-" * 30)

    # Create TF-IDF representation
    tfidf_vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    print("TF-IDF Matrix:")
    print(f"Shape: {tfidf_matrix.shape}")
    print()

    # Show TF-IDF matrix
    print("TF-IDF Matrix (dense format):")
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=tfidf_vectorizer.get_feature_names_out(),
        index=[f"Doc {i+1}" for i in range(len(documents))],
    )
    print(tfidf_df.round(3))
    print()

    # 4. Document Similarity
    print("4. DOCUMENT SIMILARITY:")
    print("-" * 30)

    # Calculate cosine similarity between documents
    from sklearn.metrics.pairwise import cosine_similarity

    # BoW similarity
    bow_similarity = cosine_similarity(bow_matrix)
    print("BoW Cosine Similarity Matrix:")
    bow_sim_df = pd.DataFrame(
        bow_similarity,
        index=[f"Doc {i+1}" for i in range(len(documents))],
        columns=[f"Doc {i+1}" for i in range(len(documents))],
    )
    print(bow_sim_df.round(3))
    print()

    # TF-IDF similarity
    tfidf_similarity = cosine_similarity(tfidf_matrix)
    print("TF-IDF Cosine Similarity Matrix:")
    tfidf_sim_df = pd.DataFrame(
        tfidf_similarity,
        index=[f"Doc {i+1}" for i in range(len(documents))],
        columns=[f"Doc {i+1}" for i in range(len(documents))],
    )
    print(tfidf_sim_df.round(3))
    print()

    # 5. Visualization
    print("5. VISUALIZATION:")
    print("-" * 20)

    plt.figure(figsize=(15, 5))

    # BoW heatmap
    plt.subplot(1, 3, 1)
    sns.heatmap(bow_df, annot=True, cmap="Blues", fmt="d")
    plt.title("Bag of Words Matrix")
    plt.xlabel("Terms")
    plt.ylabel("Documents")

    # TF-IDF heatmap
    plt.subplot(1, 3, 2)
    sns.heatmap(tfidf_df, annot=True, cmap="Greens", fmt=".2f")
    plt.title("TF-IDF Matrix")
    plt.xlabel("Terms")
    plt.ylabel("Documents")

    # Similarity heatmap
    plt.subplot(1, 3, 3)
    sns.heatmap(
        tfidf_sim_df,
        annot=True,
        cmap="Reds",
        fmt=".3f",
        xticklabels=True,
        yticklabels=True,
    )
    plt.title("Document Similarity (TF-IDF)")
    plt.xlabel("Documents")
    plt.ylabel("Documents")

    plt.tight_layout()
    plt.savefig("text_representation.png", dpi=300, bbox_inches="tight")
    print("✅ Text representation visualization saved as 'text_representation.png'")
    plt.close()


def demonstrate_nlp_applications():
    """Demonstrate practical NLP applications."""
    print("NLP Applications:")
    print("-" * 40)

    print("Let's implement some practical NLP applications")
    print("to demonstrate real-world usage.")
    print()

    # 1. Sentiment Analysis
    print("1. SENTIMENT ANALYSIS:")
    print("-" * 30)

    # Create a larger and more balanced sentiment dataset
    sentiment_data = [
        # Positive sentiments
        "I love this product! It's amazing and works perfectly.",
        "Absolutely fantastic! Exceeded all my expectations.",
        "Great value for money. Highly recommend to everyone.",
        "Excellent customer support and fast delivery.",
        "Wonderful experience! Will definitely buy again.",
        "This is the best purchase I've ever made. Outstanding quality!",
        "Amazing product with incredible features. Love it!",
        "Superb service and excellent product quality.",
        "Fantastic experience from start to finish.",
        "Outstanding value and exceptional performance.",
        "Brilliant product that delivers on all promises.",
        "Exceptional quality and great customer service.",
        "Perfect product that exceeded my expectations.",
        "Incredible value and amazing functionality.",
        "Superior quality and outstanding performance.",
        # Negative sentiments
        "This is the worst purchase I've ever made. Terrible quality.",
        "Disappointed with the service. Very slow and unhelpful.",
        "Not satisfied at all. Waste of money and time.",
        "The quality is poor and it broke after a week.",
        "Terrible product that doesn't work at all.",
        "Awful customer service and defective product.",
        "Complete waste of money. Very disappointed.",
        "Poor quality and terrible user experience.",
        "Horrible product that failed immediately.",
        "Bad service and inferior product quality.",
        "Useless product that doesn't function properly.",
        "Disgusting quality and terrible support.",
        "Worthless purchase with no value.",
        "Terrible experience and poor performance.",
        "Bad product that broke on first use.",
    ]

    sentiment_labels = [1] * 15 + [0] * 15  # 1: positive, 0: negative

    print("Sentiment Analysis Dataset:")
    print(f"Total samples: {len(sentiment_data)}")
    print(f"Positive samples: {sum(sentiment_labels)}")
    print(f"Negative samples: {len(sentiment_labels) - sum(sentiment_labels)}")
    print()

    print("Sample entries:")
    for i, (text, label) in enumerate(zip(sentiment_data[:5], sentiment_labels[:5])):
        sentiment = "Positive" if label == 1 else "Negative"
        print(f"  {i+1}. [{sentiment}] {text}")
    print("  ... (showing first 5 of 30 total)")
    print()

    # 2. Text Classification Pipeline
    print("2. TEXT CLASSIFICATION PIPELINE:")
    print("-" * 35)

    # Create and train the pipeline with better parameters
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    max_features=200,
                    ngram_range=(1, 2),  # Include bigrams
                    min_df=2,  # Minimum document frequency
                    max_df=0.8,  # Maximum document frequency
                ),
            ),
            (
                "classifier",
                MultinomialNB(alpha=0.1),
            ),  # Adjust alpha for better performance
        ]
    )

    # Split data with more reasonable split
    X_train, X_test, y_train, y_test = train_test_split(
        sentiment_data,
        sentiment_labels,
        test_size=0.3,
        random_state=42,
        stratify=sentiment_labels,
    )

    # Train the model
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    print("Training Results:")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print()

    # 3. Model Evaluation
    print("3. MODEL EVALUATION:")
    print("-" * 25)

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Negative", "Positive"]))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    print("Confusion Matrix:")
    print("                Predicted")
    print("                Negative  Positive")
    print(f"Actual Negative    {cm[0,0]:>8}    {cm[0,1]:>8}")
    print(f"      Positive     {cm[1,0]:>8}    {cm[1,1]:>8}")
    print()

    # 4. Feature Importance
    print("4. FEATURE IMPORTANCE:")
    print("-" * 25)

    # Get feature names and importance
    feature_names = pipeline.named_steps["tfidf"].get_feature_names_out()
    feature_importance = (
        pipeline.named_steps["classifier"].feature_log_prob_[1]
        - pipeline.named_steps["classifier"].feature_log_prob_[0]
    )

    # Create feature importance DataFrame
    feature_df = pd.DataFrame(
        {"feature": feature_names, "importance": feature_importance}
    ).sort_values("importance", ascending=False)

    print("Top 10 Most Important Features:")
    for i, (_, row) in enumerate(feature_df.head(10).iterrows()):
        sentiment = "Positive" if row["importance"] > 0 else "Negative"
        print(f"  {i+1:2d}. {row['feature']:<20} ({sentiment})")
    print()

    print("Bottom 10 Least Important Features:")
    for i, (_, row) in enumerate(feature_df.tail(10).iterrows()):
        sentiment = "Positive" if row["importance"] > 0 else "Negative"
        print(f"  {i+1:2d}. {row['feature']:<20} ({sentiment})")
    print()

    # 5. New Text Classification
    print("5. NEW TEXT CLASSIFICATION:")
    print("-" * 30)

    new_texts = [
        "This product is absolutely wonderful and amazing!",
        "I hate this terrible product, it's awful.",
        "The service was okay, nothing special.",
        "Fantastic quality and excellent customer support!",
        "This is the worst experience ever.",
        "Outstanding performance and great value.",
        "Terrible quality and poor service.",
        "Amazing features and incredible results.",
    ]

    print("Predicting sentiment for new texts:")
    predictions = pipeline.predict(new_texts)
    probabilities = pipeline.predict_proba(new_texts)

    for i, (text, pred, prob) in enumerate(zip(new_texts, predictions, probabilities)):
        sentiment = "Positive" if pred == 1 else "Negative"
        confidence = max(prob)
        print(f"  {i+1}. Text: {text}")
        print(f"     Prediction: {sentiment} (confidence: {confidence:.3f})")
        print()

    # 6. Model Performance Analysis
    print("6. MODEL PERFORMANCE ANALYSIS:")
    print("-" * 35)

    # Cross-validation score
    from sklearn.model_selection import cross_val_score

    cv_scores = cross_val_score(pipeline, sentiment_data, sentiment_labels, cv=5)

    print("Cross-validation Results:")
    print(f"  CV Scores: {cv_scores}")
    print(f"  Mean CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    print()

    # Feature analysis
    print("Feature Analysis:")
    print(f"  Total features: {len(feature_names)}")
    print(f"  Positive features: {sum(feature_importance > 0)}")
    print(f"  Negative features: {sum(feature_importance < 0)}")
    print(f"  Neutral features: {sum(feature_importance == 0)}")
    print()

    # 7. Visualization
    print("7. VISUALIZATION:")
    print("-" * 20)

    plt.figure(figsize=(15, 10))

    # Confusion Matrix
    plt.subplot(2, 3, 1)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Negative", "Positive"],
        yticklabels=["Negative", "Positive"],
    )
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")

    # Feature Importance
    plt.subplot(2, 3, 2)
    top_features = feature_df.head(15)
    colors = ["green" if x > 0 else "red" for x in top_features["importance"]]
    plt.barh(range(len(top_features)), top_features["importance"], color=colors)
    plt.yticks(range(len(top_features)), top_features["feature"])
    plt.xlabel("Feature Importance")
    plt.title("Top 15 Feature Importance")

    # Prediction Confidence
    plt.subplot(2, 3, 3)
    max_probs = [max(prob) for prob in probabilities]
    colors = ["green" if pred == 1 else "red" for pred in predictions]
    plt.bar(range(len(new_texts)), max_probs, color=colors)
    plt.xlabel("Text Sample")
    plt.ylabel("Prediction Confidence")
    plt.title("Prediction Confidence for New Texts")
    plt.ylim(0, 1)

    # CV Scores
    plt.subplot(2, 3, 4)
    plt.bar(range(len(cv_scores)), cv_scores, color="skyblue")
    plt.xlabel("Fold")
    plt.ylabel("Accuracy")
    plt.title("Cross-Validation Scores")
    plt.ylim(0, 1)

    # Feature Distribution
    plt.subplot(2, 3, 5)
    plt.hist(feature_importance, bins=30, alpha=0.7, color="lightgreen")
    plt.xlabel("Feature Importance")
    plt.ylabel("Frequency")
    plt.title("Feature Importance Distribution")

    # Performance Metrics
    plt.subplot(2, 3, 6)
    metrics = ["Precision", "Recall", "F1-Score"]
    positive_scores = [
        cm[1, 1] / (cm[1, 1] + cm[0, 1]),
        cm[1, 1] / (cm[1, 1] + cm[1, 0]),
        2 * cm[1, 1] / (2 * cm[1, 1] + cm[0, 1] + cm[1, 0]),
    ]
    negative_scores = [
        cm[0, 0] / (cm[0, 0] + cm[1, 0]),
        cm[0, 0] / (cm[0, 0] + cm[0, 1]),
        2 * cm[0, 0] / (2 * cm[0, 0] + cm[1, 0] + cm[0, 1]),
    ]

    x = np.arange(len(metrics))
    width = 0.35

    plt.bar(
        x - width / 2,
        positive_scores,
        width,
        label="Positive",
        color="green",
        alpha=0.7,
    )
    plt.bar(
        x + width / 2, negative_scores, width, label="Negative", color="red", alpha=0.7
    )

    plt.xlabel("Metrics")
    plt.ylabel("Score")
    plt.title("Performance by Class")
    plt.xticks(x, metrics)
    plt.legend()
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig("nlp_applications.png", dpi=300, bbox_inches="tight")
    print("✅ NLP applications visualization saved as 'nlp_applications.png'")
    plt.close()

    print("NLP Applications Summary:")
    print("✅ Implemented improved sentiment analysis pipeline")
    print("✅ Created balanced dataset with 30 samples")
    print("✅ Enhanced model with better parameters and features")
    print("✅ Achieved better classification performance")
    print("✅ Analyzed feature importance and model performance")
    print("✅ Applied model to new texts with confidence scores")


if __name__ == "__main__":
    main()
