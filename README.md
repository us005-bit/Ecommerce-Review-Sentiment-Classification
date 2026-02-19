# 📊 E-Commerce Review Sentiment Classification

## Overview

This project builds a production-ready sentiment classification system
for Amazon product reviews. The objective is to classify reviews into:

-   Negative
-   Neutral
-   Positive

The system combines classical machine learning baselines and
transformer-based deep learning (DistilBERT), and is deployed using a
Streamlit web interface.

------------------------------------------------------------------------

## Dataset

Source: Stanford SNAP Amazon Review Dataset

Categories used: - Electronics - Beauty - Home & Kitchen - Sports - Toys

Training Scale: - Initial Prototype: \~50K reviews - Final Model (v2):
\~100K reviews (20K per category)

Sentiment Mapping: - Rating ≤ 2 → Negative - Rating = 3 → Neutral -
Rating ≥ 4 → Positive

------------------------------------------------------------------------

## Project Phases Completed

### Phase 1 --- Data Collection & Cleaning

-   Multi-category ingestion
-   Duplicate removal
-   Missing value handling
-   Text preprocessing

### Phase 2 --- Exploratory Data Analysis

-   Rating distribution analysis
-   Review length distribution
-   Verified vs non-verified review comparison
-   Category-wise sentiment breakdown
-   Temporal sentiment trends

### Phase 3 --- Classical ML Baselines

Models trained: - Logistic Regression - Linear SVM - Naive Bayes -
XGBoost - LightGBM

Best classical model optimized using Macro F1 score.

### Phase 4 --- Transformer Modeling (DistilBERT)

Base Model: distilbert-base-uncased

Training configuration: - Max length: 256 - Batch size: 16 - FP16
enabled - Stratified split - Class-weighted loss (v2)

------------------------------------------------------------------------

## Model Versions

### v1 --- Initial Transformer Model

-   Trained on 50K dataset
-   Strong overall accuracy
-   Weak neutral recall

### v2 --- Final Large Model (Deployed Version)

-   Trained on \~100K reviews
-   Class-weighted loss
-   Improved neutral handling
-   Better batch prediction performance

Final Results:

Accuracy: \~90%\
Macro F1: \~0.69

Per-Class F1: - Negative: \~0.70 - Neutral: \~0.43 - Positive: \~0.95

Category-wise evaluation confirms generalization across domains.

⚠️ The Streamlit deployment uses **v2 (100K model)**.

Because this is a transformer model (\~250MB+), inference is slower
compared to classical ML models. The latency is due to model size and
deep neural computation, not inefficient implementation.

------------------------------------------------------------------------

## Error Analysis

Key Observations: - Neutral reviews often contain mixed sentiment. -
Short reviews (e.g., "Good but small") are difficult. - Positive
polarity tends to dominate prediction confidence. - Neutral class
imbalance remains a core challenge.

------------------------------------------------------------------------

## Streamlit Deployment Features

-   Single review prediction
-   Batch prediction
-   Probability visualization
-   Confidence scores
-   Clean interactive UI

The app loads the trained DistilBERT v2 model for real-time inference.

------------------------------------------------------------------------

## What Makes This Project Strong

-   Multi-model comparison (ML + DL)
-   Large-scale transformer training (100K reviews)
-   Class imbalance handling
-   Category-wise evaluation
-   Detailed error analysis
-   Production-ready deployment
-   Clean project structure

------------------------------------------------------------------------

## Limitations

-   Neutral sentiment remains challenging
-   Inference latency due to large transformer model
-   Only English reviews used
-   No domain-specific fine-tuning

------------------------------------------------------------------------

## Tech Stack

-   Python
-   Pandas
-   Scikit-learn
-   XGBoost
-   LightGBM
-   PyTorch
-   HuggingFace Transformers
-   Streamlit
-   Matplotlib / Seaborn

------------------------------------------------------------------------

## Status

✔ Model Training Complete\
✔ Error Analysis Complete\
✔ Streamlit Deployment (v2 Model)\
⏳ Phase 9 --- Business Dashboard (Planned)\
⏳ Phase 10 --- Architecture Documentation (Planned)

------------------------------------------------------------------------

This project demonstrates end-to-end data science workflow, transformer
fine-tuning, deployment capability, and practical handling of real-world
sentiment ambiguity.
