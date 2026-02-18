
# E-Commerce Review Sentiment Classification

## Project Overview

This project builds a complete end-to-end NLP pipeline to classify Amazon product reviews into:
- Negative
- Neutral
- Positive

It includes:
- Classical Machine Learning benchmarks
- Transformer fine-tuning (DistilBERT)
- Class imbalance handling
- Scaling from 50K to 100K reviews
- Error analysis
- Streamlit deployment

---

## Dataset

Source: Amazon Review Dataset (Stanford SNAP)

Categories used:
- Electronics
- Beauty
- Toys
- Sports
- Home & Kitchen

Dataset sizes:
- 50K (10K × 5 categories)
- 100K (20K × 5 categories)

Label Mapping:
- Rating 1–2 → Negative (0)
- Rating 3 → Neutral (1)
- Rating 4–5 → Positive (2)

---

## Phase 1 – Data Cleaning & Preprocessing

- Removed missing reviews
- Removed duplicates
- Text normalization (lowercase, stopword removal, lemmatization)
- Created clean text column
- Generated numeric sentiment labels

---

## Phase 2 – Exploratory Data Analysis

Performed:
- Rating distribution analysis
- Review length analysis
- Sentiment distribution by category
- Verified vs rating comparison
- Sentiment trend over time

Key insight:
Dataset is highly imbalanced with majority positive reviews.

---

## Phase 3 – Classical ML Benchmark

Vectorization:
- TF-IDF (1–2 grams)
- 50,000 features

Models trained:
- Logistic Regression
- Linear SVM
- Naive Bayes
- XGBoost
- LightGBM

Observation:
Classical ML models struggled with neutral sentiment classification.

---

## Phase 4 – Transformer Fine-Tuning (DistilBERT)

Model:
- distilbert-base-uncased
- 3 output labels

Training:
- Max length: 256
- Learning rate: 2e-5
- Batch size: 16
- FP16 enabled
- GPU training

---

## Phase 5 – Model Evaluation (100K Model)

Accuracy: ~90%
Macro F1: ~0.69

Per-Class F1:
- Negative: ~0.70
- Neutral: ~0.42
- Positive: ~0.95

Neutral remains the most challenging class due to mixed sentiment language.

---

## Phase 6 – Error Analysis

- Inspected misclassified neutral samples
- Analyzed prediction confidence
- Per-category performance breakdown

Finding:
Neutral reviews often contain both positive and negative cues.

---

## Phase 7 – Scaling

Scaled training from 50K → 100K reviews.

Scaling improved stability and overall macro F1 score.

---

## Phase 8 – Streamlit Deployment

Features:
- Single review prediction
- Batch review prediction
- Confidence score display
- Clean UI

Model packaged with:
- config.json
- model.safetensors
- tokenizer files
- label_map.json

---

## Model Comparison Summary

| Model | Accuracy | Macro F1 |
|-------|----------|----------|
| Logistic Regression | ~85% | ~0.58 |
| Linear SVM | ~86% | ~0.56 |
| DistilBERT (50K) | ~88% | ~0.66 |
| DistilBERT (100K) | ~90% | ~0.69 |

---

## Tech Stack

- Python
- Pandas
- Scikit-learn
- XGBoost
- LightGBM
- HuggingFace Transformers
- PyTorch
- Streamlit
