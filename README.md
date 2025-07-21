# Spam Email Detection

This repository contains a Jupyter Notebook for detecting spam emails using machine learning techniques. The project involves data loading, exploratory data analysis (EDA), preprocessing, and model training to classify emails as spam or not spam (ham).

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Key Features](#key-features)
- [Analysis Summary](#analysis-summary)
- [Findings](#findings)
- [Streamlit App Summary](#streamlit-app-summary)
- [Model Deployment](#model-deployment)
- [Running Streamlit App](#running-streamlit-app)

## Project Overview
The goal of this project is to classify emails as spam or ham using:
1. A Jupyter Notebook with complete analysis (`spam_classifier.ipynb`)
2. Data preprocessing and feature engineering
3. Machine learning model training and evaluation

The project covers:
- Data loading and initial exploration
- Data cleaning and preprocessing
- Exploratory data analysis (EDA) with visualizations
- Feature engineering (text processing)
- Training and evaluating classification models

## Dataset
The dataset used in this project contains email messages with two categories:
- `ham`: Legitimate emails (not spam)
- `spam`: Unwanted emails

Key features include:
- `type`: Email classification (0 for ham, 1 for spam)
- `messages`: The email content
- Derived features:
  - `total_characters`: Length of email in characters
  - `total_words`: Word count of email
  - `total_sentences`: Sentence count of email

## Dependencies
To run this notebook, you need the following Python libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn nltk wordcloud
```

## Analysis Summary
- Data Loading & Cleaning:
- Loaded dataset with 5572 emails (5169 after cleaning)
- Removed unnecessary columns
- Encoded target variable (spam=1, ham=0)
- Handled duplicates and missing values

Exploratory Data Analysis:
- Class imbalance: ~87% ham, ~13% spam
- Spam emails tend to be longer (more characters/words)
- Created visualizations showing distribution differences

Text Preprocessing:
- Tokenized email text
- Removed stopwords
- Applied stemming (PorterStemmer)
- Created features for text length analysis

## Findings
Key observations:
- Spam emails are typically longer than ham emails
- The dataset has significant class imbalance
- Text length features may be useful for classification
- Further preprocessing needed for NLP modeling

Future steps include:
- Vectorization (TF-IDF, CountVectorizer)
- Model training (Naive Bayes, Random Forest etc.)
- Performance evaluation

## Streamlit App Summary
A user-friendly interface for email spam detection with:

- **Input Options**:
  - Text input for single emails
  - File upload for batch processing

- **Core Features**:
  - Real-time spam/ham classification
  - Confidence score display
  - Visualizes suspicious keywords

## Model Deployment
The model is deployed using Streamlit Cloud

**Deployment Process:**
- Pushed code to GitHub repository
- Created Streamlit Cloud account
- Connected GitHub repository
- Specified main file as app/spam_classification_app.py

**Deployed App Live URL:**
Email Spam Classifier
```bash
https://spam-email-detection-sneha.streamlit.app/
```

## Running Streamlit App
1. Ensure you have all dependencies installed:
```bash
pip install streamlit pandas numpy scikit-learn nltk wordcloud
streamlit run app/spam_classification_app.py
```
