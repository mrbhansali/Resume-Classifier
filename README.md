# Resume Classification System using NLP
## Overview

Recruiters receive thousands of resumes for multiple job roles. Manually screening them is time-consuming and inefficient.

This project builds an automated Resume Classification System using Natural Language Processing (NLP) and supervised machine learning models to categorize resumes into predefined job roles.

The goal is to assist in automated resume screening and improve hiring efficiency.

## Dataset

Source: Kaggle Resume Dataset

Link: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset

The dataset contains resumes labeled with different job categories.

## Tech Stack

Python

Pandas & NumPy

Matplotlib (for visualization)

Scikit-learn

NLP (TF-IDF Vectorization)

Gensim

FastText

## Methodology

Data Preprocessing

Text cleaning (removal of special characters, lowercasing, etc.)

Handling missing values

Feature Extraction

TF-IDF Vectorization to convert resume text into numerical features

Model Training
The following machine learning models were trained and compared:

Logistic Regression

Linear SVM

Random Forest Classifier

Gensim (Word2Vec)

FastText

Model Evaluation

Accuracy Score

Confusion Matrix

Classification Report

## Results

Compared multiple ML models for performance.

Achieved strong classification accuracy on test data.

Identified the best-performing model based on evaluation metrics.


## Key Learnings

Practical implementation of NLP pipelines

Text preprocessing techniques

Feature engineering using TF-IDF

Model comparison and evaluation

End-to-end ML workflow

## Future Improvements

Use advanced embeddings (Word2Vec, GloVe, BERT)

Hyperparameter tuning using GridSearchCV

Deploy as a web application using Flask/Streamlit

Handle imbalanced class distribution
