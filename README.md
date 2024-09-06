# Spam Classification Project

## Overview

This project implements and compares multiple machine learning approaches for classifying email messages as spam or ham (non-spam). The system utilizes natural language processing techniques and three different models:

1. A neural network classifier
2. A Naive Bayes classifier
3. A BERT-based deep learning model

## Dataset

The project uses a dataset of 5,572 email messages, labeled as either spam or ham. The dataset is imbalanced, with approximately 13% spam messages.

## Data Preprocessing

The following preprocessing steps are applied to the raw text data:

1. Removal of punctuation, symbols, and special characters
2. Tokenization and lemmatization using NLTK's WordNetLemmatizer
3. Removal of stop words
4. TF-IDF vectorization for the neural network and Naive Bayes models
5. BERT tokenization and encoding for the BERT-based model

To address the class imbalance, Synthetic Minority Over-sampling Technique (SMOTE) is applied to the training data.

## Models

### 1. Neural Network Classifier

- Architecture: 3-layer feedforward neural network
- Input layer: 48,159 neurons (based on TF-IDF features)
- Hidden layers: 512 neurons and 256 neurons with ReLU activation
- Output layer: 1 neuron with sigmoid activation
- Dropout layers (0.3) for regularization
- Optimizer: Adam
- Loss function: Binary cross-entropy

### 2. Naive Bayes Classifier

- Implementation: Scikit-learn's MultinomialNB
- Features: TF-IDF vectors

### 3. BERT-based Model

- Pre-trained BERT model: 'bert_en_uncased_L-12_H-768_A-12'
- Fine-tuning layers:
  - Dense layer (512 neurons, ReLU activation)
  - Dropout layer (0.3)
  - Dense layer (256 neurons, ReLU activation)
  - Dropout layer (0.3)
  - Output layer (1 neuron, sigmoid activation)
- Optimizer: Adam
- Loss function: Binary cross-entropy

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Scikit-learn
- NLTK
- Gensim
- Pandas
- NumPy
- Matplotlib
- Imbalanced-learn (for SMOTE)
- TensorFlow Hub (for BERT model)

## Results

The performance of each model on the test set:

1. Neural Network: [Add accuracy and other metrics]
2. Naive Bayes: [Add accuracy and other metrics]
3. BERT-based Model: 96.41% accuracy

## Future Work

- Experiment with other pre-trained language models (e.g., RoBERTa, DistilBERT)
- Implement cross-validation for more robust evaluation
- Explore ensemble methods combining multiple models
- Analyze feature importance and model interpretability
