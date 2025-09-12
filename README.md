---

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

#  Toxic Comment Classification

A multi-label text classification system to detect toxic, obscene, insulting, and hateful comments using:

-  TF-IDF with ML models (Logistic Regression, SVM, Naive Bayes)
-  Deep learning architectures (BiLSTM, BiGRU, TextCNN)
-  Transformers (DistilBERT)

The project compares performance across models, addresses class imbalance, and deploys results in a Streamlit app.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Business Use Case](#business-use-case)
- [Overview](#overview)
- [Demo Streamlit Preview](#demo-streamlit-preview)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Models](#models)
- [Results](#results)
- [Conclusion](#conclusion)
- [Installation & Usage](#installation--usage)
- [Key Learnings](#key-learnings)
- [Project Structure](#project-structure)
- [Connect With Me](#connect-with-me)
- [Acknowledgements](#acknowledgements)

---

## Problem Statement

Online platforms struggle to moderate toxic content. This project builds a multi-label classification system to detect toxic, obscene, threatening, and hateful comments using classical ML, deep learning, and transformer-based models.

---

## Business Use Case

Helps platforms like forums, social media, and comment sections automatically flag harmful content, improving user safety and experience.

---

##  Overview

This project focuses on detecting toxic language in online comments. The dataset contains user comments labeled into multiple categories of toxicity:

- toxic
- severe toxic
- obscene
- threat
- insult
- identity hate

It is a multi-label text classification problem — a comment may belong to more than one category.

---

## Demo Streamlit Preview
<p align='center'>
<img src="https://github.com/user-attachments/assets/46574a54-1eac-4b7a-9c92-7d9a04666204" alt="Single Comment Preview" width="45%" height="500"/>
<img src="https://github.com/user-attachments/assets/14ae0c0b-e0ac-4c81-907f-aa45cd1e1ad8" alt="Bulk CSV Upload" width="45%" height="500"/>
</p>

### Demo Video – Streamlit App Walkthrough

 [Watch the App Demo on Google Drive](https://drive.google.com/file/d/1dvzBUgoyh_4qZaXyaw8sBvU67uuGZDtB/view?usp=sharing)

---

## Objectives

- Preprocess text (cleaning, tokenization, stopword removal)
- Apply TF-IDF vectorization for baseline models
- Build classical ML baselines (Logistic Regression, SVM, Naive Bayes)
- Explore deep learning models (BiLSTM, BiGRU, CNN)
- Fine-tune a Transformer (DistilBERT) for contextual understanding
- Evaluate and compare model performance

---

## Dataset

- Source: [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- Train size: ~160k comments
- Test size: ~64k comments
- Labels: Multi-label, highly imbalanced

---

##  Data Preprocessing

1. Text Cleaning:
   - Lowercasing
   - Removing URLs, HTML tags, numbers, punctuation, and extra spaces
2. Tokenization & Stopword Removal (using NLTK)
3. Vectorization:
   - TF-IDF baseline (top 5000 features)
   - Tokenizer + Padding for deep models

---

##  Models

###  Baseline (TF-IDF + Classical ML)

- Logistic Regression (One-vs-Rest)
- Linear SVM (One-vs-Rest)
- Multinomial Naive Bayes

###  Deep Learning Models

- BiLSTM (captures long-range sequential dependencies)
- BiGRU (lightweight alternative to LSTM)
- TextCNN (captures local n-gram patterns)

###  Transformers

- DistilBERT Fine-Tuning (contextual embeddings, best for nuanced toxic patterns)

---

##  Results

###  Baseline Models

| Model                   | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|------------------------|----------|-----------------|--------------|----------|
| Logistic Regression     | 91.9%    | 0.76            | 0.38         | 0.49     |
| Linear SVM              | 93.1%    | 0.86            | 0.55         | 0.66     |
| Multinomial Naive Bayes | 91.4%    | 0.63            | 0.31         | 0.41     |

### Deep Learning Models (Tokenized Sequences)

- BiLSTM
   - Precision: strong on frequent labels (toxic: 0.85, obscene: 0.77).
   - Weak on minority labels (threat, identity_hate → near zero recall).
   - Macro F1 ≈ 0.45.
   - Captured sequential dependencies, but biased toward frequent classes.

- TextCNN
   - High recall for rare labels (threat recall ≈ 0.70), but precision dropped.
   - Overall Macro F1 ≈ 0.47.
   -  Efficient and recall-focused, but many false positives.

- BiGRU
   - Balanced performance across frequent + rare labels.
   - Macro F1 ≈ 0.57 (highest among RNN/CNN).
   - Faster to train than BiLSTM, better recall than TextCNN, strong compromise model.

- Transformer (DistilBERT Fine-tuning)
   - Achieved F1-micro ≈ 0.777 and F1-macro ≈ 0.620 after 3 epochs.
   - Outperformed all classical ML and RNN/CNN models on macro F1, showing stronger handling of minority classes.
   - Trade-off: Higher compute cost and slower inference vs BiGRU.
   
---
## Conclusion
### Final Choice: BiGRU for Deployment

BiGRU chosen as the deployed model because it offers:
- Best trade-off → balanced precision & recall across all labels.
- Lightweight → faster training/inference compared to BERT.
- Deployment-friendly on limited resources (Streamlit + CPU).
- DistilBERT is retained as a research/secondary model for future scaling, since it provides the best overall macro performance and captures long-range dependencies better.

### Future Scope:
DistilBERT or larger transformers can replace BiGRU when scaling to more powerful hardware or production pipelines.Also, explore ONNX/quantization to optimize DistilBERT for production or use a cascade approach (BiGRU fast path, DistilBERT fallback on low-confidence cases).

##  Installation & Usage

### Environment Setup

```bash
git clone https://github.com/ut-si-ch/Toxic-Comment-Classification.git
cd Toxic-Comment-Classification

conda create -n toxicity_env python=3.9
conda activate toxicity_env

pip install -r requirements.txt
```

### Run Notebook

```bash
jupyter notebook Toxicity_Project.ipynb
```

### Streamlit App (Deployment)

```bash
streamlit run app.py
```

---

## Key Learnings

- Classical ML (TF-IDF + SVM) provides strong baselines
- Deep models (BiLSTM, GRU, CNN) learn semantic and sequential patterns
- Transformers (DistilBERT) capture context & nuance best, especially for rare labels so it can be tried 
- Class imbalance significantly impacts minority label detection (handled with sample weighting / focal loss)

---

## Project Structure

```
├── data/                  # Dataset
├── notebooks/             # Jupyter notebooks
├── saved_models/          # Trained models
├── app.py                 # Streamlit deployment script
├── requirements.txt       # Dependencies
├── Toxicity_Project.ipynb # Main notebook
└── README.md              # Project documentation
```

---

## Connect With Me

- [LinkedIn](https://www.linkedin.com/in/uttam-singh-chaudhary-98408214b)
- [Portfolio](https://datascienceportfol.io/uttamsinghchaudhary)
- [Email](mailto:uttamsinghchaudhary@gmail.com)

---

## Acknowledgements

- [Kaggle Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- Hugging Face Transformers
- TensorFlow & Scikit-learn

---
