# -*- coding: utf-8 -*-
"""PDRS Bert Model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1MkMvkOyX5bunhqSZa3FOR2AEejiZ2nCh
"""

!pip install vaderSentiment transformers torch

import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from transformers import RobertaTokenizer, RobertaModel, pipeline
import torch
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

file_path = 'fraud_email_.csv'
data = pd.read_csv(file_path)

print(data.head())
print(data.columns)

sentiment_analyzer = SentimentIntensityAnalyzer()
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

keywords = ["urgent", "password reset", "limited time", "account locked"]
spoofed_entities = ["Microsoft", "Amazon", "Bank", "Paypal"]

def extract_nlp_features(text):
    features = {}
    if isinstance(text, str):
        features['has_suspicious_keywords'] = int(any(keyword in text.lower() for keyword in keywords))
        sentiment = sentiment_analyzer.polarity_scores(text)
        features['negative_sentiment'] = sentiment['neg']
        features['positive_sentiment'] = sentiment['pos']
        entities = ner_pipeline(text)
        features['spoofed_entity'] = int(any(ent['word'] in spoofed_entities for ent in entities))
    else:
        features['has_suspicious_keywords'] = 0
        features['negative_sentiment'] = 0
        features['positive_sentiment'] = 0
        features['spoofed_entity'] = 0
    return features

nlp_features_df = data['Text'].apply(lambda x: pd.Series(extract_nlp_features(x)))
data = pd.concat([data, nlp_features_df], axis=1)

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta_model = RobertaModel.from_pretrained("roberta-base")

def get_roberta_embeddings(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    outputs = roberta_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].detach().numpy().flatten()

data['roberta_embeddings'] = data['Text'].apply(lambda x: get_roberta_embeddings(x))

nlp_features = nlp_features_df.values
roberta_features = np.stack(data['roberta_embeddings'].values)
combined_features = np.hstack([nlp_features, roberta_features])

labels = data['Class']
X_train, X_test, y_train, y_test = train_test_split(combined_features, labels, test_size=0.2, random_state=42)

classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)

train_accuracy = classifier.score(X_train, y_train)
test_accuracy = classifier.score(X_test, y_test)
print(f"Train Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")