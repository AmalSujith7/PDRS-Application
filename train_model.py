import spacy
import torch
from transformers import RobertaTokenizer, RobertaModel
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
import joblib

nlp = spacy.load("en_core_web_sm")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta_model = RobertaModel.from_pretrained("roberta-base")

def extract_ner_features(text):
    doc = nlp(text)
    entities = {"ORG": 0, "PERSON": 0, "GPE": 0, "MONEY": 0, "EMAIL": 0, "URL": 0}
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_] += 1
    return list(entities.values())

def extract_roberta_features(text):
    inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding="max_length")
    outputs = roberta_model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()
    return cls_embedding.flatten()

file_path = 'fraud_email_.csv'
dataset = pd.read_csv(file_path)

X_ner = np.array([extract_ner_features(str(text)) for text in dataset["Text"]])
X_roberta = np.array([extract_roberta_features(str(text)) for text in dataset["Text"]])
X = np.hstack((X_ner, X_roberta))
y = dataset["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
joblib.dump(model, 'model.joblib')
print("Model saved as 'model.joblib'")
