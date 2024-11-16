import joblib
import numpy as np
from transformers import RobertaTokenizer, RobertaModel
import spacy
model = joblib.load('model.joblib')
print("Model loaded successfully.")
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

def detect_phishing(text):
    X_ner = extract_ner_features(text)
    X_roberta = extract_roberta_features(text)
    X = np.hstack((X_ner, X_roberta)).reshape(1, -1)

    prediction = model.predict(X)
    return prediction[0]
