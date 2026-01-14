import pandas as pd
import pickle
import re
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


os.makedirs("models", exist_ok=True)

data = pd.read_csv("dataset/Symptom2Disease.csv")

def clean_text(text):
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    return text.strip()

data['text'] = data['text'].apply(clean_text)

X = data['text']
y = data['label']

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


vectorizer = TfidfVectorizer(stop_words="english")
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

model = MultinomialNB()
model.fit(x_train_vec, y_train)

y_pred = model.predict(x_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))

NON_SYMPTOM_WORDS = {
    "feel", "feeling", "really", "lot", "experiencing", "experience", "having", 
    "suffering", "suffer", "get", "got", "makes", "make", "seems", "still", 
    "high", "low", "severe", "mild", "quite", "body", "like", "difficult", 
    "uncomfortable", "discomfort", "lost", "hurts", "bit", "just", "going", 
    "often", "also", "times", "days", "weeks", "months", "since", "usually", 
    "always", "sometimes", "started", "felt", "years", "recently", "noticed", 
    "trouble", "frequently", "time", "constantly", "feels", "parts", "mainly", 
    "areas", "different", "certain", "particularly", "especially", "getting", 
    "even", "every", "feelings", "known"
}

feature_names = vectorizer.get_feature_names_out()

tfidf_scores = x_train_vec.mean(axis=0).A1

important_words = sorted(
    zip(feature_names, tfidf_scores),
    key=lambda x: x[1],
    reverse=True
)

common_symptoms = []
for word, score in important_words:
    if (
        len(word) > 3 and
        word not in NON_SYMPTOM_WORDS
    ):
        common_symptoms.append(word)
    if len(common_symptoms) == 25:
        break

with open("models/model.pkl", "wb") as f:
    pickle.dump((model, vectorizer), f)

with open("models/symptoms.pkl", "wb") as f:
    pickle.dump(common_symptoms, f)

print("Model and clean symptom list saved successfully")