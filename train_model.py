import pandas as pd
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

data1=pd.read_csv('dataset\data.csv')
data= pd.concat([data1, data2], ignore_index=True)

data = data[['symptom_text', 'diseases']].dropna()

def clean_text(text):
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    return text.strip()

data['symptom_text'] = data['symptom_text'].apply(clean_text)

vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(data['symptom_text'])
y=data['diseases']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)
for a in [0.01, 0.05, 0.1, 0.5, 1.0]:
    model = MultinomialNB(alpha=a)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


with open("naive_bayes_disease_model.pkl", "wb") as f:
    pickle.dump(model, f)


with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
