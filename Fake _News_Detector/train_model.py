import pandas as pd
import re
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier # Often better for text
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load dataset
# Ensure these paths are correct relative to your script
fake = pd.read_csv("datasets/Fake.csv")
true = pd.read_csv("datasets/True.csv")

fake["label"] = 0
true["label"] = 1

# Take a sample if the dataset is too large for your RAM, 
# otherwise use the full set
data = pd.concat([fake, true]).reset_index(drop=True)

# 2. Enhanced Cleaning Function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'<.*?>', '', text) # Remove HTML tags
    text = re.sub(r'\W', ' ', text) # Remove punctuation
    text = re.sub(r'\d', '', text) # Remove numbers
    text = re.sub(r'\s+', ' ', text) # Remove extra spaces
    return text.strip()

print("Cleaning text...")
data["text"] = data["text"].apply(clean_text)

# 3. Splitting the Data (CRITICAL STEP)
# You should never train on your whole dataset. 
# You need a "test set" to see if the model actually works on new data.
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.2, random_state=42
)

# 4. Vectorization
# Adding max_features limits the vocabulary to the most important words
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7, max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Model Selection
# PassiveAggressiveClassifier is excellent for large text datasets
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_vec, y_train)

# 6. Evaluation (What recruiters look for)
y_pred = model.predict(X_test_vec)
score = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(score*100,2)}%')
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Save model and vectorizer
with open("news_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("Model and vectorizer saved successfully!")