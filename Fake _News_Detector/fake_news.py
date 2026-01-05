import pandas as pd 
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
from nltk.tokenize import sent_tokenize

fake =pd.read_csv("datasets\Fake.csv")
true =pd.read_csv("datasets\True.csv")

fake["label"]=0
true["label"]=1

data = pd.concat([fake, true])

def clean_text(text):
    text = text.lower()  
    text = re.sub(r'\W',' ',text)   # remove special chara
    text = re.sub(r'\s+',' ',text)   # remove extra space
    return text

data["text"]= data["text"].apply(clean_text)

x = data["text"]
y = data["label"]
vect= TfidfVectorizer(stop_words="english", max_df=0.7)
x = vect.fit_transform(x)

x_train,x_test, y_train, y_test = train_test_split(
    x,y,test_size=0.2, random_state=42
)

model =LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accur= accuracy_score(y_test, y_pred)

def predict_news(news_text):
    news_text = clean_text(news_text)
    vector = vect.transform([news_text])
    prediction = model.predict(vector)

    if prediction[0] == 1:
        return "Real news"
    else: 
        return "Fake news"
print("\n--- TESTING WITH SAMPLE NEWS ---\n")

sample_news = """
The White House announced on Tuesday that the administration will continue its efforts to strengthen economic growth and job creation.
Officials stated that unemployment rates have remained stable and that new policies are aimed at supporting middle-class families.
The announcement was made during a press briefing attended by senior government officials.
"""

result = predict_news(sample_news)
print("Prediction:", result)
