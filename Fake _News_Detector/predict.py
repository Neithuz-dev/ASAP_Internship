import pickle

# 1. Load the saved model and vectorizer
model = pickle.load(open("news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def predict_news(text):
    # Vectorize the input text
    data = vectorizer.transform([text])
    # Make prediction
    prediction = model.predict(data)
    return "REAL" if prediction[0] == 1 else "FAKE"

# 2. Test it
user_input = input("Enter news text to verify: ")
print(f"The news is: {predict_news(user_input)}")