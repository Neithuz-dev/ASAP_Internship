import streamlit as st
import pickle

model = pickle.load(open("news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Student Fake News Detector")
st.write("Enter the news article text below to check its authenticity.")

user_input = st.text_area("Paste News Content Here:")

if st.button("Analyze News"):
    if user_input:
        data = vectorizer.transform([user_input])
        prediction = model.predict(data)
        result = "This news looks REAL" if prediction[0] == 1 else " This news looks FAKE"
        
        if prediction[0] == 1:
            st.success(result)
        else:
            st.error(result)
    else:
        st.warning("Please enter some text first!")