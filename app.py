import streamlit as st
import pickle
import re

st.set_page_config(
    page_title="DiagnosAI – Smart Health Diagnosis",
    layout="centered"
)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d+', '', text)
    return text.strip()

@st.cache_resource
def load_assets():
    with open("models/model.pkl", "rb") as f:
        model, vectorizer = pickle.load(f)
    return model, vectorizer

# Load symptom list at module level
with open("models/symptoms.pkl", "rb") as f:
    symptom_list = pickle.load(f)

try:
    model, vectorizer = load_assets()

    with st.sidebar:
        st.markdown("🩺DiagnosAI")
        st.caption("AI-Powered Disease Prediction")
        st.divider()
        st.markdown(" How it works")
        st.write(
            """
            1. Enter your symptoms
            2. Click **Predict Disease**
            3. Get AI-based insights
            """
        )
        st.warning("This is not a medical diagnosis.")

    st.markdown(
        """
        <h1 style='text-align:center;'>DiagnosAI</h1>
        <p style='text-align:center; color: gray;'>
        Smart symptom-based disease prediction using AI
        </p>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    user_input = st.text_area(
        "Example: I have fever, headache and body pain"
    )
    st.markdown("<br>", unsafe_allow_html=True)

    prediction = None
    confidence = None

    if st.button("Predict Disease"):
        if user_input.strip():
            cleaned = clean_text(user_input)
            vectorized = vectorizer.transform([cleaned])

            prediction = model.predict(vectorized)[0]
            confidence = model.predict_proba(vectorized).max() * 100

            input_words = cleaned.split()
            suggestions = [s for s in symptom_list if s not in input_words][:6]

            if len(cleaned.split()) < 4:
                st.warning("Please add more symptoms for better accuracy.")
                st.write("**Common related symptoms you may have:**")
                st.write(", ".join(suggestions))

            if any(word in prediction.lower() for word in [
                    "heart", "stroke", "attack", "pneumonia"
                ]):
                    st.error(
                        "**High Risk Detected**\n\n"
                        "Please seek immediate medical attention."
                    )
            else:
                    st.info(
                        " **Recommendation**\n\n"
                        "Consult a qualified doctor for confirmation."
                    )
        else:
            st.warning(" Please enter symptoms.")

    if prediction is not None and confidence is not None:
        st.divider()
        st.markdown("##  Diagnosis Report")

        st.success(f"**Predicted Condition:** {prediction}")
        st.write(f"**Confidence:** {confidence:.2f}%")

    st.divider()
    st.caption(" DiagnosAI | Built with Streamlit & Machine Learning")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
