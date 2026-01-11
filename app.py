import streamlit as st
import pickle
import re

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="DiagnosAI – Smart Health Diagnosis",
    layout="centered"
)

# ---------------- TEXT CLEANING ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d+', '', text)
    return text.strip()

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_assets():
    with open("naive_bayes_disease_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

try:
    model, vectorizer = load_assets()

    # ---------------- SIDEBAR ----------------
    with st.sidebar:
        st.markdown(" DiagnosAI")
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

    # ---------------- MAIN UI ----------------
    st.markdown(
        """
        <h1 style='text-align:center;'>DiagnosAI</h1>
        <p style='text-align:center; color: gray;'>
        Smart symptom-based disease prediction using AI
        </p>
         <h5 style='text-align:center; color: gray;'>
        Enter atleast 4 symptoms to diagnose accurate disease</h5>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # ---------------- INPUT ----------------
    user_input = st.text_area(
        " Enter your symptoms (comma separated)",
        placeholder="fever, headache, nausea, vomiting"
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # ---------------- PREDICTION ----------------
    if st.button("🔍 Predict Disease", use_container_width=True):
        if user_input.strip():
            cleaned = clean_text(user_input)
            vectorized = vectorizer.transform([cleaned])

            prediction = model.predict(vectorized)[0]
            confidence = model.predict_proba(vectorized).max() * 100

            st.divider()
            st.markdown("## 🧾 Diagnosis Report")

            st.success(f"**Predicted Condition:** {prediction}")
            st.write(f"**Confidence:** {confidence:.2f}%")

            if any(word in prediction.lower() for word in [
                "heart", "stroke", "attack", "pneumonia"
            ]):
                st.error(
                    "🚨 **High Risk Detected**\n\n"
                    "Please seek immediate medical attention."
                )
            else:
                st.info(
                    "🩺 **Recommendation**\n\n"
                    "Consult a qualified doctor for confirmation."
                )

        else:
            st.warning("⚠️ Please enter symptoms.")

    # ---------------- FOOTER ----------------
    st.divider()
    st.caption("© 2026 DiagnosAI | Built with Streamlit & Machine Learning")

except FileNotFoundError:
    st.error("❌ Model files not found. Train the model and save pickle files first.")
