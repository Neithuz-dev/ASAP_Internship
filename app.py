import streamlit as st
import pickle
import re
try:
    from duckduckgo_search import DDGS
except ImportError:
    from ddgs import DDGS
import pandas as pd
from collections import Counter
try:
    from googlesearch import search
except ImportError:
    print("googlesearch not found")


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

def search_online(query):
    try:
        with DDGS() as ddgs:
            # region='us-en' enforces English results
            # specific query structure to find conditions
            # Search from trusted medical sources only
            trusted_sites = "site:mayoclinic.org OR site:webmd.com OR site:cdc.gov OR site:healthline.com OR site:nhs.uk OR site:clevelandclinic.org"
            search_query = f"{query} symptoms {trusted_sites}"
            results = list(ddgs.text(
                search_query,
                region='us-en',
                max_results=3
            ))
            
            # Fallback if no results (often due to IP blocking on cloud)
            if not results:
                 print("Trying fallback backend='html'")
                 try:
                    results = list(ddgs.text(
                        search_query,
                        region='us-en',
                        max_results=3,
                        backend='html'
                    ))
                 except Exception:
                     pass # Proceed to next fallback
            
            # Final Fallback: Google Search (googlesearch-python)
            if not results:
                print("Trying fallback: Google Search")
                google_results = search(search_query, num_results=3, advanced=True)
                for res in google_results:
                    results.append({
                        "title": res.title,
                        "href": res.url,
                        "body": res.description
                    })

        return results
    except Exception as e:
        st.error(f"Debug Error: {e}") # Show error in UI for debugging
        print(f"Search Error: {e}")
        return []

@st.cache_resource
def load_data():
    try:
        df = pd.read_csv("dataset/Symptom2Disease.csv")
        return df
    except Exception:
        return pd.DataFrame() # Return empty if file missing

@st.cache_resource
def load_assets():
    with open("models/model.pkl", "rb") as f:
        model, vectorizer = pickle.load(f)
    return model, vectorizer

# Load data
with open("models/symptoms.pkl", "rb") as f:
    symptom_list = pickle.load(f)

df = load_data()

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
                
                # Context-aware suggestions
                suggestions = []
                if not df.empty:
                    # Find rows containing ANY of the input words
                    mask = df['text'].str.contains(cleaned, case=False, na=False)
                    relevant_rows = df[mask]
                    
                    if not relevant_rows.empty:

                        all_text = " ".join(relevant_rows['text'].tolist()).lower()
                        
                        # Count occurrences of each known symptom phrase in the filtered text
                        symptom_counts = Counter()
                        for symptom in symptom_list:
                             # Check if symptom is not already in user input
                            if symptom not in cleaned:
                                count = all_text.count(symptom)
                                if count > 0:
                                    symptom_counts[symptom] = count
                        
                        # Get top 8 most frequent symptoms from context
                        suggestions = [s for s, c in symptom_counts.most_common(8)]
                
                # Fallback if no relevant context found
                if not suggestions:
                     suggestions = [s for s in symptom_list if s not in input_words][:6]

                st.markdown("**Common related symptoms you may have:**")
                # Display as chips/tags
                st.write(", ".join(suggestions))

            # 1. Always show Prediction Report
            st.divider()
            st.markdown("##  Diagnosis Report")
            st.success(f"**Predicted Condition:** {prediction}")
            st.write(f"**Confidence:** {confidence:.2f}%")

            # 2. Risk Assessment
            if any(word in prediction.lower() for word in ["heart", "stroke", "attack", "pneumonia"]):
                st.error("**High Risk Detected**\n\nPlease seek immediate medical attention.")
            else:
                st.info("**Recommendation**\n\nConsult a qualified doctor for confirmation.")

            # 3. Dynamic Web Insights (Low Confidence)
            if confidence < 60:
                st.divider()
                st.markdown("### 🌐 Additional Internet Insights (Low Confidence)")
                st.warning(f"⚠️ Service is uncertain ({confidence:.2f}%). Below are related findings from the web:")
                
                results = search_online(cleaned)
                if results:
                    for res in results:
                        # Display Title with clickable link and Body
                        st.markdown(f"**[{res['title']}]({res['href']})**")
                        st.caption(f"{res['body']}")
                        st.markdown("---")
                else:
                    st.warning("Could not fetch detailed results automatically.")
                    # Provide a direct link to Google Search as a reliable fallback
                    google_url = f"https://www.google.com/search?q=medical+condition+with+symptoms+{cleaned.replace(' ', '+')}"
                    st.markdown(f"**👉 [Click here to search on Google]({google_url})**")
        else:
            st.warning(" Please enter symptoms.")



    st.divider()
    st.caption(" DiagnosAI | Built with Streamlit & Machine Learning")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
