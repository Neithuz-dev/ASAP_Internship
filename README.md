# 🩺 DiagnosAI – Smart Health Diagnosis

**DiagnosAI** is an AI-powered web application designed to predict potential medical conditions based on user-described symptoms. It utilizes machine learning (Naive Bayes) to analyze symptoms and provides instant predictions, confidence scores, and relevant internet insights.

🚀 **Live Demo:** [https://diagnosai-smart.streamlit.app/](https://diagnosai-smart.streamlit.app/)

---

## ✨ Features

*   **Symptom-Based Prediction**: Users can enter symptoms in natural language (e.g., "I have fever and stomach pain").
*   **Smart Suggestions**: The app suggests relevant symptoms (including multi-word phrases like "joint pain") based on the input context to improve prediction accuracy.
*   **Confidence Score**: Displays the model's confidence in the prediction.
*   **Risk Assessment**: Automatically flags high-risk conditions (e.g., Heart Attack, Pneumonia) and advises immediate medical attention.
*   **Internet Insights**:
    *   Fetches real-time related articles from the web using DuckDuckGo.
    *   **Fallback Mechanism**: Provides a direct Google Search link if automated retrieval fails.
*   **Responsive UI**: Built with Streamlit for a clean and interactive user experience.

## 🛠️ Tech Stack

*   **Language**: Python
*   **Framework**: [Streamlit](https://streamlit.io/)
*   **Machine Learning**: Scikit-Learn (Multinomial Naive Bayes, TF-IDF Vectorizer)
*   **Data Processing**: Pandas, NumPy
*   **Search Integration**: `duckduckgo-search` library

## 📂 Project Structure

```bash
DiagnoseAI Project/
├── app.py                 # Main Streamlit application
├── train_model.py         # Script to train and save the ML model
├── requirements.txt       # Python dependencies
├── models/                # Saved model artifacts
│   ├── model.pkl          # Trained classifier and vectorizer
│   └── symptoms.pkl       # Extracted symptom list for suggestions
├── dataset/               # Dataset used for training
│   └── Symptom2Disease.csv
└── notebook/              # Jupyter notebooks for data analysis
```

## 🚀 How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Neithuz-dev/ASAP_Internship.git
    cd "ASAP_Internship/DiagnoseAI Project"
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

4.  **Train the model (Optional):**
    If you modify the dataset or want to retrain the model:
    ```bash
    python train_model.py
    ```

## ⚠️ Disclaimer
This application is for **informational purposes only** and does not constitute professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified health provider with any questions you may have regarding a medical condition.
