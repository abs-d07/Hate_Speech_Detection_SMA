import streamlit as st
import joblib as joblib
import json
from sklearn.metrics import classification_report, f1_score
import pandas as pd, joblib
import joblib
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd, joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score
)
import joblib, os, json
import pandas as pd

# # --- Load model and vectorizer ---
# @st.cache_resource
# def load_model():
#     model = joblib.load("logreg.pkl")
#     vectorizer = joblib.load("tfidf.pkl")
#     return model, vectorizer

# model, vectorizer = load_model()

# # --- Streamlit App UI ---
# st.set_page_config(page_title="Hate Speech Detection", page_icon="💬")
# st.title("🧠 Hate Speech / Toxic Comment Detector")
# st.write("Enter any social media comment below to test if it is toxic or not:")

# user_input = st.text_area("💬 Type your comment here:", height=120)

# if st.button("Predict"):
#     if user_input.strip() == "":
#         st.warning("Please enter some text to analyze.")
#     else:
#         X = vectorizer.transform([user_input])
#         pred = model.predict(X)[0]
#         prob = model.predict_proba(X)[0].max()
#         label = "TOXIC" if pred == 1 else "NON-TOXIC"
#         color = "red" if pred == 1 else "green"
#         st.markdown(f"### Prediction: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
#         st.write(f"**Confidence:** {prob:.2f}")
#         st.success("✅ Model tested successfully!")

import streamlit as st
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# -----------------------------
# PAGE SETUP
# -----------------------------
st.set_page_config(page_title="Toxic Comment Detector", page_icon="💬", layout="centered")

st.markdown("""
    <h1 style='text-align:center;'>🧠 Social Media Toxic Comment Detector</h1>
    <p style='text-align:center; font-size:18px;'>Compare Traditional ML vs Transformer-based AI models for hate speech detection.</p>
""", unsafe_allow_html=True)

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_logreg():
    model = joblib.load("logreg.pkl")
    vectorizer = joblib.load("tfidf.pkl")
    return model, vectorizer

# @st.cache_resource
# def load_bert():
#     model_path = "bert_finetuned"
#     # model = AutoModelForSequenceClassification.from_pretrained(model_path)
#     model = AutoModelForSequenceClassification.from_pretrained(model_path, from_safetensors=True)
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     model.eval()
#     return model, tokenizer, device

@st.cache_resource
def load_bert():
    import os
    model_path = os.path.join(os.getcwd(), "bert_finetuned")
    print("🔍 Loading model from:", model_path)
    
    # ✅ No 'from_safetensors' flag (it auto-detects safely)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device


logreg_model, tfidf_vectorizer = load_logreg()
bert_model, bert_tokenizer, device = load_bert()

# -----------------------------
# SIDEBAR OPTIONS
# -----------------------------
st.sidebar.title("⚙️ Model Options")
model_choice = st.sidebar.radio("Select Model:", ["Logistic Regression (TF-IDF)", "DistilBERT Transformer"])
st.sidebar.markdown("---")
st.sidebar.info("💡 Logistic Regression uses simple word frequencies.\n\nDistilBERT understands language context (transformer-based).")

# -----------------------------
# USER INPUT
# -----------------------------
st.subheader("💬 Enter a comment for analysis")
user_input = st.text_area("Type your social media comment here:", height=120)

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a valid comment.")
    else:
        if model_choice == "Logistic Regression (TF-IDF)":
            X = tfidf_vectorizer.transform([user_input])
            pred = logreg_model.predict(X)[0]
            prob = logreg_model.predict_proba(X)[0].max()
            label = "TOXIC" if pred == 1 else "NON-TOXIC"
            color = "red" if pred == 1 else "green"
            st.markdown(f"### 🧩 Model: <b>Logistic Regression (TF-IDF)</b>", unsafe_allow_html=True)
            st.markdown(f"### Prediction: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
            st.write(f"**Confidence:** {prob:.2f}")

        else:  # DistilBERT
            inputs = bert_tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
            with torch.no_grad():
                outputs = bert_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=1)[0]
                pred = torch.argmax(probs).item()
                confidence = probs[pred].item()

            label = "TOXIC" if pred == 1 else "NON-TOXIC"
            color = "red" if pred == 1 else "green"
            st.markdown(f"### 🧩 Model: <b>DistilBERT Transformer</b>", unsafe_allow_html=True)
            st.markdown(f"### Prediction: <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)
            st.write(f"**Confidence:** {confidence:.2f}")

        st.success("✅ Prediction complete!")

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("""
    <p style='text-align:center; font-size:15px;'>
    👩‍💻 Built with ❤️ using Streamlit, Scikit-learn, and Hugging Face Transformers.<br>
    Compare models and explore how AI detects toxic content on social media.
    </p>
""", unsafe_allow_html=True)
