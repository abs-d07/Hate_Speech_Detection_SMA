import streamlit as st
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
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
import streamlit as st
import joblib
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification



# ----------------------------------------
# PAGE CONFIG
# ----------------------------------------
st.set_page_config(
    page_title="Toxic Comment Detection — Model Comparison",
    page_icon="💬",
    layout="wide"
)

st.markdown("""
    <h1 style='text-align:center;'>🧠 Toxic Comment Detector</h1>
    <p style='text-align:center; font-size:18px;'>
    Compare <b>Traditional ML (TF-IDF + Logistic Regression)</b> vs <b>Transformer-based DistilBERT</b> models side-by-side.
    </p>
""", unsafe_allow_html=True)

# ----------------------------------------
# LOAD MODELS
# ----------------------------------------
@st.cache_resource
def load_logreg():
    model = joblib.load("logreg.pkl")
    vectorizer = joblib.load("tfidf.pkl")
    return model, vectorizer

@st.cache_resource
def load_bert():
    # model_path = os.path.join(os.getcwd(), "bert_finetuned")
    model_path = "abbu1402/bert_finetuned"
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return model, tokenizer, device

logreg_model, tfidf_vectorizer = load_logreg()
bert_model, bert_tokenizer, device = load_bert()

# ----------------------------------------
# USER INPUT
# ----------------------------------------
st.subheader("💬 Enter a comment to analyze")
user_input = st.text_area("Type a social media comment here:", height=120)

if st.button("🔍 Analyze Comment"):
    if user_input.strip() == "":
        st.warning("Please enter a valid comment.")
    else:
        # -----------------------
        # Logistic Regression Prediction
        # -----------------------
        X = tfidf_vectorizer.transform([user_input])
        pred_lr = logreg_model.predict(X)[0]
        prob_lr = logreg_model.predict_proba(X)[0].max()
        label_lr = "TOXIC" if pred_lr == 1 else "NON-TOXIC"
        color_lr = "red" if pred_lr == 1 else "green"

        # -----------------------
        # DistilBERT Prediction
        # -----------------------
        inputs = bert_tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)[0]
            pred_bert = torch.argmax(probs).item()
            conf_bert = probs[pred_bert].item()

        label_bert = "TOXIC" if pred_bert == 1 else "NON-TOXIC"
        color_bert = "red" if pred_bert == 1 else "green"

        # -----------------------
        # DISPLAY SIDE BY SIDE
        # -----------------------
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### 🤖 Logistic Regression (TF-IDF)")
            st.markdown(f"<h3 style='color:{color_lr};'>{label_lr}</h3>", unsafe_allow_html=True)
            st.progress(prob_lr)
            st.caption(f"Confidence: {prob_lr:.2f}")

        with col2:
            st.markdown(f"### 🧩 DistilBERT Transformer")
            st.markdown(f"<h3 style='color:{color_bert};'>{label_bert}</h3>", unsafe_allow_html=True)
            st.progress(conf_bert)
            st.caption(f"Confidence: {conf_bert:.2f}")

        # Summary
        st.markdown("---")
        st.info(f"🗨️ Input: \"{user_input}\"")
        st.success("✅ Both models analyzed successfully!")

# ----------------------------------------
# FOOTER
# ----------------------------------------
st.markdown("""
<hr>
<p style='text-align:center; font-size:15px;'>
DistilBERT achieves higher contextual accuracy than Logistic Regression for toxic comment detection.
Built by Abbu Bucker, Chandana and Arunima.
</p>
""", unsafe_allow_html=True)




