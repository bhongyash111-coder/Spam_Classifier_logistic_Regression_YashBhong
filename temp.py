# spam_classifier_app.py
"""
Streamlit Spam Classifier — ready-to-run single-file app.

Features:
- Optional file uploader (checkbox in sidebar). If user uploads a CSV it will be used.
- If no upload, it will try to load mail_data.csv from the app folder (good for Streamlit Cloud / repo deployments).
- If mail_data.csv is not present, it falls back to a small built-in demo dataset so the UI always works.
- Trains a LogisticRegression on TF-IDF features.
- Saves trained model + vectorizer (model.joblib, vect.joblib) so subsequent runs in the same session load the cached artifacts instead of retraining.
- Animated gradient background (soft tones, non-red) and a clean card-style layout.
- Automatically detects common column names for message/text and label/category. Falls back to first two columns if detection fails.
- Signature: "Made with ❤️ by Yash Bhong"

Usage:
1) Put this file in your project folder.
2) (Optional) Add your mail_data.csv to the same folder (columns like 'Category' and 'Message').
3) Create requirements.txt (see below), push to GitHub and deploy to Streamlit Cloud, or run locally with:

    streamlit run spam_classifier_app.py

Notes about deployment:
- For Streamlit Cloud, add mail_data.csv to the repository root if you want the app to use your dataset automatically.
- If you want to avoid retraining on every cold start you can pre-train locally and include the saved model.joblib & vect.joblib in the repo — the app will load them.

Included (at bottom of this file) there's a small snippet showing the lines to place into requirements.txt and runtime.txt.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import os

# ---------------- Page config & CSS ----------------
st.set_page_config(page_title="Spam Classifier", page_icon="📧", layout="centered")

page_bg = """
<style>
@keyframes gradient {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
.stApp {
    background: linear-gradient(-45deg, #e0f7fa, #e8f5e9, #f3e5f5, #fffde7);
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
}
.card {
    background: rgba(255,255,255,0.80);
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 6px 30px rgba(0,0,0,0.06);
}
h1 { color: #0b3954; }
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.title("About")
    st.write("Interactive Spam classifier — paste a message and get prediction. Optionally upload your CSV.")
    st.markdown("---")
    st.write("Dataset options:")
    use_uploader = st.checkbox("Show file uploader", value=True)
    show_model_details = st.checkbox("Show model details (probabilities & raw output)", value=True)
    st.markdown("---")
    st.write("Made with ❤️ by **Yash Bhong**")
    st.markdown("---")
    st.write("Tip: For deployment push `mail_data.csv` to the repo root so the app can auto-load it.")

# ---------------- Dataset loading helpers ----------------
DEFAULT_CSV = "mail_data.csv"

# Try to auto-detect sensible text / label column names
def detect_columns(df):
    cols = [c.lower() for c in df.columns]
    text_col = None
    label_col = None
    for candidate in ["message","text","content","body","message_text","sms","msg"]:
        if candidate in cols:
            text_col = df.columns[cols.index(candidate)]
            break
    for candidate in ["category","label","class","target","type"]:
        if candidate in cols:
            label_col = df.columns[cols.index(candidate)]
            break
    return text_col, label_col

def load_default_df():
    # prefer mail_data.csv if present in working dir (suitable for repo deployments)
    if os.path.exists(DEFAULT_CSV):
        try:
            return pd.read_csv(DEFAULT_CSV)
        except Exception as e:
            st.warning(f"Found {DEFAULT_CSV} but couldn't read it: {e}. Falling back to built-in sample.")
    # fallback small demo dataset so the app always works
    data = {
        "Category": ["ham","ham","spam","spam","ham","spam"],
        "Message": [
            "Hey, are we still meeting for coffee tomorrow?",
            "Don't forget the meeting at 10am.",
            "Congratulations! You have won a $1000 gift card. Click here to claim.",
            "Free entry in a weekly competition. Reply to claim.",
            "Can you review my PR?",
            "You have been selected for a cash prize! Visit link."
        ]
    }
    return pd.DataFrame(data)

# ---------------- Load Data (uploader OR default) ----------------
uploaded_file = None
if use_uploader:
    uploaded_file = st.file_uploader("Upload CSV (must contain a label and a message/text column)", type=["csv"]) 

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Couldn't read uploaded file: {e}")
        df = load_default_df()
else:
    df = load_default_df()

# ---------------- Ensure correct columns ----------------
text_col, label_col = detect_columns(df)
if text_col is None or label_col is None:
    st.warning("Couldn't find expected text/label columns automatically. The app will use the first two columns as fallback.\n"
                "(Prefer columns named 'Message' and 'Category' or similar.)")
    if df.shape[1] >= 2:
        text_col = df.columns[0]
        label_col = df.columns[1]
    else:
        st.error("Dataset doesn't have enough columns. The built-in demo will be used instead.")
        df = load_default_df()
        text_col, label_col = detect_columns(df)

# Normalize and rename
df = df[[text_col, label_col]].rename(columns={text_col: "Message", label_col: "Category"})
# Make sure category is string
df['Category'] = df['Category'].astype(str).str.lower().str.strip()
# Map to binary label: spam->0, ham->1
mapping = {'spam':0, 'ham':1}
# If dataset uses other encodings (1/0, true/false), try to be forgiving
if set(df['Category'].unique()) <= {"spam","ham"}:
    df['label'] = df['Category'].map(mapping)
else:
    # try numeric conversion
    try:
        df['label'] = pd.to_numeric(df['Category'])
        # if it's 0/1 assume 1 == ham; convert if necessary
        if set(df['label'].unique()) <= {0,1}:
            # keep as-is
            pass
        else:
            # last fallback: simple text check
            df['label'] = df['Category'].apply(lambda x: 0 if 'spam' in x else (1 if 'ham' in x else np.nan))
    except Exception:
        df['label'] = df['Category'].apply(lambda x: 0 if 'spam' in x else (1 if 'ham' in x else np.nan))

# drop rows missing message or label
df = df.dropna(subset=['Message','label'])
# ensure label is int
df['label'] = df['label'].astype(int)

# ---------------- Show preview ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("### 📊 Dataset Preview")
st.write(df.head())
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- Training / Loading model ----------------
MODEL_FILE = "model.joblib"
VECT_FILE = "vect.joblib"

def train_and_save(df):
    X = df['Message']
    Y = df['label']
    # keep stratify for balanced split if possible
    stratify = Y if len(set(Y)) > 1 else None
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=stratify)
    vect = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    X_train_features = vect.fit_transform(X_train)
    X_test_features = vect.transform(X_test)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_features, Y_train)
    train_acc = accuracy_score(Y_train, clf.predict(X_train_features))
    test_acc = accuracy_score(Y_test, clf.predict(X_test_features))
    # persist artifacts
    joblib.dump(clf, MODEL_FILE)
    joblib.dump(vect, VECT_FILE)
    return clf, vect, train_acc, test_acc

model = None
vectorizer = None
train_acc = test_acc = 0.0

# If user uploaded a file we retrain right away (so they see results for their data).
# Otherwise if cached model exists we load it for fast startup.
if uploaded_file is not None:
    with st.spinner("Training model on uploaded dataset..."):
        model, vectorizer, train_acc, test_acc = train_and_save(df)
else:
    if os.path.exists(MODEL_FILE) and os.path.exists(VECT_FILE):
        try:
            model = joblib.load(MODEL_FILE)
            vectorizer = joblib.load(VECT_FILE)
            # compute approximate accuracy on current df if you want (optional)
            # but to keep startup fast we won't recompute unless needed
            st.success("Loaded cached model & vectorizer.")
            # quick check: if df size is tiny or labels changed, optionally retrain
        except Exception as e:
            st.warning("Cached model couldn't be loaded. Retraining...")
            with st.spinner("Training model..."):
                model, vectorizer, train_acc, test_acc = train_and_save(df)
    else:
        with st.spinner("Training model (this runs once) ..."):
            model, vectorizer, train_acc, test_acc = train_and_save(df)

# ---------------- Display performance ----------------
st.markdown("### 📈 Model Performance")
col1, col2 = st.columns(2)
col1.metric("Training Accuracy", f"{train_acc:.2f}")
col2.metric("Test Accuracy", f"{test_acc:.2f}")

# ---------------- User input / prediction ----------------
st.subheader("✉️ Paste a message to classify")
user_input = st.text_area("Enter message here...", height=140)

if st.button("🔎 Classify"):
    if not user_input.strip():
        st.warning("Please paste a message to classify.")
    else:
        X_new = vectorizer.transform([user_input])
        proba = model.predict_proba(X_new)[0]
        pred = model.predict(X_new)[0]
        if pred == 1:
            st.success("✅ This is a **Ham (Not Spam)** message.")
        else:
            st.error("🚨 This is a **Spam** message.")
        if show_model_details:
            st.write(f"Prediction probability — Ham: {proba[1]:.3f}, Spam: {proba[0]:.3f}")

st.markdown("---")
st.markdown("<p style='text-align:center;color: #0b3954;'>✨ Made with ❤️ by <b>Yash Bhong</b></p>", unsafe_allow_html=True)

# ---------------- requirements + runtime (paste to separate files) ----------------
# requirements.txt content (paste into a file named requirements.txt):
# streamlit
# pandas
# scikit-learn
# joblib

# runtime.txt content (paste into a file named runtime.txt):
# python-3.11

# Optional: If you'd like me to create the repo structure (spam_classifier_app.py, requirements.txt, runtime.txt, sample mail_data.csv)
# I can create them for you — tell me and I'll prepare the files in this conversation.
