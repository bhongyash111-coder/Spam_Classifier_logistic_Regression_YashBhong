import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


st.set_page_config(page_title="Spam Classifier", page_icon="📧", layout="centered")


page_bg = """
<style>
@keyframes gradient {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}
.stApp {
    background: linear-gradient(-45deg, #89f7fe, #66a6ff, #ffdde1, #ee9ca7);
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)


st.markdown(
    "<h1 style='text-align: center; color: #2c3e50;'>📧 Spam Message Classifier</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align: center; font-size:18px;'>Classify SMS/Emails as <b style='color:green;'>Ham</b> or <b style='color:#c0392b;'>Spam</b></p>",
    unsafe_allow_html=True,
)


df = pd.read_csv("mail_data.csv")


df['label'] = df['Category'].map({'spam': 0, 'ham': 1})
X = df['Message']
Y = df['label']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)


feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
X_train_features = feature_extraction.fit_transform(X_train)
X_test_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')


model = LogisticRegression()
model.fit(X_train_features, Y_train)


training_accuracy = accuracy_score(Y_train, model.predict(X_train_features))
test_accuracy = accuracy_score(Y_test, model.predict(X_test_features))


st.subheader("📈 Model Performance")
col1, col2 = st.columns(2)
col1.metric("Training Accuracy ✅", f"{training_accuracy:.2f}")
col2.metric("Test Accuracy 🎯", f"{test_accuracy:.2f}")


st.subheader("✉️ Try Your Own Message")
user_input = st.text_area("Enter a message to classify:", placeholder="Type your message here...")

if st.button("🔍 Classify Message"):
    if user_input.strip() != "":
        input_features = feature_extraction.transform([user_input])
        prediction = model.predict(input_features)[0]

        if prediction == 1:
            st.success("✅ This is a **Ham (Not Spam)** message.")
        else:
            st.error("🚨 This is a **Spam** message.")
    else:
        st.warning("⚠️ Please enter a message to classify.")


st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #2c3e50;'>✨ Made with ❤️ and ☕ by <b style='color:#2980b9;'>Yash Bhong</b></p>",
    unsafe_allow_html=True,
)
