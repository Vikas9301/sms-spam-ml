import streamlit as st
import pandas as pd
import string
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Load and preprocess the data
@st.cache_data
def load_data():
    df = pd.read_csv("spam.csv", encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

def train_model(data):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', MultinomialNB())
    ])
    pipeline.fit(data['message'], data['label'])
    return pipeline

# Load data and train model
st.title("📩 SMS Spam Detector")

st.markdown("""
Enter a text message below to check if it's spam or not. 
The model is trained using the Naive Bayes algorithm.
""")

with st.spinner("Loading data and training model..."):
    df = load_data()
    model = train_model(df)

# User input
user_input = st.text_area("Type your message here:")

if st.button("Predict"):
    prediction = model.predict([user_input])[0]
    label = "🚫 Spam" if prediction == 1 else "✅ Not Spam"
    st.subheader("Prediction:")
    st.success(label)

    # Display probability
    prob = model.predict_proba([user_input])[0][prediction]
    st.write(f"Confidence: {prob:.2f}")

# Optional: show data
if st.checkbox("Show raw data"):
    st.dataframe(df.head(10))
