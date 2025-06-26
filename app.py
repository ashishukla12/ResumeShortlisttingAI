# app.py

import fitz
import pandas as pd
import numpy as np
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import gradio as gr
import joblib

nltk.download('punkt')
nltk.download('stopwords')

# Load your labeled dataset
df = pd.read_csv("resume_dataset.csv")

# Preprocess
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    cleaned = [w for w in words if w not in stop_words]
    return ' '.join(cleaned)

df['cleaned_text'] = df['resume_text'].apply(clean_text)

# TF-IDF + Model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['label']

model = RandomForestClassifier()
model.fit(X, y)

# Prediction function
def predict_resume(pdf_file):
    doc = fitz.open(pdf_file.name)
    text = ""
    for page in doc:
        text += page.get_text()
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    return "✅ Selected" if prediction == 1 else "❌ Rejected"

# Gradio app
demo = gr.Interface(
    fn=predict_resume,
    inputs=gr.File(label="Upload Resume (PDF)"),
    outputs=gr.Textbox(label="Prediction"),
    title="Resume Shortlister AI",
    description="Upload a resume to check if it would be shortlisted for ML roles."
)

demo.launch()
