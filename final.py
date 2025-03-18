import streamlit as st
import spacy
import re
import json
import pandas as pd
import nltk
import numpy as np
from collections import defaultdict
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Download stopwords if not available
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Task-related phrases
TASK_KEYWORDS = [
    "has to", "needs to", "should", "must", "is required to", "is expected to", 
    "is supposed to", "is scheduled to", "is assigned to"
]

# Deadline patterns
TIME_PATTERNS = [
    r'\bby\s+\d{1,2}\s*(am|pm)?\b',
    r'\bbefore\s+\w+\b',
    r'\btomorrow\b',
    r'\btoday\b',
    r'\bin\s+\d+\s+\w+\b',
    r'\bby end of the day\b',
    r'\bwithin\s+\d+\s+(hours|days|minutes)\b'
]

# Task categories
TASK_CATEGORIES = {
    "Personal": ["buy", "get", "shop", "visit"],
    "Academic": ["submit", "study", "complete", "assignment", "exam", "project"],
    "Work": ["send", "email", "call", "schedule", "meeting", "review"],
    "Household": ["clean", "wash", "cook", "arrange", "fix"],
    "Health": ["exercise", "run", "walk", "meditate"],
    "Finance": ["pay", "invest", "deposit", "withdraw", "budget"]
}

def extract_tasks(text):
    doc = nlp(text)
    extracted_tasks = []
    for sent in doc.sents:
        sentence = sent.text.strip()
        sentence_lower = sentence.lower()
        if any(keyword in sentence_lower for keyword in TASK_KEYWORDS):
            task = {"who": None, "task": None, "deadline": None, "category": "Uncategorized"}
            subjects = [token.text for token in sent if token.dep_ in {"nsubj", "nsubjpass"} and token.pos_ in {"PROPN", "PRON"}]
            task["who"] = ", ".join(subjects) if subjects else "Unknown"
            task_start = -1
            for keyword in TASK_KEYWORDS:
                if keyword in sentence_lower:
                    task_start = sentence_lower.find(keyword) + len(keyword)
                    break
            if task_start != -1:
                task_text = sentence[task_start:].strip()
                task["task"] = task_text
            for pattern in TIME_PATTERNS:
                match = re.search(pattern, sentence_lower)
                if match:
                    task["deadline"] = match.group(0)
                    break
            for ent in sent.ents:
                if ent.label_ in {"DATE", "TIME"}:
                    task["deadline"] = ent.text
                    break
            for category, keywords in TASK_CATEGORIES.items():
                if any(keyword in task["task"].lower() for keyword in keywords):
                    task["category"] = category
                    break
            if task["task"]:
                extracted_tasks.append(task)
    return extracted_tasks

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

@st.cache_data
def load_data():
    df = pd.read_csv("imdb.csv")
    df["sentiment"] = df["sentiment"].map({"positive": 1, "negative": 0})
    return df

def train_model():
    df = load_data()
    df["cleaned_review"] = df["review"].apply(preprocess_text)
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
    X = vectorizer.fit_transform(df["cleaned_review"])
    y = df["sentiment"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model, vectorizer

@st.cache_resource
def get_trained_model():
    return train_model()

st.title("📝 Task & Sentiment Analysis App")

st.header("Task Extraction")
user_task_input = st.text_area("Enter text for task extraction:", "")
if st.button("Extract Tasks"):
    if user_task_input.strip():
        tasks = extract_tasks(user_task_input)
        st.subheader("Extracted Tasks")
        st.json(tasks)
    else:
        st.warning("Please enter some text to process.")

st.header("Sentiment Analysis")
user_review_input = st.text_area("Enter a review for sentiment analysis:", "")
model, vectorizer = get_trained_model()
if st.button("Analyze Sentiment"):
    if user_review_input.strip():
        processed_input = preprocess_text(user_review_input)
        transformed_input = vectorizer.transform([processed_input])
        prediction = model.predict(transformed_input)
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        st.subheader(f"Predicted Sentiment: {sentiment}")
    else:
        st.warning("Please enter a review to analyze.")