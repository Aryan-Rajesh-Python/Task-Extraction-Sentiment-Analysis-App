# 🚀 Task Extraction & Sentiment Analysis Streamlit App

## 📌 Overview
This is a **Streamlit-based NLP & Machine Learning web app** that performs:
1. **Task Extraction** → Identifies tasks from unstructured text, extracts "who" has to do the task, deadlines, and categorizes the task.
2. **Sentiment Analysis** → Classifies user-input text (e.g., movie reviews) as **Positive or Negative** using a trained ML model.

---

## 📂 Project Structure
- **`task_sentiment_streamlit.py`** → Main Streamlit app combining both functionalities.
- **`imdb.csv`** → Dataset used for training the sentiment analysis model.
- **`requirements.txt`** → List of dependencies to install.

---

## 🛠 Installation & Setup
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Aryan-Rajesh-Python/Task-Extraction-Sentiment-Analysis-App
.git
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Download NLP Model (spaCy)
```bash
python -m spacy download en_core_web_sm
```

### 4️⃣ Run the Streamlit App
```bash
streamlit run app.py
```

---

## 📝 Features
### ✅ **Task Extraction**
- Extracts **who is responsible**, **what the task is**, and **when it needs to be done**.
- Uses **spaCy Named Entity Recognition (NER)** and **regex-based deadline extraction**.
- Categorizes tasks into predefined types (e.g., Personal, Work, Academic, etc.).

### ✅ **Sentiment Analysis**
- Uses **TF-IDF vectorization** to extract text features.
- Trained **Logistic Regression model** for classification.
- Predicts **Positive or Negative** sentiment from user-input text.

---

## 📊 Expected Output
### **Task Extraction Example:**
**Input:**
```
Rahul needs to submit his assignment by 5 PM today.
```
**Output (JSON):**
```json
[
  {
    "who": "Rahul",
    "task": "submit his assignment",
    "deadline": "5 PM today",
    "category": "Academic"
  }
]
```

### **Sentiment Analysis Example:**
**Input:**
```
This movie was absolutely fantastic! I loved every second of it.
```
**Output:**
```
Predicted Sentiment: Positive
```

---

## 📌 Notes
- Make sure `imdb.csv` is present in the project directory.
- The first time you run the app, it may take a few seconds to train the sentiment model.
- If Streamlit does not auto-open, go to `localhost:8501` in your browser.

---

## 🎯 Future Enhancements
- Improve task categorization using **Transformer-based models (e.g., BERT)**.
- Replace **TF-IDF + Logistic Regression** with **Deep Learning models** for better sentiment accuracy.
- Add a **database** to store extracted tasks and user inputs.
