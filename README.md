# 🚫 Hate Speech Detection Using NLP Techniques

## 📌 Project Overview
This project detects hate speech in text using **NLP techniques** and **deep learning models**. It processes raw text (including spelling errors), cleans data, augments with paraphrasing, trains classification models (e.g., LSTM), and deploys a **Django interface** for real-time prediction. The LSTM model is selected for its superior performance in classifying hate speech.

---

## 📂 Dataset
- **Custom Dataset**: Collected text data with labels (hate speech or not), stored in a local CSV file.
- Input: Raw text with possible errors; Output: Binary classification (0/1) and the type of hate speech.

---

## 🔍 Project Workflow

### **1. Data Cleaning**
Preprocess raw text by removing links, special characters, emojis, and correcting grammar.

```python
import re
import pandas as pd
import string
import nltk
import language_tool_python

tool = language_tool_python.LanguageTool('en-US')

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove links
    text = re.sub(r'[^\w\s]', '', text)  # Remove special chars
    text = re.sub(u"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF]+", '', text)  # Remove emojis
    text = tool.correct(text)  # Correct grammar
    return text
```

### **2. Data Augmentation**
Paraphrase sentences using Pegasus to expand the dataset.

```python
from transformers import PegasusTokenizer, PegasusForConditionalGeneration

model_name = 'tuner007/pegasus_paraphrase'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)

def get_response(input_text, num_return_sequences=3, num_beams=10):
    batch = tokenizer([input_text], truncation=True, padding='longest', max_length=60, return_tensors="pt")
    translated = model.generate(**batch, max_length=60, num_beams=num_beams, num_return_sequences=num_return_sequences, temperature=1.5)
    return tokenizer.batch_decode(translated, skip_special_tokens=True)

# Augment DataFrame
new_comments = []
for comment in df['cleaned_comment']:
    if isinstance(comment, str) and comment.strip():
        paraphrases = get_response(comment)
        new_comments.extend(paraphrases)
```

### **3. Model Training**
Train multiple classification models; LSTM is selected as the best.

```python
# Models in 'Models' folder
# Example LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(128, input_shape=(timesteps, features)))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)
```

### **4. Model Testing with Django Interface**
Test the model via a Django web app, processing input text and classifying hate speech.

```python
# In Django views.py
from django.shortcuts import render
from .models import HateSpeechModel

def predict_hate_speech(request):
    if request.method == 'POST':
        text = request.POST['text']
        cleaned_text = preprocess_text(text)
        prediction = model.predict(cleaned_text)
        return render(request, 'result.html', {'prediction': prediction})
    return render(request, 'form.html')
```

---

## 📊 Results
- **Cleaning Visualization**:

  ![Cleaning](cleaning.png)
  
- **Models Comparison**:

  ![Models](models.png)
  
- **Steps**:

  ![Steps](steps.png)
  
- **Test Video**:

  https://github.com/user-attachments/assets/0f370dc7-e445-431e-b6bf-ddace48885c9

---

## 📦 Requirements
```bash
pip install pandas numpy transformers tensorflow matplotlib keras seaborn scikit-learn torch sentencepiece nltk spacy language-tool-python plotly nbformat wordcloud palettable textblob cufflinks lime shap tf-keras imblearn faiss-cpu accelerate xgboost openai shap ipython
```

---

## ▶️ How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/hate-speech-detection.git
   cd hate-speech-detection
   ```
2. Create and activate virtual environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run Django server:
   ```bash
   python manage.py runserver
   ```
   Access at `http://127.0.0.1:8000/` and input text for classification.

---

## 📌 Key Insights
- Data cleaning corrects errors and removes noise for reliable classification.
- Pegasus augmentation expands the dataset with paraphrased variations.
- LSTM excels in capturing sequential patterns in text for hate speech detection.
- Django interface enables real-time testing with preprocessing.

## Alternative Approaches
- **Falcon LLM**: A large language model for generating and classifying text, effective for nuanced hate speech detection.
- **DeBERTa**: Decoder-only BERT variant optimized for understanding text nuances in classification tasks.
- **BERT**: Bidirectional Encoder Representations from Transformers, pre-trained for contextual text understanding.
- **RoBERTa**: Robustly optimized BERT approach, improving performance on downstream NLP tasks like classification.

See `other_approaches/` for test details.

---

## 📜 License
MIT License
