from django.shortcuts import render
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
from transformers import  BertModel
import torch
import joblib
from .data_cleaning import DataCleaning 
import warnings
warnings.filterwarnings('ignore')

# Charger le modèle BERT (pour générer les embeddings)
bert_model = BertModel.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model.to(device)

# Charger le tokenizer sauvegardé
with open("./tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Charger le tokenizer sauvegardé
with open("./tokenizer_multi.pkl", "rb") as f:
    tokenizer_multi = pickle.load(f)

# Charger le modèle LSTM
lstm_model = load_model("./lstm_model.h5")

# Charger le modèle SVM avec gestion des erreurs
try:
    svm_model = joblib.load("./model.pkl")
    print("SVM model loaded successfully")
except Exception as e:
    print(f"Error loading SVM model: {e}")
    svm_model = None  # Définir à None pour éviter les plantages

# Dictionnaire pour mapper les valeurs SVM aux noms des classes
SVM_CLASSES = {
    0: "Directed vs Generalized",
    1: "Disability",
    2: "Gender",
    3: "National Origin",
    4: "Race",
    5: "Religion",
    6: "Sexual Orientation",
    7: "Violence"
}


def get_bert_embedding(comment, max_length=128):
    """Générer un embedding BERT pour un commentaire."""
    bert_model.eval()
    with torch.no_grad():
        inputs = tokenizer(comment, return_tensors='pt', truncation=True, 
                          padding=True, max_length=max_length)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        outputs = bert_model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Embedding [CLS]
    return cls_embedding


def predict_comment(comment):
    """Faire une prédiction avec le modèle LSTM et SVM si nécessaire."""
    # Générer l'embedding BERT
    embedding = get_bert_embedding(comment)
    
    # Prédiction LSTM
    embedding_tensor = tf.constant(embedding, dtype=tf.float32)
    if len(embedding_tensor.shape) == 2:
        embedding_tensor = tf.expand_dims(embedding_tensor, axis=1)  # (1, 1, 768)
    lstm_prediction = lstm_model.predict(embedding_tensor, verbose=0)
    lstm_result = "Positive" if lstm_prediction[0] > 0.5 else "Negative"
    print('LSTM Prediction:', lstm_result)

    # Si Positive (discours de haine), utiliser le modèle SVM
    svm_result = None
    if lstm_result == "Positive":
        svm_prediction = svm_model.predict(embedding)  # Prédiction multi-classes
        svm_result = SVM_CLASSES.get(svm_prediction[0], "Unknown")  # Mapper à la classe
        print('SVM Prediction:', svm_result)

    return lstm_result, svm_result

def home(request):
    lstm_prediction = None
    svm_prediction = None
    input_text = None  
    text = None
    
    if request.method == "POST":
        text = request.POST.get('text', '')
        if text:
            preprocessed_text = DataCleaning.preprocess_text(text)
            lstm_prediction, svm_prediction = predict_comment(preprocessed_text)
            input_text = preprocessed_text  
            
    
    context = {
        'lstm_prediction': lstm_prediction,
        'svm_prediction': svm_prediction,
        'input_text': input_text,
        'user_text':text
    }
    return render(request, 'index.html', context)
