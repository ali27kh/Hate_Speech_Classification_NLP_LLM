import torch
from transformers import DebertaTokenizer, DebertaModel
import numpy as np
import pandas as pd

class DebertaEmbedding:
    def __init__(self, train_comments, test_comments, val_comments):
        """
        Initialise la classe avec les commentaires d'entraînement, de test et de validation.
        
        Args:
            train_comments (list or pd.Series): Commentaires pour l'entraînement.
            test_comments (list or pd.Series): Commentaires pour le test.
            val_comments (list or pd.Series): Commentaires pour la validation.
        """
        self.train_comments = train_comments
        self.test_comments = test_comments
        self.val_comments = val_comments
        self.tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
        self.model = DebertaModel.from_pretrained('microsoft/deberta-base')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def generate_deberta_embeddings(self, max_length=128):
        """
        Génère les embeddings DeBERTa pour les ensembles train, val et test.
        
        Args:
            max_length (int): Longueur maximale des séquences (défaut : 128).
        
        Returns:
            tuple: (tokenizer, embeddings_train, embeddings_val, embeddings_test)
                   Each embeddings_* is a NumPy array of [CLS] embeddings.
        """
        def get_embeddings(comments):
            embeddings = []
            self.model.eval()
            with torch.no_grad():
                for comment in comments:
                    inputs = self.tokenizer(comment, return_tensors='pt', truncation=True, 
                                            padding=True, max_length=max_length)
                    inputs = {key: val.to(self.device) for key, val in inputs.items()}
                    outputs = self.model(**inputs)
                    # Prendre l'embedding du token [CLS] (premier token)
                    cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    embeddings.append(cls_embedding[0])
            return np.array(embeddings)

        embeddings_train = get_embeddings(self.train_comments)
        embeddings_val = get_embeddings(self.val_comments)
        embeddings_test = get_embeddings(self.test_comments)

        return self.tokenizer, embeddings_train, embeddings_val, embeddings_test
    
    def create_dataset(self, encodings, labels):
        """
        Crée un DataFrame à partir des embeddings et des labels.
        
        Args:
            encodings (np.array): Embeddings générés (e.g., embeddings_train).
            labels (list or np.array): Labels correspondants.
        
        Returns:
            pd.DataFrame: DataFrame avec colonnes 'comment_embedding' et 'label'.
        """
        dataset = pd.DataFrame({
            "comment_embedding": list(encodings),  # Convertir chaque embedding en liste pour DataFrame
            "label": labels
        })
        return dataset
