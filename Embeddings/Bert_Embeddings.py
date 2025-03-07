import torch
from transformers import BertTokenizer, BertModel
import numpy as np
import pandas as pd

class BertEmbedding:
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
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def generate_bert_embeddings(self, max_length=128):
        """
        Génère les embeddings BERT pour les ensembles train, val et test.
        
        Args:
            max_length (int): Longueur maximale des séquences (défaut : 128).
        
        Returns:
            tuple: (tokenizer, embeddings_train, embeddings_val, embeddings_test)
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
                    # Prendre l'embedding du token [CLS]
                    cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    embeddings.append(cls_embedding[0])
            return np.array(embeddings)

        embeddings_train = get_embeddings(self.train_comments)
        embeddings_val = get_embeddings(self.val_comments)
        embeddings_test = get_embeddings(self.test_comments)

        return self.tokenizer, embeddings_train, embeddings_val, embeddings_test
    
    def create_dataset(self, encodings, labels):

      dataset = pd.DataFrame({
         "comment_embedding": encodings,
         "label": labels
      })
      return dataset
