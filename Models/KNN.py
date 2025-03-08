import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

class KNNClassifier:
    def __init__(self, n_neighbors=2):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict_data(self, X):
        return self.model.predict(X)
    
    def evaluate_knn_model(self, X_test, y_test):
        predictions = self.predict_data(X_test)
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='macro')
        recall = recall_score(y_test, predictions, average='macro')
        f1 = f1_score(y_test, predictions, average='macro')
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, predictions))
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=set(y_test), yticklabels=set(y_test))
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix for KNN Model")
        plt.show()
