import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

class SVMClassifier:
    def __init__(self, kernel='linear'):
        self.model = SVC(kernel=kernel)
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict_data(self, X):
        return self.model.predict(X)
    
    def plot_metrics(self):
        """
        Plot training and validation metrics stored in the class.
        """
        if not self.train_losses:
            raise ValueError("No metrics available to plot. Train the model first!")
        
        # Plot Losses
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_losses, label='Training Loss', color='blue', linewidth=2)
        plt.plot(self.val_losses, label='Validation Loss', color='red', linewidth=2)
        plt.xlabel('Step')
        plt.ylabel('Loss Value')
        plt.title('Loss Over Training Steps')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()

        # Plot Accuracies
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_accuracies, label='Training Accuracy', color='green', linestyle='--', linewidth=2)
        plt.plot(self.val_accuracies, label='Validation Accuracy', color='orange', linestyle='--', linewidth=2)
        plt.xlabel('Step')
        plt.ylabel('Accuracy Score')
        plt.title('Accuracy Over Training Steps')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()
    
    def evaluate_svm_model(self, X_test, y_test):
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
        plt.title("Confusion Matrix for SVM Model")
        plt.show()