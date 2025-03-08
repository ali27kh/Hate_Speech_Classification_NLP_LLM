import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import xgboost as xgb
from xgboost import XGBClassifier

class XGBoostClassifier:
    def __init__(self, learning_rate=0.1, n_estimators=100, max_depth=3):
        """
        Initialize XGBoost model with hyperparameters.

        :param learning_rate: Step size for optimization (default 0.1).
        :param n_estimators: Number of boosting rounds (default 100).
        :param max_depth: Maximum depth of the tree (default 3).
        """
        self.model = XGBClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)
    
    def train_model(self, X_train, y_train):
        """
        Train the XGBoost model.

        :param X_train: Training data features.
        :param y_train: Training data labels.
        """
        self.model.fit(X_train, y_train)
    
    def predict_data(self, X):
        """
        Predict using the trained XGBoost model.

        :param X: Test data features.
        :return: Predicted labels.
        """
        return self.model.predict(X)
    
    def evaluate_xgboost_model(self, X_test, y_test):
        """
        Evaluate the XGBoost model and print metrics.

        :param X_test: Test data features.
        :param y_test: Test data labels.
        """
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
        plt.title("Confusion Matrix for XGBoost Model")
        plt.show()
