# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Import XGBoost classifier
import xgboost as xgb
from xgboost import XGBClassifier

# Define a custom class for XGBoost classification
class XGBoostClassifier:
    def __init__(self, learning_rate=0.1, n_estimators=100, max_depth=3):
        """
        Constructor to initialize the XGBoost classifier with chosen hyperparameters.

        :param learning_rate: Step size shrinkage used to prevent overfitting (default is 0.1).
        :param n_estimators: Number of boosting rounds / trees to build (default is 100).
        :param max_depth: Maximum tree depth for base learners (default is 3).
        """
        self.model = XGBClassifier(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth
        )
    
    def train_model(self, X_train, y_train):
        """
        Train the XGBoost model using training data.

        :param X_train: Feature matrix for training.
        :param y_train: Target labels for training.
        """
        self.model.fit(X_train, y_train)
    
    def predict_data(self, X):
        """
        Make predictions on the input data using the trained model.

        :param X: Feature matrix for prediction (e.g., test data).
        :return: Predicted labels.
        """
        return self.model.predict(X)
    
    def evaluate_xgboost_model(self, X_test, y_test):
        """
        Evaluate the model performance on test data.
        Outputs accuracy, precision, recall, F1-score, classification report, and confusion matrix heatmap.

        :param X_test: Feature matrix for testing.
        :param y_test: True labels for testing.
        """
        # Get predictions
        predictions = self.predict_data(X_test)

        # Compute performance metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='macro')
        recall = recall_score(y_test, predictions, average='macro')
        f1 = f1_score(y_test, predictions, average='macro')

        # Print metrics
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
        
        # Print detailed classification report
        print("\nDetailed Classification Report:")
        print(classification_report(y_test, predictions))

        # Generate and visualize confusion matrix
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=set(y_test), yticklabels=set(y_test))
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix for XGBoost Model")
        plt.show()
