import numpy as np
import torch
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class HardVotingClassifier(BaseEstimator):
    def __init__(self, svm_model, knn_model, lstm_model, cnn_model, xgboost_model, lstm_threshold=0.5):
        self.svm_model = svm_model
        self.knn_model = knn_model
        self.lstm_model = lstm_model
        self.cnn_model = cnn_model
        self.xgboost_model = xgboost_model
        self.lstm_threshold = lstm_threshold

    def fit(self, X, y):
        # This method is for compatibility with sklearn's model selection tools
        pass

    def predict(self, X_val, X_val_lstm, val_inputs_tensor):
        # Predict with SVM
        svm_preds = self.svm_model.predict_data(X_val).reshape(-1)

        # Predict with KNN
        knn_preds = self.knn_model.predict_data(X_val).reshape(-1)

        # Predict with LSTM (binary classification assumption)
        lstm_preds = self.lstm_model.predict(X_val_lstm)
        lstm_preds_binary = (lstm_preds > self.lstm_threshold).astype(int).reshape(-1)

        # Predict with CNN
        with torch.no_grad():
            cnn_preds_raw = self.cnn_model(val_inputs_tensor)
            cnn_preds = torch.argmax(torch.softmax(cnn_preds_raw, dim=1), dim=1).cpu().numpy().reshape(-1)

        # Predict with XGBoost
        xgboost_preds = self.xgboost_model.predict_data(X_val).reshape(-1)

        # Combine all predictions for hard voting
        all_preds = np.vstack([svm_preds, knn_preds, lstm_preds_binary, cnn_preds, xgboost_preds])
        final_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=all_preds)

        return final_preds
    
    def evaluate(self, X_val, X_val_lstm, val_inputs_tensor, y_val):
        # Get predictions using the hard voting model
        y_pred = self.predict(X_val, X_val_lstm, val_inputs_tensor)

        # Classification report
        print("Classification Report:")
        print(classification_report(y_val, y_pred))

        # Confusion matrix
        conf_matrix = confusion_matrix(y_val, y_pred)
        print("\nConfusion Matrix:")

        # Plot the confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_val), yticklabels=np.unique(y_val))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
