import numpy as np
import torch
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class HardVotingClassifier(BaseEstimator):
    def __init__(self, svm_model, knn_model, nn_model, cnn_model):
        self.svm_model = svm_model
        self.knn_model = knn_model
        self.nn_model = nn_model  # Neural Network Classifier
        self.cnn_model = cnn_model

    def fit(self, X, y):
        # This method is for compatibility with sklearn's model selection tools
        pass

    def predict(self, X_val, val_inputs_tensor):
        # Predict with SVM
        svm_preds = self.svm_model.predict_data(X_val).reshape(-1)

        # Predict with KNN
        knn_preds = self.knn_model.predict_data(X_val).reshape(-1)

        # Predict with NNClassifier (assuming it has predict method)
        nn_preds = self.nn_model.predict_data(X_val).reshape(-1)

        # Predict with CNN
        with torch.no_grad():
            cnn_preds_raw = self.cnn_model(val_inputs_tensor)
            cnn_preds = torch.argmax(torch.softmax(cnn_preds_raw, dim=1), dim=1).cpu().numpy().reshape(-1)

        # Combine all predictions for hard voting
        all_preds = np.vstack([svm_preds, knn_preds, nn_preds, cnn_preds])
        final_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=all_preds)

        return final_preds
    
    def evaluate(self, X_val, val_inputs_tensor, y_val):
        # Get predictions using the hard voting model
        y_pred = self.predict(X_val, val_inputs_tensor)

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
