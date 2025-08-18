import numpy as np
import torch
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Hard Voting Classifier combining SVM, KNN, Neural Network, and CNN models
class HardVotingClassifier(BaseEstimator):
    def __init__(self, svm_model, knn_model, nn_model, cnn_model):
        # Initialize with the four models
        self.svm_model = svm_model
        self.knn_model = knn_model
        self.nn_model = nn_model
        self.cnn_model = cnn_model

    def fit(self, X, y):
        # Placeholder method for compatibility with sklearn's API
        pass

    def predict(self, X_val, val_inputs_tensor):
        # Get predictions from each model
        svm_preds = self.svm_model.predict_data(X_val).reshape(-1)
        knn_preds = self.knn_model.predict_data(X_val).reshape(-1)
        nn_preds = self.nn_model.predict_data(X_val).reshape(-1)

        # CNN predictions using softmax and argmax
        with torch.no_grad():
            cnn_preds_raw = self.cnn_model(val_inputs_tensor)
            cnn_preds = torch.argmax(torch.softmax(cnn_preds_raw, dim=1), dim=1).cpu().numpy().reshape(-1)

        # Perform hard voting by selecting the most common prediction
        all_preds = np.vstack([svm_preds, knn_preds, nn_preds, cnn_preds])
        final_preds = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=all_preds)

        return final_preds
    
    def evaluate(self, X_val, val_inputs_tensor, y_val):
        # Generate predictions using the voting classifier
        y_pred = self.predict(X_val, val_inputs_tensor)

        # Display classification report with precision, recall, and F1-score
        print("Classification Report:")
        print(classification_report(y_val, y_pred))

        # Compute and visualize confusion matrix
        conf_matrix = confusion_matrix(y_val, y_pred)
        print("\nConfusion Matrix:")
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_val), yticklabels=np.unique(y_val))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()