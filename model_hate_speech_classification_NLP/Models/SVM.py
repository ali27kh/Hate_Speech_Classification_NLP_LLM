# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import SVM model and evaluation metrics from scikit-learn
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Define a custom class to encapsulate the SVM model and related methods
class SVMClassifier:
    def __init__(self, kernel='linear'):
        """
        Constructor for the SVMClassifier class.
        Initializes the SVM model with the specified kernel (default is 'linear'),
        and prepares lists to store training and validation metrics.
        """
        self.model = SVC(kernel=kernel)
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

    def train_model(self, X_train, y_train):
        """
        Fit the SVM model on the training data.
        """
        self.model.fit(X_train, y_train)

    def predict_data(self, X):
        """
        Predict labels for the input data X using the trained model.
        """
        return self.model.predict(X)

    def plot_metrics(self):
        """
        Plot training and validation loss and accuracy curves.
        """
        if not self.train_losses:
            raise ValueError("No metrics available to plot. Train the model first!")

        # Plot training and validation loss
        plt.figure(figsize=(12, 6))
        plt.plot(self.train_losses, label='Training Loss', color='blue', linewidth=2)
        plt.plot(self.val_losses, label='Validation Loss', color='red', linewidth=2)
        plt.xlabel('Step')
        plt.ylabel('Loss Value')
        plt.title('Loss Over Training Steps')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()

        # Plot training and validation accuracy
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
        """
        Evaluate the SVM model on test data and display:
        - Accuracy, precision, recall, and F1-score
        - Full classification report
        - F1 score bar plot per class
        - Confusion matrix heatmap
        """
        # Make predictions on test data
        predictions = self.predict_data(X_test)

        # Compute evaluation metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='macro')
        recall = recall_score(y_test, predictions, average='macro')
        f1 = f1_score(y_test, predictions, average='macro')

        # Display basic metrics
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
        print("\nDetailed Classification Report:")
        report = classification_report(y_test, predictions, output_dict=True)
        print(classification_report(y_test, predictions))

        # Extract F1 scores for each class
        classes = [str(key) for key in report.keys() if key not in ['accuracy', 'macro avg', 'weighted avg']]
        f1_scores = [report[key]['f1-score'] for key in classes]

        # Define a color for each class using seaborn's husl palette
        colors = sns.color_palette("husl", len(classes))

        # Plot F1 scores in a bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(classes, f1_scores, color=colors, edgecolor='black')
        plt.xlabel('Class')
        plt.ylabel('F1 Score')
        plt.title('F1 Score per Class')
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig('f1_scores_per_class.png')  # Save plot to a file
        plt.show()

        # Plot confusion matrix
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                    xticklabels=set(y_test), yticklabels=set(y_test))
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix for SVM Model")
        plt.show()
