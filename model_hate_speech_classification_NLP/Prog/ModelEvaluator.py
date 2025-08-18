import pandas as pd
from sklearn.metrics import accuracy_score
import torch

class ModelEvaluator:
    def __init__(self, models, model_instances, X_val, X_val_lstm, val_inputs_tensor, y_val):
        self.models = models
        self.model_instances = model_instances
        self.X_val = X_val
        self.X_val_lstm = X_val_lstm
        self.val_inputs_tensor = val_inputs_tensor
        self.y_val = y_val
        self.accuracies = {}
    
    def get_accuracy(self, model):
        model_instance = self.model_instances.get(model)
        if model == 'hard_voting':
            y_pred = model_instance.predict(self.X_val, self.X_val_lstm, self.val_inputs_tensor)
        elif model == 'svm':
            y_pred = model_instance.predict_data(self.X_val)
        elif model == 'knn':
            y_pred = model_instance.predict_data(self.X_val)
        elif model == 'lstm':
            y_pred = model_instance.predict(self.X_val_lstm)
            y_pred = (y_pred > 0.5).astype(int)  # Assuming binary classification
        elif model == 'cnn':
            with torch.no_grad():
                cnn_preds_raw = model_instance(self.val_inputs_tensor)
                y_pred = torch.argmax(torch.softmax(cnn_preds_raw, dim=1), dim=1).cpu().numpy()
        elif model == 'xgboost':
            y_pred = model_instance.predict_data(self.X_val)
        
        return accuracy_score(self.y_val, y_pred)
    
    def evaluate_models(self):
        for model in self.models:
            self.accuracies[model] = self.get_accuracy(model)
        
    def display_results(self):
        accuracy_df = pd.DataFrame(list(self.accuracies.items()), columns=['Model', 'Accuracy'])
        print(accuracy_df)
        return accuracy_df

