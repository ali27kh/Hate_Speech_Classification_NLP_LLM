
import shap
import torch

def prepare_data_for_shap(traindata, index=0):
    '''
    Extract a single instance and convert it to tensor format suitable for the model.
    '''
    instance = torch.tensor(traindata['comment_embedding'].iloc[index], dtype=torch.float32)
    background = torch.stack([torch.tensor(emb, dtype=torch.float32) for emb in traindata['comment_embedding'].iloc[:100]])
    return background, instance.unsqueeze(0)

class WrappedModel(torch.nn.Module):
    def __init__(self, model, class_index=1):
        super().__init__()
        self.model = model
        self.class_index = class_index
    
    def forward(self, x):
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.nn.functional.softmax(logits, dim=1)
            return probs[:, self.class_index]

def explain_with_shap(model, traindata, index=0):
    '''
    Explains the prediction of an LSTM model using SHAP.
    '''
    background, instance = prepare_data_for_shap(traindata, index)
    wrapped_model = WrappedModel(model)
    wrapped_model.eval()

    explainer = shap.DeepExplainer(wrapped_model, background)
    shap_values = explainer.shap_values(instance)
    return shap_values, instance
