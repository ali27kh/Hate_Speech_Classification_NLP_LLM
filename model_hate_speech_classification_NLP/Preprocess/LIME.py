import numpy as np
import tensorflow as tf
from lime import lime_tabular
from IPython.display import display, HTML

def lime_predict(model, x_np):
    """
    Prediction function for LIME
   
    """
    x_reshaped = x_np.reshape(-1, 1, x_np.shape[-1])
    
    # Convert to TensorFlow tensor
    x_tensor = tf.constant(x_reshaped, dtype=tf.float32)
    
    probs_class1 = model.predict(x_tensor)
    
    probs = np.hstack((1 - probs_class1, probs_class1))
    
    return probs

def predict_fn(model, x):
    """
    Prediction wrapper for LIME. Calls `lime_predict` with the provided model.
    
    Parameters:
        model: Keras model (e.g., LSTMClassifier instance).
        x (numpy.ndarray): Input data of shape (num_samples, input_size).
    
    Returns:
        numpy.ndarray: Predicted probabilities from the model.
    """
    return lime_predict(model, x)



def explain_instance(traindata, model, instance_idx, num_features=10, num_samples=5000, feature_names=None, class_names=['non-hate', 'hate']):
    """
    Generate and display a LIME explanation for a specific instance from the training data.
    
    Parameters:
        traindata (pd.DataFrame): DataFrame containing 'comment_embedding' (list of embeddings) and optionally 'comment' columns.
        model: Keras model (e.g., LSTMClassifier instance).
        instance_idx (int): Index of the instance in traindata to explain.
        num_features (int): Number of features to include in the explanation (default: 10).
        num_samples (int): Number of perturbed samples for LIME (default: 5000).
        feature_names (list): List of feature names for the embeddings (default: None, uses 'feature_0', 'feature_1', ...).
        class_names (list): Names of the classes (default: ['non-hate', 'hate'] for binary classification).
    
    Returns:
        lime.lime_tabular.LimeTabularExplainer.explanation: The LIME explanation object.
    
    Outputs:
        - Prints the comment (if available) and the explanation as a list.
        - Displays the explanation as an HTML table in a Jupyter notebook.
        - Shows a plot of the explanation (requires matplotlib).
    """
    # Prepare training embeddings
    train_embeddings = np.array(traindata['comment_embedding'].tolist())
    input_size = train_embeddings.shape[1]
    
    # Set feature names if not provided
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(input_size)]
    
    # Instantiate the LIME Tabular Explainer
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=train_embeddings,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification',
        discretize_continuous=True
    )
    
    # Choose the instance to explain
    if instance_idx < 0 or instance_idx >= len(train_embeddings):
        raise ValueError(f"instance_idx {instance_idx} is out of bounds for traindata with {len(train_embeddings)} samples.")
    instance = train_embeddings[instance_idx]
    
    # Create predict_fn for the model
    predict_function = lambda x: predict_fn(model, x)
    
    # Generate LIME explanation
    exp = explainer.explain_instance(
        data_row=instance,
        predict_fn=predict_function,
        num_features=num_features,
        num_samples=num_samples
    )
    
    # Print the associated comment if available
    if 'comment' in traindata.columns:
        print(f"Explained comment (index {instance_idx}): {traindata.iloc[instance_idx]['comment']}")
    
    # Print the explanation as a list
    print("Explanation for the chosen instance:")
    print(exp.as_list())
    
    # Display the explanation as HTML in a Jupyter notebook
    display(HTML(exp.as_html(show_table=True)))
    
    # Show the explanation plot (requires matplotlib)
    exp.show_in_notebook(show_table=True, show_all=False)
    
    return exp