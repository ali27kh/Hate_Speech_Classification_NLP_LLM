import numpy as np
import tensorflow as tf
import shap
import matplotlib.pyplot as plt

class SHAPExplainer:
    def __init__(self, traindata, model, background_size=100, random_seed=42, use_deep_explainer=True):
        """
        Initialize the SHAP explainer for a Keras LSTMClassifier.
        
        Parameters:
            traindata (pd.DataFrame): DataFrame with 'comment_embedding' column (list of embeddings).
            model: Keras model (e.g., LSTMClassifier instance).
            background_size (int): Number of background samples to use (default: 100).
            random_seed (int): Random seed for reproducibility (default: 42).
            use_deep_explainer (bool): Use DeepExplainer if True, else KernelExplainer (default: True).
        """
        self.traindata = traindata
        self.model = model
        self.background_size = background_size
        self.random_seed = random_seed
        self.use_deep_explainer = use_deep_explainer
        self.train_embeddings = np.array(traindata['comment_embedding'].tolist())
        self.input_size = self.train_embeddings.shape[1]  # e.g., 1536
        self.background_data = self._prepare_background()
        self.explainer = self._create_explainer()

    def _prepare_background(self):
        """
        Prepare background data for SHAP by sampling from train_embeddings.
        
        Returns:
            np.ndarray: Background data of shape (background_size, 1, input_size) for DeepExplainer,
                       or (background_size, input_size) for KernelExplainer.
        """
        np.random.seed(self.random_seed)
        background_indices = np.random.choice(
            self.train_embeddings.shape[0], self.background_size, replace=False
        )
        background_np = self.train_embeddings[background_indices]  # shape: (background_size, input_size)
        if self.use_deep_explainer:
            # DeepExplainer expects the same input shape as the model: (num_samples, 1, input_size)
            background_np = background_np.reshape(-1, 1, self.input_size)
        return background_np

    def _predict_fn(self, x):
        """
        Prediction function for SHAP, converting model output to [P(non-hate), P(hate)].
        
        Parameters:
            x (np.ndarray): Input data of shape (num_samples, input_size) or (num_samples, 1, input_size).
        
        Returns:
            np.ndarray: Probabilities for [non-hate, hate] of shape (num_samples, 2).
        """
        # Reshape input to (num_samples, 1, input_size) if needed
        if len(x.shape) == 2:
            x_reshaped = x.reshape(-1, 1, self.input_size)
        else:
            x_reshaped = x
        x_tensor = tf.constant(x_reshaped, dtype=tf.float32)
        
        # Get sigmoid probability for class_1 (hate)
        probs_class1 = self.model.predict(x_tensor, verbose=0)
        
        # Return [P(non-hate), P(hate)] = [1 - prob, prob]
        return np.hstack((1 - probs_class1, probs_class1))

    def _create_explainer(self):
        """
        Create a SHAP explainer for the model.
        
        Returns:
            shap.DeepExplainer or shap.KernelExplainer: SHAP explainer object.
        """
        if self.use_deep_explainer:
            # Use the Keras model directly (lstm_classifier.model)
            try:
                return shap.DeepExplainer(
                    model=self.model.model,  # Access the Keras model from LSTMClassifier
                    data=self.background_data
                )
            except Exception as e:
                print(f"DeepExplainer failed: {e}. Falling back to KernelExplainer.")
                self.use_deep_explainer = False
                self.background_data = self.background_data.reshape(self.background_size, self.input_size)
        
        # Use KernelExplainer as a fallback
        return shap.KernelExplainer(
            model=self._predict_fn,
            data=self.background_data,
            link="logit"
        )

    def explain_instance(self, instance_idx, class_idx=1, num_features=10, show_plot=True):
        """
        Generate and display a SHAP explanation for a specific instance.
        
        Parameters:
            instance_idx (int): Index of the instance in traindata to explain.
            class_idx (int): Class index to explain (0: non-hate, 1: hate; default: 1).
            num_features (int): Number of features to display in the explanation (default: 10).
            show_plot (bool): Whether to display the SHAP bar plot (default: True).
        
        Returns:
            shap.Explanation: SHAP explanation object.
        
        Outputs:
            - Prints the comment (if available) and SHAP values summary.
            - Displays a SHAP bar plot (if show_plot=True).
        """
        if instance_idx < 0 or instance_idx >= len(self.train_embeddings):
            raise ValueError(
                f"instance_idx {instance_idx} is out of bounds for traindata with {len(self.train_embeddings)} samples."
            )

        # Get the instance to explain
        instance_np = self.train_embeddings[instance_idx]  # shape: (input_size,)
        if self.use_deep_explainer:
            instance_reshaped = instance_np.reshape(1, 1, self.input_size)  # shape: (1, 1, input_size)
        else:
            instance_reshaped = instance_np.reshape(1, self.input_size)  # shape: (1, input_size)

        # Get SHAP values
        shap_values = self.explainer.shap_values(
            instance_reshaped, check_additivity=False
        )

        # Extract SHAP values for the specified class
        if self.use_deep_explainer:
            # DeepExplainer returns a list: [array for class_0, array for class_1]
            single_shap_values = shap_values[class_idx][0, 0]  # shape: (input_size,)
        else:
            # KernelExplainer returns a list: [array for class_0, array for class_1]
            single_shap_values = shap_values[class_idx][0]  # shape: (input_size,)

        # Get base value (expected value for the class)
        base_value = self.explainer.expected_value[class_idx]

        # Create SHAP Explanation object
        explanation = shap.Explanation(
            values=single_shap_values,
            base_values=base_value,
            data=instance_np,
            feature_names=[f"emb_dim_{i}" for i in range(self.input_size)]
        )

        # Print comment if available
        if 'comment' in self.traindata.columns:
            print(f"Explained comment (index {instance_idx}): {self.traindata.iloc[instance_idx]['comment']}")

        # Print summary of SHAP values
        print(f"SHAP values for class {['non-hate', 'hate'][class_idx]} (top {num_features} features):")
        top_indices = np.argsort(np.abs(single_shap_values))[-num_features:][::-1]
        for idx in top_indices:
            print(f"Feature emb_dim_{idx}: SHAP value = {single_shap_values[idx]:.6f}, Feature value = {instance_np[idx]:.6f}")

        # Display SHAP bar plot
        if show_plot:
            shap.plots.bar(explanation, max_display=num_features)
            plt.show()

        return explanation