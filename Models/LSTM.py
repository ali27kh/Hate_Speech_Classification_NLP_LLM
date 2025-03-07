import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

class LSTMClassifier:
    def __init__(self, tokenizer, embedding_dim=768, lstm_units=300, dropout_rate=0.3):
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.max_length = tokenizer.model_max_length
        self.model = self._build_model()
        
    def _build_model(self):
        model = Sequential([
            Bidirectional(LSTM(units=self.lstm_units, input_shape=(None, self.embedding_dim), return_sequences=True)),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            Bidirectional(LSTM(units=self.lstm_units // 2, return_sequences=False)),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(self.dropout_rate),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(self.dropout_rate // 2),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999),
            loss='binary_crossentropy',
            metrics=['accuracy', 'Precision', 'Recall']
        )
        return model
    
    def train_model(self, trainset, testset, epochs=50, batch_size=32, validation_split=0.2):
        X_train = tf.constant(trainset["comment_embedding"].tolist(), dtype=tf.float32)
        y_train = tf.constant(trainset["label"].tolist(), dtype=tf.int32)
        X_test = tf.constant(testset["comment_embedding"].tolist(), dtype=tf.float32)
        y_test = tf.constant(testset["label"].tolist(), dtype=tf.int32)
        
        if len(X_train.shape) == 4:
            X_train = tf.squeeze(X_train, axis=2)
            X_test = tf.squeeze(X_test, axis=2)
        elif len(X_train.shape) == 2:
            X_train = tf.expand_dims(X_train, axis=1)
            X_test = tf.expand_dims(X_test, axis=1)
            
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, mode='min')
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, mode='min')
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(X_test, y_test, verbose=0)
        # print(f"\nTest Loss: {test_loss:.4f}")
        # print(f"Test Accuracy: {test_accuracy:.4f}")
        # print(f"Test Precision: {test_precision:.4f}")
        # print(f"Test Recall: {test_recall:.4f}")
        
        return history
    
    def predict(self, X):
        if len(X.shape) == 4:
            X = tf.squeeze(X, axis=2)
        elif len(X.shape) == 2:
            X = tf.expand_dims(X, axis=1)
        return self.model.predict(X)
    
    def evaluate_model(self, test_data):
        """
        Evaluate the LSTM model on test data.
        
        Args:
            test_data: DataFrame with 'comment_embedding' and 'label' columns
        """
        test_inputs = tf.constant(test_data["comment_embedding"].tolist(), dtype=tf.float32)
        test_labels = tf.constant(test_data["label"].tolist(), dtype=tf.int32)

        # Reshape inputs to match model expectations
        if len(test_inputs.shape) == 4:
            test_inputs = tf.squeeze(test_inputs, axis=2)
        elif len(test_inputs.shape) == 2:
            test_inputs = tf.expand_dims(test_inputs, axis=1)

        # Get predictions
        predictions = self.model.predict(test_inputs)
        binary_predictions = (predictions > 0.5).astype(int).flatten()

        # Calculate metrics
        accuracy = accuracy_score(test_labels, binary_predictions)
        precision = precision_score(test_labels, binary_predictions, average='macro')
        recall = recall_score(test_labels, binary_predictions, average='macro')
        f1score = f1_score(test_labels, binary_predictions, average='macro')

        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1score:.4f}")

        # Classification Report
        print("\nClassification Report:")
        print(classification_report(test_labels, binary_predictions))

        # Confusion Matrix
        cm = confusion_matrix(test_labels, binary_predictions)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                   xticklabels=np.unique(test_labels), 
                   yticklabels=np.unique(test_labels))
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

    def plot_metrics(self, history):
        """
        Plot training and validation loss and accuracy.
        
        Args:
            history: Training history object from model.fit()
        """
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        train_accuracy = history.history['accuracy']
        val_accuracy = history.history['val_accuracy']

        # Plot Loss
        plt.figure(figsize=(10, 5))
        plt.plot(train_loss, label='Train Loss', color='blue')
        plt.plot(val_loss, label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()

        # Plot Accuracy
        plt.figure(figsize=(10, 5))
        plt.plot(train_accuracy, label='Train Accuracy', color='green', linestyle='dashed')
        plt.plot(val_accuracy, label='Validation Accuracy', color='orange', linestyle='dashed')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()

# Usage example:
"""
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
lstm_classifier = LSTMClassifier(tokenizer, embedding_dim=768)

# Train
history = lstm_classifier.train_model(traindata, testdata)

# Evaluate
lstm_classifier.evaluate_model(testdata)

# Plot metrics
lstm_classifier.plot_metrics(history)
"""