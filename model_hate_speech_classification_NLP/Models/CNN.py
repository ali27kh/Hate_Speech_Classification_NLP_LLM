import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# Define the CNN model class
class CNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNN, self).__init__()
        # Fully connected layer 1: input_size to 128 units
        self.fc1 = nn.Linear(input_size, 128)
        # Fully connected layer 2: 128 to 64 units
        self.fc2 = nn.Linear(128, 64)
        # Fully connected layer 3: 64 to 32 units
        self.fc3 = nn.Linear(64, 32)
        # Fully connected layer 4: 32 to output_size units
        self.fc4 = nn.Linear(32, output_size)
        # ReLU activation function
        self.relu = nn.ReLU()
        # Dropout layer with 30% probability
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        # Apply first layer, ReLU activation, and dropout
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        # Apply second layer, ReLU activation, and dropout
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        # Apply third layer and ReLU activation
        x = self.relu(self.fc3(x))
        # Apply final layer to get output
        x = self.fc4(x)
        return x

# Define the model training class
class Model_training:
    def __init__(self, model, train_loader, val_loader, epochs):
        self.model = model  # CNN model instance
        self.train_loader = train_loader  # DataLoader for training data
        self.val_loader = val_loader  # DataLoader for validation data
        self.num_epochs = epochs  # Number of training epochs
        self.criterion = nn.CrossEntropyLoss()  # Loss function
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)  # Adam optimizer with learning rate 0.001

    def train(self):
        # Lists to store training and validation metrics
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []

        # Training loop for each epoch
        for epoch in range(self.num_epochs):
            self.model.train()  # Set model to training mode
            running_loss, correct, total = 0.0, 0, 0  # Initialize metrics for epoch
            for inputs, labels in self.train_loader:
                self.optimizer.zero_grad()  # Clear gradients
                outputs = self.model(inputs)  # Forward pass
                loss = self.criterion(outputs, labels)  # Compute loss
                loss.backward()  # Backpropagation
                self.optimizer.step()  # Update weights
                running_loss += loss.item()  # Accumulate loss

                _, predicted = torch.max(outputs, 1)  # Get predicted labels
                correct += (predicted == labels).sum().item()  # Count correct predictions
                total += labels.size(0)  # Count total samples

            # Calculate average training loss and accuracy for epoch
            train_loss = running_loss / len(self.train_loader)
            train_accuracy = correct / total
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            self.model.eval()  # Set model to evaluation mode
            val_loss, val_correct, val_total = 0.0, 0, 0  # Initialize validation metrics
            with torch.no_grad():  # Disable gradient computation
                for inputs, labels in self.val_loader:
                    outputs = self.model(inputs)  # Forward pass
                    loss = self.criterion(outputs, labels)  # Compute loss
                    val_loss += loss.item()  # Accumulate loss

                    _, predicted = torch.max(outputs, 1)  # Get predicted labels
                    val_correct += (predicted == labels).sum().item()  # Count correct predictions
                    val_total += labels.size(0)  # Count total samples

            # Calculate average validation loss and accuracy for epoch
            val_loss /= len(self.val_loader)
            val_accuracy = val_correct / val_total
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            # Print metrics every 20 epochs
            if (epoch + 1) % 20 == 0:
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

        return train_losses, val_losses, train_accuracies, val_accuracies  # Return metrics

# Function to load and preprocess data
def load_data(trainset, valset, testset, batch_size=32):
    # Convert input data to PyTorch tensors
    train_inputs = torch.tensor(trainset["comment_embedding"].tolist(), dtype=torch.float32)
    train_labels = torch.tensor(trainset["label"].tolist(), dtype=torch.long)
    val_inputs = torch.tensor(valset["comment_embedding"].tolist(), dtype=torch.float32)
    val_labels = torch.tensor(valset["label"].tolist(), dtype=torch.long)
    test_inputs = torch.tensor(testset["comment_embedding"].tolist(), dtype=torch.float32)
    test_labels = torch.tensor(testset["label"].tolist(), dtype=torch.long)

    # Pad validation and test inputs if they are shorter than training inputs
    if len(train_inputs[0]) > len(val_inputs[0]):
        diff = len(train_inputs[0]) - len(val_inputs[0])
        val_inputs = torch.cat([val_inputs, torch.zeros(val_inputs.size(0), diff)], dim=1)
        test_inputs = torch.cat([test_inputs, torch.zeros(test_inputs.size(0), diff)], dim=1)

    # Create TensorDatasets for training, validation, and test sets
    train_dataset = TensorDataset(train_inputs, train_labels)
    val_dataset = TensorDataset(val_inputs, val_labels)
    test_dataset = TensorDataset(test_inputs, test_labels)

    # Create DataLoaders for batch processing
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader  # Return DataLoaders

# Function to plot training and validation metrics
def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    Plot two figures: one for losses and one for accuracies.

    Args:
        train_losses (list): List of training losses per epoch.
        val_losses (list): List of validation losses per epoch.
        train_accuracies (list): List of training accuracies per epoch.
        val_accuracies (list): List of validation accuracies per epoch.
    """
    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

    # Plot training and validation accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Train Accuracy', color='green', linestyle='dashed')
    plt.plot(val_accuracies, label='Val Accuracy', color='orange', linestyle='dashed')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

# Function to evaluate the model on test data
def evaluate_cnn_model(model, test_loader):
    model.eval()  # Set model to evaluation mode
    test_predictions = []  # Store predicted labels
    true_labels = []  # Store true labels
    with torch.no_grad():  # Disable gradient computation
        for inputs, labels in test_loader:
            outputs = model(inputs)  # Forward pass
            _, predicted = torch.max(outputs, 1)  # Get predicted labels
            test_predictions.extend(predicted.tolist())  # Accumulate predictions
            true_labels.extend(labels.tolist())  # Accumulate true labels

    # Calculate evaluation metrics
    accuracy = accuracy_score(true_labels, test_predictions)
    precision = precision_score(true_labels, test_predictions, average='macro')
    recall = recall_score(true_labels, test_predictions, average='macro')
    f1score = f1_score(true_labels, test_predictions, average='macro')

    # Print metrics
    print(f"Accuracy {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1score:.4f}")

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(true_labels, test_predictions))

    # Plot confusion matrix
    cm = confusion_matrix(true_labels, test_predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=set(true_labels), yticklabels=set(true_labels))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()