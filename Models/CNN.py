import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# CNN model
class CNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(CNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Training model class
class Model_training:
    def __init__(self, model, train_loader, val_loader,epochs):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self):
        train_losses, val_losses = [], []
        train_accuracies, val_accuracies = [], []

        for epoch in range(self.num_epochs):
            self.model.train()
            running_loss, correct, total = 0.0, 0, 0
            for inputs, labels in self.train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

            train_loss = running_loss / len(self.train_loader)
            train_accuracy = correct / total
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            self.model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, labels in self.val_loader:
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)

            val_loss /= len(self.val_loader)
            val_accuracy = val_correct / val_total
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            if (epoch+1) % 20 == 0:
                print(f"Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}")

        return train_losses, val_losses, train_accuracies, val_accuracies

def load_data(trainset, valset, testset, batch_size=32):
    train_inputs = torch.tensor(trainset["comment_embedding"].tolist(), dtype=torch.float32)
    train_labels = torch.tensor(trainset["label"].tolist(), dtype=torch.long)
    val_inputs = torch.tensor(valset["comment_embedding"].tolist(), dtype=torch.float32)
    val_labels = torch.tensor(valset["label"].tolist(), dtype=torch.long)
    test_inputs = torch.tensor(testset["comment_embedding"].tolist(), dtype=torch.float32)
    test_labels = torch.tensor(testset["label"].tolist(), dtype=torch.long)

    if len(train_inputs[0]) > len(val_inputs[0]):
        diff = len(train_inputs[0]) - len(val_inputs[0])
        val_inputs = torch.cat([val_inputs, torch.zeros(val_inputs.size(0), diff)], dim=1)
        test_inputs = torch.cat([test_inputs, torch.zeros(test_inputs.size(0), diff)], dim=1)

    train_dataset = TensorDataset(train_inputs, train_labels)
    val_dataset = TensorDataset(val_inputs, val_labels)
    test_dataset = TensorDataset(test_inputs, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader


def plot_metrics(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    Affiche deux figures : une pour les pertes (loss) et une pour les précisions (accuracy).
    
    Args:
        train_losses (list): Liste des pertes d'entraînement par époque.
        val_losses (list): Liste des pertes de validation par époque.
        train_accuracies (list): Liste des précisions d'entraînement par époque.
        val_accuracies (list): Liste des précisions de validation par époque.
    """
    # Figure 1 : Plot des pertes (loss)
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()

    # Figure 2 : Plot des précisions (accuracy)
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Train Accuracy', color='green', linestyle='dashed')
    plt.plot(val_accuracies, label='Val Accuracy', color='orange', linestyle='dashed')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

# Evaluate Model
def evaluate_cnn_model(model, test_loader):
    model.eval()
    test_predictions = []
    true_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_predictions.extend(predicted.tolist())
            true_labels.extend(labels.tolist())

    # Metrics Calculation
    accuracy = accuracy_score(true_labels, test_predictions)
    precision = precision_score(true_labels, test_predictions, average='macro')
    recall = recall_score(true_labels, test_predictions, average='macro')
    f1score = f1_score(true_labels, test_predictions, average='macro')

    print(f"Accuracy {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1score:.4f}")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(true_labels, test_predictions))

    # Confusion Matrix
    cm = confusion_matrix(true_labels, test_predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=set(true_labels), yticklabels=set(true_labels))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

