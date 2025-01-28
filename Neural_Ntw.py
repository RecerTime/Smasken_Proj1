import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
import numpy as np

# Define the neural network class
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(SimpleNN, self).__init__()
        layers = []

        # Create hidden layers dynamically
        for i, hidden_size in enumerate(hidden_sizes):
            if i == 0:
                layers.append(nn.Linear(input_size, hidden_size))
            else:
                layers.append(nn.Linear(hidden_sizes[i-1], hidden_size))
            layers.append(nn.ReLU())

        # Add the output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # Combine all layers into a sequential module
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Function to train the neural network
def train_neural_network(dataframe, epochs=2500, batch_size=32, learning_rate=0.001, use_gpu=True, threshold=0.5):
    # Check for GPU
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    # Convert target column to categorical (0 for Low, 1 for High)
    dataframe['increase_stock'] = dataframe['increase_stock'].apply(lambda x: 1 if x == 'high_bike_demand' else 0)

    # Display correlations between features and target
    feature_correlation = dataframe.corr()["increase_stock"].sort_values(ascending=False)
    print("Feature correlations with increase_stock:\n", feature_correlation)

    # Filter features based on correlation threshold
    correlation_threshold = 0.1  # You can adjust this threshold as needed
    selected_features = feature_correlation[~feature_correlation.isna() & (feature_correlation.abs() >= correlation_threshold)].index
    selected_features = selected_features.drop("increase_stock")  # Exclude the target from features
    print(f"Selected features based on correlation threshold ({correlation_threshold}): {selected_features.tolist()}")

    # Update DataFrame with selected features
    dataframe = dataframe[selected_features.tolist() + ["increase_stock"]]

    # Split the DataFrame into inputs (X) and output (y)
    X = dataframe.iloc[:, :-1].values
    y = dataframe.iloc[:, -1].values

    # Oversample minority class using SMOTE
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    # Calculate class weights for the loss function
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    # Define the model, loss function, and optimizer
    input_size = X_train.shape[1]
    hidden_sizes = (32, 16, 8)  # Example hidden layer sizes
    output_size = 2  # Two classes: High and Low

    model = SimpleNN(input_size, hidden_sizes, output_size).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)  # Weighted loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Custom metrics for accuracy of each class
    def compute_class_accuracy(y_true, y_pred):
        y_true = y_true.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        cm = confusion_matrix(y_true, y_pred)
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        return per_class_accuracy

    # Training loop
    train_losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    # Testing the model
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted_classes = torch.max(outputs, 1)  # Get the class with the highest probability
        test_loss = criterion(outputs, y_test_tensor)
        print(f'Test Loss: {test_loss.item():.4f}')

        # Calculate and display per-class accuracy
        per_class_accuracy = compute_class_accuracy(y_test_tensor, predicted_classes)
        print(f"Per-Class Accuracy: Low Demand: {per_class_accuracy[0]:.4f}, High Demand: {per_class_accuracy[1]:.4f}")

        # Print a few predictions vs actual values for inspection
        print("Sample Predictions vs Actual Values:")
        for i in range(min(10, len(predicted_classes))):
            print(f"Predicted: {predicted_classes[i].item()}, Actual: {y_test_tensor[i].item()}")

    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()

    # Plot confusion matrix
    cm = confusion_matrix(y_test_tensor.cpu().numpy(), predicted_classes.cpu().numpy())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["low_bike_demand", "high_bike_demand"])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix for Categorical Output')
    plt.show()

    return model

# Function to save the trained model
def save_model(model, file_path):
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")

# Function to load a saved model
def load_model(file_path, input_size, hidden_sizes, output_size):
    model = SimpleNN(input_size, hidden_sizes, output_size)
    model.load_state_dict(torch.load(file_path))
    model.eval()
    print(f"Model loaded from {file_path}")
    return model

# Example usage
# The DataFrame `df` should have 14 input features as columns and the 15th column as the target output.
# Each row represents a training sample.
# For example, df should look like this:
#    feature1  feature2  ...  feature14  target
#    0.5       1.2      ...  -0.3       low_bike_demand
#    0.1       0.8      ...   0.5       high_bike_demand

df = pd.read_csv('training_data_vt2025.csv')
trained_model = train_neural_network(df, threshold=0.5)
# save_model(trained_model, 'trained_model.pth')
# loaded_model = load_model('trained_model.pth', input_size=14, hidden_sizes=(32, 16, 8), output_size=2)
