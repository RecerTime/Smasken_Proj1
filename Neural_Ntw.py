import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import pandas as pd
import matplotlib.pyplot as plt
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

def neural_network_setup(dataframe, hidden_sizes_multiplier = 8):
    # Split the DataFrame into inputs (X) and output (y)
    X = dataframe.iloc[:, :-1].values
    y = dataframe.iloc[:, -1].values

    # Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Calculate class weights for the loss function
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)

    # Define the model, loss function, and optimizer
    input_size = X_train.shape[1]
    hidden_sizes = hidden_sizes_multiplier*np.array([4, 2, 1])  # Example hidden layer sizes
    output_size = 1
    neural_network = SimpleNN(input_size, hidden_sizes, output_size)

    return (X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, neural_network), class_weights

# Function to train the neural network
def train_neural_network(args, class_weights_tensor, epochs=750, learning_rate=0.001, use_gpu=True):
    # Check for GPU
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, neural_network = args

    X_train_tensor = X_train_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device)
    X_test_tensor = X_test_tensor.to(device)
    y_test_tensor = y_test_tensor.to(device)
    class_weights_tensor = class_weights_tensor.to(device)
    model = neural_network.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights_tensor[1])  # Weighted loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
        '''
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
        '''
    # Testing the model
    model.eval()
    with torch.no_grad():
        predictions = torch.sigmoid(model(X_test_tensor))  # Apply sigmoid to map outputs to (0, 1)
        test_loss = criterion(predictions, y_test_tensor)
        '''
        print(f'Test Loss: {test_loss.item():.4f}')

        # Print a few predictions vs actual values for inspection
        print("Sample Predictions vs Actual Values:")
        for i in range(min(10, len(predictions))):
            print(f"Predicted: {predictions[i].item():.4f}, Actual: {y_test_tensor[i].item():.4f}")
        '''
    '''
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()

    # Plot confusion matrix (with rounded predictions)
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    rounded_predictions = (predictions > 0.5).float()
    cm = confusion_matrix(y_test_tensor.cpu().numpy().round(), rounded_predictions.cpu().numpy())
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap='Blues')
    plt.title('Confusion Matrix (Rounded Predictions)')
    plt.show()
    '''

    return test_loss.item()

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
#    0.5       1.2      ...  -0.3       0.73
#    0.1       0.8      ...   0.5       0.45

# df = pd.read_csv('your_data.csv')
# trained_model = train_neural_network(df)
# save_model(trained_model, 'trained_model.pth')
# loaded_model = load_model('trained_model.pth', input_size=14, hidden_size=16, output_size=1)

df = pd.read_csv('training_data_vt2025.csv')

hidden_sizes_multipliers = np.arange(11, 32, 2)
class_weights_multipliers = np.linspace(0.8, 8, 15)
epochs = np.arange(50, 1500, 25)

min_loss = np.inf
min_loss_params = ()

for hidden_sizes_multiplier in hidden_sizes_multipliers:
    args, class_weights = neural_network_setup(df, 4)
    for class_weights_multiplier in class_weights_multipliers:
        class_weights[1] *= class_weights_multiplier
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
        for epoch in epochs:
            loss = train_neural_network(args, class_weights_tensor, epoch)
            if loss < min_loss:
                min_loss = loss
                min_loss_params = (hidden_sizes_multiplier, class_weights_multiplier, epoch)
                print(f'New min loss: {loss} - {min_loss_params}')

#(11, 0.8, 700)