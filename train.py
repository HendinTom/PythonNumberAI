import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pickle
from simple_nn import SimpleNN

# Load MNIST data
def load_mnist_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    return train_data

# Training function
def train(model, train_data, epochs=5, learning_rate=0.01):
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

    for images, labels in train_loader:
        # Flatten each 28x28 image to a 784-dimensional vector
        images = images.view(images.size(0), -1)  # Reshape to [batch_size, 784]

        # Convert images from [-1, 1] range to [0, 1] range
        images = (images + 1) / 2

        # Process each `images` batch with your model
        outputs = model.forward(images)  # Forward pass through the model

        # `labels` contains the ground truth digit (0-9) for each image
        print("Batch labels:", labels)  # Labels for the batch (tensor of integers)
    

# Save the model parameters
def save_model(model, filename="trained_model.pkl"):
    model_params = {
        "weights1": model.weights1,
        "biases1": model.biases1,
        "weights2": model.weights2,
        "biases2": model.biases2,
        "weights3": model.weights3,
        "biases3": model.biases3,
    }
    with open(filename, "wb") as f:
        pickle.dump(model_params, f)
    print(f"Model saved to {filename}")

# Main training procedure
if __name__ == "__main__":
    train_data = load_mnist_data()
    model = SimpleNN()
    train(model, train_data, epochs=5, learning_rate=0.01)
    save_model(model)






    # for epoch in range(epochs):
    #     total_loss = 0
    #     for i, (image, label) in enumerate(train_data):
    #         # Flatten image to 784x1 vector and normalize pixel values
    #         input_data = image.view(784).numpy()
    #         input_data = (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data))
            
    #         # Forward pass
    #         output, _ = model.forward(input_data)
            
    #         # One-hot encode the label
    #         target = np.zeros(10)
    #         target[label] = 1
            
    #         # Compute loss (Mean Squared Error)
    #         loss = np.mean((output.flatten() - target) ** 2)
    #         total_loss += loss
            
    #         # Backpropagation
    #         output_error = output.flatten() - target
            
    #         # Gradients for weights and biases of each layer
    #         d_weights3 = np.dot(output_error.reshape(-1, 1), model.a2.T.reshape(1, -1))  # Shape (10, 16)
    #         d_biases3 = output_error
            
    #         hidden_error2 = np.dot(model.weights3.T, output_error) * model.a2 * (1 - model.a2)
    #         d_weights2 = np.dot(hidden_error2.reshape(-1, 1), model.a1.T.reshape(1, -1))  # Shape (16, 16)
    #         d_biases2 = hidden_error2
            
    #         hidden_error1 = np.dot(model.weights2.T, hidden_error2) * model.a1 * (1 - model.a1)
    #         d_weights1 = np.dot(hidden_error1.reshape(-1, 1), input_data.reshape(1, -1))  # Shape (16, 784)
    #         d_biases1 = hidden_error1

    #         # Gradient descent update
    #         model.weights3 -= learning_rate * d_weights3
    #         model.biases3 -= learning_rate * d_biases3[:, None]
    #         model.weights2 -= learning_rate * d_weights2
    #         model.biases2 -= learning_rate * d_biases2[:, None]
    #         model.weights1 -= learning_rate * d_weights1
    #         model.biases1 -= learning_rate * d_biases1[:, None]
        
    #     print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_data)}")