"""
Name: Henin Tom Vadakkeveettilan Hilariyos
Date: Wednesday June 18
Course: ICS-4U1
Description:
This program defines the FinalNetwork class for training a simple neural network on the MNIST dataset,
including data loading, training with backpropagation, and saving the trained model.
"""

import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pickle
from simple_nn import SimpleNN

class FinalNetwork(SimpleNN):
    """
    FinalNetwork extends SimpleNN and adds training, saving, and data loading for MNIST.
    """
    def __init__(self):
        super().__init__()

    # Sigmoid activation function
    def sig(self, x):
        return 1 / (1 + np.exp(-x))

    # Derivative of sigmoid for backpropagation
    def sig_derivative(self, x):
        return self.sig(x) * (1 - self.sig(x))

    # Loads MNIST data and returns a DataLoader for training
    def load_mnist_data(self, batch_size=100):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_loader = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        # Create batches of random samples from the training set
        train_data = DataLoader(train_loader, batch_size=batch_size, shuffle=True)
        return train_data

    # Trains the neural network using backpropagation and gradient descent
    def train(self, train_data, epochs=5, learning_rate=0.01):
        for epoch in range(epochs):
            total_cost = 0
            correct = 0
            # Loop through each batch in the training data
            for batch_idx, (data, targets) in enumerate(train_data):
                # Initialize gradients to zero for each batch
                d_avg_cost_to_weights1 = np.zeros_like(self.weights1)
                d_avg_cost_to_bias1 = np.zeros_like(self.biases1)
                d_avg_cost_to_weights2 = np.zeros_like(self.weights2)
                d_avg_cost_to_bias2 = np.zeros_like(self.biases2)
                d_avg_cost_to_weights3 = np.zeros_like(self.weights3)
                d_avg_cost_to_bias3 = np.zeros_like(self.biases3)

                # Loop through each image in the batch
                for i in range(len(data)):
                    image = data[i]
                    label = targets[i]

                    # Flatten and normalize the image to (784,)
                    input_data = image.view(784).numpy()
                    input_data = (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data))

                    # Forward pass through the network
                    output, prediction = self.forward(input_data)
                
                    # Create one-hot encoded target vector
                    y = np.zeros(10)
                    y[label] = 1
                    
                    # Compute mean squared error loss
                    cost = np.mean((output - y) ** 2)
                    total_cost += cost

                    # Count correct predictions
                    if prediction == label:
                        correct += 1
                    
                    # Backpropagation: calculate gradients for each layer
                    d_cost_to_a3 = 2*(output - y) # Gradient of loss w.r.t. output

                    d_z3_to_a3 = self.sig_derivative(self.z3)
                    d_weights3_to_z3 = self.a2
                    d_cost_to_weights3 = np.outer(d_weights3_to_z3, (d_z3_to_a3 * d_cost_to_a3).flatten())
                    d_avg_cost_to_weights3 += d_cost_to_weights3
                    d_cost_to_bias3 = (d_z3_to_a3 * d_cost_to_a3).reshape(1, -1)
                    d_avg_cost_to_bias3 += d_cost_to_bias3

                    d_cost_to_a2 = np.sum((self.weights3.T * (d_z3_to_a3 * d_cost_to_a3).reshape(-1, 1)).T, axis=1).reshape(1, -1)

                    d_z2_to_a2 = self.sig_derivative(self.z2)
                    d_weights2_to_z2 = self.a1
                    d_cost_to_weights2 = np.outer(d_weights2_to_z2, (d_z2_to_a2 * d_cost_to_a2).flatten())
                    d_avg_cost_to_weights2 += d_cost_to_weights2
                    d_cost_to_bias2 = (d_z2_to_a2 * d_cost_to_a2).reshape(1, -1)
                    d_avg_cost_to_bias2 += d_cost_to_bias2

                    d_cost_to_a1 = np.sum((self.weights2.T * (d_z2_to_a2 * d_cost_to_a2).reshape(-1, 1)).T, axis=1).reshape(1, -1)
                    d_z1_to_a1 = self.sig_derivative(self.z1)
                    d_weights1_to_z1 = input_data
                    d_cost_to_weights1 = np.outer(d_weights1_to_z1, (d_z1_to_a1 * d_cost_to_a1).flatten())
                    d_avg_cost_to_weights1 += d_cost_to_weights1
                    d_cost_to_bias1 = (d_z1_to_a1 * d_cost_to_a1).reshape(1, -1)
                    d_avg_cost_to_bias1 += d_cost_to_bias1

                # Update weights and biases using gradient descent
                self.weights3 -= learning_rate * d_avg_cost_to_weights3
                self.biases3 -= learning_rate * d_avg_cost_to_bias3
                self.weights2 -= learning_rate * d_avg_cost_to_weights2
                self.biases2 -= learning_rate * d_avg_cost_to_bias2
                self.weights1 -= learning_rate * d_avg_cost_to_weights1
                self.biases1 -= learning_rate * d_avg_cost_to_bias1
    
            # Print epoch summary (loss and accuracy)
            print(f"Epoch {epoch + 1}, Loss: {total_cost / len(train_data)}, Accruacy: {round((correct / len(train_data)), 2)}%")

    # Saves the trained model parameters to a file
    def save_model(self, filename="trained_model.pkl"):
        np.savez(filename,
                 weights1=self.weights1, biases1=self.biases1,
                 weights2=self.weights2, biases2=self.biases2,
                 weights3=self.weights3, biases3=self.biases3)

# Main training procedure
if __name__ == "__main__":
    # Instantiate the network and load data
    model = FinalNetwork()
    train_data = model.load_mnist_data(batch_size=100)
    # Train the model and save parameters
    model.train(train_data, epochs=40, learning_rate=0.01)
    model.save_model()