"""
Name: Henin Tom Vadakkeveettilan Hilariyos
Date: Wednesday June 18
Course: ICS-4U1
Description:
This program loads a trained neural network model and tests its accuracy on the MNIST test dataset.
"""

import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pickle
from simple_nn import SimpleNN

# Softmax function to convert logits to probabilities
def softmax(x):
    """
    The softmax(x) function
    Converts a vector of logits to a probability distribution.
    """
    x = np.array(x)
    if x.ndim > 1:
        x = x.flatten()
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# Loads the saved model parameters from file and returns a SimpleNN instance
def load_model(filename="trained_model.pkl"):
    """
    The load_model(filename) function
    Loads model weights and biases from a file and returns a SimpleNN object.
    """
    with open(filename, "rb") as f:
        model_params = pickle.load(f)
    model = SimpleNN()
    # Set model weights and biases from loaded parameters
    model.weights1 = model_params["weights1"]
    model.biases1 = model_params["biases1"]
    model.weights2 = model_params["weights2"]
    model.biases2 = model_params["biases2"]
    model.weights3 = model_params["weights3"]
    model.biases3 = model_params["biases3"]
    return model

# Main testing procedure
if __name__ == "__main__":
    # Instantiate and load trained model
    model = load_model()

    # Set up MNIST test data loader
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    test_loader = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    test_data = DataLoader(test_loader, batch_size=1, shuffle=False)

    correct = 0  # Counter for correct predictions
    total = 0    # Counter for total samples

    # Loop through each test sample
    for data, target in test_data:
        # Flatten and normalize the image to (784,)
        input_data = data.view(784).numpy()
        input_data = (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data))

        # Forward pass through the model
        output, prediction = model.forward(input_data)
        label = int(target)

        # Compare prediction to true label
        if prediction == label:
            correct += 1
        total += 1

        # Optionally, print the prediction and confidence for the first 5 samples
        if total <= 5:
            probs = softmax(output)
            confidence = probs[prediction]
            print(f"Sample {total}: Predicted {prediction}, True {label}, Confidence: {round(confidence*100, 2)}%")

    # Print overall accuracy
    print(f"Test Accuracy: {correct}/{total} = {round(100 * correct / total, 2)}%")