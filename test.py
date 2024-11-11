import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import pickle
from simple_nn import SimpleNN

# np.set_printoptions(threshold=np.inf)

def sig(x):
    return 1 / (1 + np.exp(-x))

def sig_derivative(x):
    return sig(x) * (1 - sig(x))

# Load MNIST data
def load_mnist_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    return train_data

def test(model, train_data):
    correct = 0
    for i, (image, label) in enumerate(train_data):
        # Flatten image and normalize
        input_data = image.view(784).numpy().reshape(-1, 1)
        input_data = (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data))
        
        # Forward pass
        output, _ = model.forward(input_data)

        #Get which ones it got right
        for i in range (len(_)):
            if np.array_equal(_[i], label):
                correct += 1
    
    print(f"Accruacy: {round((correct / len(train_data)) * 100, 2)}%")

# Load the saved model
def load_model(filename="trained_model.pkl"):
    with open(filename, "rb") as f:
        model_params = pickle.load(f)
    
    model = SimpleNN()
    model.weights1 = model_params["weights1"]
    model.biases1 = model_params["biases1"]
    model.weights2 = model_params["weights2"]
    model.biases2 = model_params["biases2"]
    model.weights3 = model_params["weights3"]
    model.biases3 = model_params["biases3"]
    model.weights4 = model_params["weights4"]  # Add this line
    model.biases4 = model_params["biases4"]    # And this line
    return model

# Main training procedure
if __name__ == "__main__":
    model = load_model()
    train_data = load_mnist_data()
    test(model, train_data)