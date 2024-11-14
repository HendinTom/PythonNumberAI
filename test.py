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
def load_mnist_data(batch_size=100):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_loader = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_data = DataLoader(train_loader, batch_size=batch_size, shuffle=True)
    return train_data

def test(model, train_data):
    correct = 0
    for batch_idx, (data, targets) in enumerate(train_data):
        #Going through each image
            for i in range(len(data)):
                image = data[i]
                label = targets[i]
                # Flatten image and normalize
                input_data = image.view(784).numpy()#.reshape(-1, 1)
                input_data = (input_data - np.min(input_data)) / (np.max(input_data) - np.min(input_data))
                
                # Forward pass
                output, _ = model.forward(input_data)

                #Get which ones it got right
                if _ == label:
                    correct += 1
    
    print(f"Accruacy: {round((correct / len(train_data)), 2)}%")

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
    return model

# Main training procedure
if __name__ == "__main__":
    model = load_model()
    train_data = load_mnist_data(batch_size=100)
    test(model, train_data)