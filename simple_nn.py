import numpy as np

def sig(x):
    return 1 / (1 + np.exp(-x))

class SimpleNN:
    def __init__(self):
        # Initialize weights and biases for each layer
        self.weights1 = np.random.randn(64, 784) * 0.01
        self.biases1 = np.zeros((64, 1))
        self.weights2 = np.random.randn(64, 64) * 0.01
        self.biases2 = np.zeros((64, 1))
        self.weights3 = np.random.randn(64, 64) * 0.01
        self.biases3 = np.zeros((64, 1))
        self.weights4 = np.random.randn(10, 64) * 0.01
        self.biases4 = np.zeros((10, 1))

    def forward(self, x):
        # Reshape input to be compatible with weight multiplication
        if x.ndim == 1:  # Single sample
            x = x.reshape(-1, 1)  # Shape (784, 1)
        
        # Layer 1
        self.z1 = np.dot(self.weights1, x) + self.biases1  # Shape (64, 1)
        self.a1 = sig(self.z1)

        # Layer 2
        self.z2 = np.dot(self.weights2, self.a1) + self.biases2  # Shape (64, 1)
        self.a2 = sig(self.z2)

        # Layer 3
        self.z3 = np.dot(self.weights3, self.a2) + self.biases3  # Shape (64, 1)
        self.a3 = sig(self.z3)

        # Output layer
        self.z4 = np.dot(self.weights4, self.a3) + self.biases4  # Shape (10, 1)
        self.a4 = sig(self.z4)

        # Return final output and predictions
        predictions = np.argmax(self.a4, axis=0)
        return self.a4, predictions