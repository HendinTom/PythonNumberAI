import numpy as np

def sig(x):
    return 1 / (1 + np.exp(-x))

class SimpleNN:
    def __init__(self):
        # Initialize weights and biases for each layer
        self.weights1 = np.random.randn(16, 784) * 0.01
        self.biases1 = np.zeros((16, 1))
        self.weights2 = np.random.randn(16, 16) * 0.01
        self.biases2 = np.zeros((16, 1))
        self.weights3 = np.random.randn(10, 16) * 0.01
        self.biases3 = np.zeros((10, 1))
        
        # Placeholder for activations
        self.a1 = None
        self.a2 = None
        self.a3 = None

    def forward(self, x):
        # Forward pass
        x = np.array(x).reshape(784, 1)

        # Layer 1
        z1 = np.dot(self.weights1, x) + self.biases1
        self.a1 = sig(z1)

        # Layer 2
        z2 = np.dot(self.weights2, self.a1) + self.biases2
        self.a2 = sig(z2)

        # Layer 3 (output layer)
        z3 = np.dot(self.weights3, self.a2) + self.biases3
        self.a3 = sig(z3)

        # Return final output and the index of the highest activation
        prediction = np.argmax(self.a3)
        return self.a3, prediction