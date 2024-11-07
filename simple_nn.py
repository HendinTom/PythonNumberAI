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
        # Ensure x is a 2D array with shape [batch_size, 784]
        if x.ndim == 1:  # If x is a single sample, reshape it to (1, 784)
            x = x.reshape(1, -1)
        
        # Layer 1
        z1 = np.dot(x, self.weights1.T) + self.biases1.T  # Shape (batch_size, 16)
        self.a1 = sig(z1)

        # Layer 2
        z2 = np.dot(self.a1, self.weights2.T) + self.biases2.T  # Shape (batch_size, 16)
        self.a2 = sig(z2)

        # Layer 3 (output layer)
        z3 = np.dot(self.a2, self.weights3.T) + self.biases3.T  # Shape (batch_size, 10)
        self.a3 = sig(z3)

        # Return final output (batch of predictions) and the indices of the highest activations
        predictions = np.argmax(self.a3, axis=1)
        return self.a3, predictions