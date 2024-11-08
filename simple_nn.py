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
        
        # Placeholder for activations and pre-activations
        self.a1 = None
        self.z1 = None  # Pre-activation value for layer 1
        self.a2 = None
        self.z2 = None  # Pre-activation value for layer 2
        self.a3 = None
        self.z3 = None  # Pre-activation value for output layer

    def forward(self, x):
        # Ensure x is a 2D array with shape [batch_size, 784]
        if x.ndim == 1:  # Single sample
            x = x.reshape(1, -1)
        elif x.ndim == 2 and x.shape[1] == 784:  # Batch of flattened images
            pass
        else:
            raise ValueError("Input shape must be (784,) for single sample or (batch_size, 784) for a batch.")

        # Layer 1
        self.z1 = np.dot(x, self.weights1.T) + self.biases1.T  # Shape (batch_size, 16)
        self.a1 = sig(self.z1)  # Apply activation

        # Layer 2
        self.z2 = np.dot(self.a1, self.weights2.T) + self.biases2.T  # Shape (batch_size, 16)
        self.a2 = sig(self.z2)  # Apply activation

        # Layer 3 (output layer)
        self.z3 = np.dot(self.a2, self.weights3.T) + self.biases3.T  # Shape (batch_size, 10)
        self.a3 = sig(self.z3)  # Apply activation

        # Return final output and predictions
        predictions = np.argmax(self.a3, axis=1)
        return self.a3, predictions
