import numpy as np

def sig(x):
    return 1 / (1 + np.exp(-x))

class SimpleNN:
    def __init__(self):
        # Initialize weights and biases for each layer
        self.weights1 = np.random.randn(784, 16) * 0.01
        self.biases1 = np.zeros((1, 16))
        self.weights2 = np.random.randn(16, 16) * 0.01
        self.biases2 = np.zeros((1, 16))
        self.weights3 = np.random.randn(16, 10) * 0.01
        self.biases3 = np.zeros((1, 10))

    def forward(self, x):
        
        # Layer 1
        #Flipping weights1 to go from (784, 16) to (16, 784) because I want there to be 784 columns to allow for multiplicaiton
        self.z1 = np.sum((self.weights1.T * x) + self.biases1.T, axis=1)
        self.a1 = sig(self.z1)

        # Layer 2
        self.z2 = np.sum((self.weights2.T * self.a1) + self.biases2.T, axis=1)
        self.a2 = sig(self.z2)

        # Layer 3
        self.z3 = np.sum((self.weights3.T * self.a2) + self.biases3.T, axis=1)
        self.a3 = sig(self.z3)

        # Return final output and predictions
        predictions = np.argmax(self.a3, axis=0)
        return self.a3, predictions