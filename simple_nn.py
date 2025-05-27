import numpy as np

def sig(x):
    return 1 / (1 + np.exp(-x))

class SimpleNN:
    def __init__(self):
        # Initialize the working memeory for the neural network
        self.memory = np.zeros((100, 1))

        # Initialize weights and biases for each layer
        self.weights1 = np.random.randn(784, 16) * 0.01
        # w means write, so to write to the memory, and r means read, so to read from the memory
        self.weights1Mw = np.random.randn(16, 100)
        self.weights1Mr = np.random.randn(16, 100)
        self.biases1 = np.zeros((1, 16))
        self.biases1Mw = np.random.randn(1, 100)
        self.biases1Mr = np.random.randn(1, 100)

        self.weights2 = np.random.randn(16, 16) * 0.01
        self.weights2Mw = np.random.randn(16, 100)
        self.weights2Mr = np.random.randn(16, 100)
        self.biases2 = np.zeros((1, 16))
        self.biases2Mw = np.random.randn(1, 100)
        self.biases2Mr = np.random.randn(1, 100)

        self.weights3 = np.random.randn(16, 10) * 0.01
        self.weights3Mw = np.random.randn(10, 100)
        self.weights3Mr = np.random.randn(10, 100)
        self.biases3 = np.zeros((1, 10))
        self.biases3Mw = np.random.randn(1, 100)
        self.biases3Mr = np.random.randn(1, 100)

    def forward(self, x):
        # Layer 1
        np.set_printoptions(threshold=100000, linewidth=1000, precision=5, suppress=True)
        self.z1 = np.dot(x, self.weights1) + self.biases1
        # print(np.sum(x * self.weights1.T, axis=1) + self.biases1)
        print(np.dot(self.z1, self.weights1Mw) + self.biases1Mw)
        self.memory = np.dot(self.z1, self.weights1Mw) + self.biases1Mw
        print(self.memory)
        self.za1 = np.dot(self.memory, self.weights1Mr) + self.biases1Mr
        self.a1 = sig(self.za1)

        # Layer 2
        self.z2 = np.dot(self.a1, self.weights2) + self.biases2
        self.memory += np.dot(self.z2, self.weights2Mw) + self.biases2Mw
        self.za2 = np.dot(self.memory, self.weights2Mr) + self.biases2Mr
        self.a2 = sig(self.za2)

        # Layer 3
        self.z3 = np.dot(self.a2, self.weights3) + self.biases3
        self.memory += np.dot(self.z3, self.weights3Mw) + self.biases3Mw
        self.za3 = np.dot(self.memory, self.weights3Mr) + self.biases3Mr
        self.a3 = sig(self.za3)

        # Return final output and predictions
        predictions = np.argmax(self.a3, axis=1)
        return self.a3, predictions