import numpy as np

class SimpleAI:
    def __init__(self):
        self.weight = np.random.randn(1)
        self.bias = np.random.randn(1)

    def forward(self, x):
        z = x * self.weight + self.bias
        return self.sig(z)

    def sig(self, x):
        return 1 / (1 + np.exp(-x))


class SimpleNN(SimpleAI):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size1)
        self.biases1 = np.zeros((1, hidden_size1))
        self.weights2 = np.random.randn(hidden_size1, hidden_size2)
        self.biases2 = np.zeros((1, hidden_size2))
        self.weights3 = np.random.randn(hidden_size2, output_size)
        self.biases3 = np.zeros((1, output_size))

    def forward(self, x):
        # Ensure x is 2D for broadcasting
        if x.ndim == 1:
            x = x.reshape(1, -1)
        self.z1 = np.dot(x, self.weights1) + self.biases1
        self.a1 = self.sig(self.z1)
        self.z2 = np.dot(self.a1, self.weights2) + self.biases2
        self.a2 = self.sig(self.z2)
        self.z3 = np.dot(self.a2, self.weights3) + self.biases3
        self.a3 = self.sig(self.z3)
        prediction = np.argmax(self.a3, axis=1)[0]
        return self.a3.flatten(), prediction