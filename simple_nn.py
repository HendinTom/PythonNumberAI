"""
Name: Henin Tom Vadakkeveettilan Hilariyos
Date: Wednesday June 18
Course: ICS-4U1
Description:
This file defines a simple feedforward neural network (SimpleNN) with three layers, 
including encapsulated weights and biases, and provides forward propagation and sigmoid activation.
"""

import sys
import numpy as np

class SimpleNN:
    """
    SimpleNN class implements a 3-layer fully connected neural network with sigmoid activations.
    All weights and biases are private and accessed via property getters/setters.
    """
    def __init__(self):
        # Layer 1 weights and biases (input to first hidden layer)
        self.__weights1 = np.random.randn(784, 16) * 0.01
        self.__biases1 = np.zeros((1, 16))
        # Layer 2 weights and biases (first hidden to second hidden)
        self.__weights2 = np.random.randn(16, 16) * 0.01
        self.__biases2 = np.zeros((1, 16))
        # Layer 3 weights and biases (second hidden to output)
        self.__weights3 = np.random.randn(16, 10) * 0.01
        self.__biases3 = np.zeros((1, 10))

    # Sigmoid activation function
    def sig(self, x):
        return 1 / (1 + np.exp(-x))
    
    # Forward propagation through the network with input size check
    def forward(self, x):
        try:
            # Check if input is the correct size (784,)
            if x.shape[0] != 784:
                raise ValueError(f"Input size is {x.shape[0]}, but expected 784 (28x28 image flattened).")
            # Layer 1: input to first hidden
            self.__z1 = np.dot(x, self.__weights1) + self.__biases1
            self.__a1 = self.sig(self.__z1)
            # Layer 2: first hidden to second hidden
            self.__z2 = np.dot(self.__a1, self.__weights2) + self.__biases2
            self.__a2 = self.sig(self.__z2)
            # Layer 3: second hidden to output
            self.__z3 = np.dot(self.__a2, self.__weights3) + self.__biases3
            self.__a3 = self.sig(self.__z3)
            # Return output vector and predicted class
            predictions = int(np.argmax(self.__a3))
            return self.__a3, predictions
        except Exception as e:
            # Print error message and exit the program
            print(f"Error in SimpleNN.forward: {e}")
            sys.exit(1)

    # Property getters and setters for encapsulated weights and biases
    @property
    def weights1(self): return self.__weights1
    @weights1.setter
    def weights1(self, value): self.__weights1 = value

    @property
    def biases1(self): return self.__biases1
    @biases1.setter
    def biases1(self, value): self.__biases1 = value

    @property
    def weights2(self): return self.__weights2
    @weights2.setter
    def weights2(self, value): self.__weights2 = value

    @property
    def biases2(self): return self.__biases2
    @biases2.setter
    def biases2(self, value): self.__biases2 = value

    @property
    def weights3(self): return self.__weights3
    @weights3.setter
    def weights3(self, value): self.__weights3 = value

    @property
    def biases3(self): return self.__biases3
    @biases3.setter
    def biases3(self, value): self.__biases3 = value

    # Property getters for intermediate layer outputs (for backpropagation)
    @property
    def z1(self): return self.__z1
    @property
    def a1(self): return self.__a1
    @property
    def z2(self): return self.__z2
    @property
    def a2(self): return self.__a2
    @property
    def z3(self): return self.__z3
    @property
    def a3(self): return self.__a3