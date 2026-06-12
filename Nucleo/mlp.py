import numpy as np
from Nucleo.activations import relu, relu_derivative, softmax

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        # Inicializacion mejorada (He)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size)
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.a1 = relu(self.z1)

        self.z2 = self.a1 @ self.W2 + self.b2
        self.y_hat = softmax(self.z2)

        return self.y_hat

    def backward(self, y):
        m = y.shape[0]  # tamaño del batch

        dz2 = (self.y_hat - y) / m
        self.dW2 = self.a1.T @ dz2
        self.db2 = dz2.sum(axis=0, keepdims=True)

        dz1 = (dz2 @ self.W2.T) * relu_derivative(self.z1)
        self.dW1 = self.X.T @ dz1
        self.db1 = dz1.sum(axis=0, keepdims=True)

    def update(self, lr):
        self.W1 -= lr * self.dW1
        self.b1 -= lr * self.db1
        self.W2 -= lr * self.dW2
        self.b2 -= lr * self.db2
    
    def save(self, path, x_min, x_max):
     np.savez(
        path,
        W1=self.W1,
        b1=self.b1,
        W2=self.W2,
        b2=self.b2,
        x_min=x_min,
        x_max=x_max
    )


    def load(self, path):
     data = np.load(path)

     self.W1 = data["W1"]
     self.b1 = data["b1"]
     self.W2 = data["W2"]
     self.b2 = data["b2"]

     self.x_min = data["x_min"]
     self.x_max = data["x_max"]