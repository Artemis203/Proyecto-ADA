import numpy as np
from sklearn.datasets import load_digits, load_iris, load_wine, fetch_openml

def load_dataset(name, reduce_mnist=True, mnist_samples=5000):
    if name == "digits":
        data = load_digits()
        X = data.data.astype(float)
        y = np.eye(10)[data.target]
    elif name == "iris":
        data = load_iris()
        X = data.data.astype(float)
        y = np.eye(3)[data.target]
    elif name == "wine":
        data = load_wine()
        X = data.data.astype(float)
        y = np.eye(3)[data.target]
    elif name == "mnist":
        # Cargar MNIST
        data = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X = data.data.astype(float) / 255.0  # normalizar de 0-1
        y = data.target.astype(int)
        # Reducir para que entrene rápido
        if reduce_mnist:
            indices = np.random.RandomState(42).choice(len(X), mnist_samples, replace=False)
            X = X[indices]
            y = y[indices]
        y = np.eye(10)[y]
    else:
        raise ValueError("Dataset no válido. Opciones: digits, iris, wine, mnist")
    
    # Normalización adicional para que quede en [0,1]
    if name != "mnist":
        X_min = np.min(X)
        X_max = np.max(X)
        X = (X - X_min) / (X_max - X_min + 1e-8)
    else:
        X_min, X_max = 0.0, 1.0
    
    return X, y, X_min, X_max