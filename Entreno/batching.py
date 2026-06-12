import numpy as np

def create_batches(X, y, batch_size):
    # Mezclar datos
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    # Dividir en lotes
    for i in range(0, len(X), batch_size):
        batch = indices[i:i+batch_size]
        yield X[batch], y[batch]