import numpy as np
from sklearn.datasets import make_classification

# TERMINAL DE PRUEBA

def generar_dataset_sintetico(n_muestras=1000, n_caracteristicas=20, n_clases=10, semilla=42):
    """
    Genera un dataset sintético para pruebas de estrés.
    Parámetros:
        n_muestras : número de ejemplos
        n_caracteristicas : número de atributos
        n_clases : número de clases (etiquetas)
        semilla : fijar para reproducibilidad
    Retorna:
        X : array de características (normalizado a [0,1])
        y : etiquetas en one-hot encoding
    """
    X, y = make_classification(
        n_samples=n_muestras,
        n_features=n_caracteristicas,
        n_informative=n_caracteristicas,
        n_redundant=0,
        n_classes=n_clases,
        random_state=semilla
    )
    # Normalización min-max a [0,1]
    X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0) + 1e-8)
    # One-hot encoding
    y_onehot = np.eye(n_clases)[y]
    return X, y_onehot

def generar_suite_estres():
    """Genera datasets de distintos tamaños para pruebas de escalado."""
    tamanios = [100, 500, 1000, 5000, 10000]
    datasets = {}
    for n in tamanios:
        X, y = generar_dataset_sintetico(n_muestras=n, n_caracteristicas=64, n_clases=10)
        datasets[n] = (X, y)
        print(f"Generado dataset con {n} muestras, forma X: {X.shape}, y: {y.shape}")
    return datasets

if __name__ == "__main__":
    # Prueba rápida
    X, y = generar_dataset_sintetico(100, 20, 5)
    print(f"Datos sintéticos de ejemplo: X.shape = {X.shape}, y.shape = {y.shape}")
    print("Los valores de X están en [0,1]:", X.min(), X.max())