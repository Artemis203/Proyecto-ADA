import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from Nucleo.mlp import MLP

# TERMINAL DE PRUEBA


def gradiente_numerico(modelo, X, y, epsilon=1e-5):
    """
    Calcula el gradiente numérico para todos los parámetros del modelo.
    """
    # Guardar copias originales
    W1_original = modelo.W1.copy()
    b1_original = modelo.b1.copy()
    W2_original = modelo.W2.copy()
    b2_original = modelo.b2.copy()
    
    grad_W1_num = np.zeros_like(modelo.W1)
    grad_b1_num = np.zeros_like(modelo.b1)
    grad_W2_num = np.zeros_like(modelo.W2)
    grad_b2_num = np.zeros_like(modelo.b2)
    
    # Gradiente numérico para W1
    for i in range(modelo.W1.shape[0]):
        for j in range(modelo.W1.shape[1]):
            modelo.W1[i, j] = W1_original[i, j] + epsilon
            y_hat = modelo.forward(X)
            loss_plus = -np.mean(np.sum(y * np.log(y_hat + 1e-9), axis=1))
            
            modelo.W1[i, j] = W1_original[i, j] - epsilon
            y_hat = modelo.forward(X)
            loss_minus = -np.mean(np.sum(y * np.log(y_hat + 1e-9), axis=1))
            
            grad_W1_num[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
            modelo.W1[i, j] = W1_original[i, j]
    
    # Gradiente numérico para b1
    for j in range(modelo.b1.shape[1]):
        modelo.b1[0, j] = b1_original[0, j] + epsilon
        y_hat = modelo.forward(X)
        loss_plus = -np.mean(np.sum(y * np.log(y_hat + 1e-9), axis=1))
        
        modelo.b1[0, j] = b1_original[0, j] - epsilon
        y_hat = modelo.forward(X)
        loss_minus = -np.mean(np.sum(y * np.log(y_hat + 1e-9), axis=1))
        
        grad_b1_num[0, j] = (loss_plus - loss_minus) / (2 * epsilon)
        modelo.b1[0, j] = b1_original[0, j]
    
    # Gradiente numérico para W2
    for i in range(modelo.W2.shape[0]):
        for j in range(modelo.W2.shape[1]):
            modelo.W2[i, j] = W2_original[i, j] + epsilon
            y_hat = modelo.forward(X)
            loss_plus = -np.mean(np.sum(y * np.log(y_hat + 1e-9), axis=1))
            
            modelo.W2[i, j] = W2_original[i, j] - epsilon
            y_hat = modelo.forward(X)
            loss_minus = -np.mean(np.sum(y * np.log(y_hat + 1e-9), axis=1))
            
            grad_W2_num[i, j] = (loss_plus - loss_minus) / (2 * epsilon)
            modelo.W2[i, j] = W2_original[i, j]
    
    # Gradiente numérico para b2
    for j in range(modelo.b2.shape[1]):
        modelo.b2[0, j] = b2_original[0, j] + epsilon
        y_hat = modelo.forward(X)
        loss_plus = -np.mean(np.sum(y * np.log(y_hat + 1e-9), axis=1))
        
        modelo.b2[0, j] = b2_original[0, j] - epsilon
        y_hat = modelo.forward(X)
        loss_minus = -np.mean(np.sum(y * np.log(y_hat + 1e-9), axis=1))
        
        grad_b2_num[0, j] = (loss_plus - loss_minus) / (2 * epsilon)
        modelo.b2[0, j] = b2_original[0, j]
    
    return grad_W1_num, grad_b1_num, grad_W2_num, grad_b2_num

def error_relativo(x, y):
    """Error relativo máximo entre dos arreglos."""
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def verificar_gradientes():
    """Prueba que los gradientes analíticos coincidan con los numéricos."""
    np.random.seed(42)
    X = np.random.randn(5, 64)
    y = np.eye(10)[np.random.randint(0, 10, 5)]
    
    modelo = MLP(64, 32, 10)
    
    modelo.forward(X)
    modelo.backward(y)
    
    # Gradientes analíticos (los que calcula backprop)
    grad_W1_analitico = modelo.dW1.copy()
    grad_b1_analitico = modelo.db1.copy()
    grad_W2_analitico = modelo.dW2.copy()
    grad_b2_analitico = modelo.db2.copy()
    
    # Gradientes numéricos
    grad_W1_num, grad_b1_num, grad_W2_num, grad_b2_num = gradiente_numerico(modelo, X, y)
    
    err_W1 = error_relativo(grad_W1_analitico, grad_W1_num)
    err_b1 = error_relativo(grad_b1_analitico, grad_b1_num)
    err_W2 = error_relativo(grad_W2_analitico, grad_W2_num)
    err_b2 = error_relativo(grad_b2_analitico, grad_b2_num)
    
    print("=== Verificación de Gradientes ===")
    print(f"Error relativo en W1: {err_W1:.6e}")
    print(f"Error relativo en b1: {err_b1:.6e}")
    print(f"Error relativo en W2: {err_W2:.6e}")
    print(f"Error relativo en b2: {err_b2:.6e}")
    
    if err_W1 < 1e-6 and err_b1 < 1e-6 and err_W2 < 1e-6 and err_b2 < 1e-6:
        print("Los gradientes son correctos La retropropagación está bien implementada.")
    else:
        print("Los gradientes no coinciden. Revisa la implementación de backward().")

if __name__ == "__main__":
    verificar_gradientes()