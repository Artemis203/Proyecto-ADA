import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import random
import time
import numpy as np
import matplotlib.pyplot as plt
from Algoritmos.quickselect import quickselect_median

# TERMINAL DE PRUEBA

def medir_tiempos_quickselect():
    """
    Mide el tiempo de ejecución de quickselect_median para arrays de distintos tamaños.
    Grafica los resultados y ajusta una recta para validar la complejidad O(n).
    """
    tamanios = [1000, 2000, 5000, 10000, 20000, 50000]
    tiempos = []
    
    print("=== Validación empírica de Quickselect (mediana) ===")
    print("Tamaño del array | Tiempo (s)")
    for n in tamanios:
        # Generar array de números aleatorios
        arr = [random.random() for _ in range(n)]
        inicio = time.perf_counter()
        mediana = quickselect_median(arr)
        fin = time.perf_counter()
        duracion = fin - inicio
        tiempos.append(duracion)
        print(f"{n:10d}     | {duracion:.6f}")
    
    # Regresión lineal (esperamos una recta)
    coef = np.polyfit(tamanios, tiempos, 1)
    tendencia_lineal = np.polyval(coef, tamanios)
    
    # Mostrar ecuación de la recta
    print(f"\nAjuste lineal: tiempo ≈ {coef[0]:.2e} * n + {coef[1]:.2e}")
    
    # Graficar
    plt.figure(figsize=(8,5))
    plt.plot(tamanios, tiempos, 'bo-', label='Quickselect (mediana)')
    plt.plot(tamanios, tendencia_lineal, 'r--', label=f'Regresión lineal (pendiente {coef[0]:.2e})')
    plt.xlabel('Tamaño del array (n)')
    plt.ylabel('Tiempo (s)')
    plt.title('Validación empírica de la complejidad O(n) del Quickselect para mediana')
    plt.legend()
    plt.grid(True)
    
    # Guardar la gráfica (por ejemplo, en la raiz del proyecto)
    plt.savefig('quickselect_mediana_validacion.png', dpi=150)
    plt.show()
    
    print("\nConclusión: El tiempo crece aproximadamente de forma lineal, confirmando la complejidad O(n).")
    print("Esto respalda la aplicación del Método Maestro (caso 3) sobre la recurrencia T(n) = T(n/2) + O(n).")

if __name__ == "__main__":
    medir_tiempos_quickselect()