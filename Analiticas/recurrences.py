import random
import time
import matplotlib.pyplot as plt
import numpy as np
from Algoritmos.quickselect import quickselect_median
from Algoritmos.topk import top_k_heap, top_k_sort

def measure_time_median():
    sizes = [100, 500, 1000, 5000, 10000, 20000]
    times = []
    for n in sizes:
        arr = [random.random() for _ in range(n)]
        start = time.time()
        median = quickselect_median(arr)
        elapsed = time.time() - start
        times.append(elapsed)
        print(f"n={n}, mediana={median:.4f}, tiempo={elapsed:.6f}s")
    
    # Ajuste teórico O(n)
    plt.plot(sizes, times, 'o-', label="Quickselect (mediana)")
    # Curva teórica (escalada)
    theoretical = [t * (sizes[-1]/s) for t, s in zip(times, sizes)]  
    plt.plot(sizes, [0.000001 * n for n in sizes], '--', label="O(n) teórica")
    plt.xlabel("Tamaño del array")
    plt.ylabel("Tiempo (s)")
    plt.title("Quickselect para mediana - Recurrencia T(n)=T(n/2)+O(n)")
    plt.legend()
    plt.savefig("quickselect_median.png")
    plt.show()

def measure_time_topk():
    sizes = [1000, 5000, 10000, 20000]
    k = 100
    times_heap = []
    times_sort = []
    for n in sizes:
        arr = [random.random() for _ in range(n)]
        start = time.time()
        top_k_heap(arr, k)
        times_heap.append(time.time() - start)
        start = time.time()
        top_k_sort(arr, k)
        times_sort.append(time.time() - start)
    plt.plot(sizes, times_heap, 'o-', label="Heap O(n log k)")
    plt.plot(sizes, times_sort, 's-', label="Sort O(n log n)")
    plt.xlabel("n")
    plt.ylabel("Tiempo (s)")
    plt.title("Top-k: Heap vs Sort")
    plt.legend()
    plt.savefig("topk_comparison.png")
    plt.show()

if __name__ == "__main__":
    print("Medición Quickselect mediana")
    measure_time_median()
    print("Medición Top-k Heap vs Sort")
    measure_time_topk()