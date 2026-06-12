import random

# Quickselect para top‑k (mayores elementos) – usado en hard mining
def quickselect_topk(arr, k):
    """
    Devuelve los k elementos con mayor valor.
    arr: lista de tuplas (índice, pérdida)
    k: número de elementos a retornar
    """
    arr = list(arr)
    if len(arr) <= k:
        return arr
    pivot = random.choice(arr)[1]
    mejor = []
    igualado = []
    peor = []
    for x in arr:
        value = x[1]
        if value > pivot:
            mejor.append(x)
        elif value < pivot:
            peor.append(x)
        else:
            igualado.append(x)
    if k <= len(mejor):
        return quickselect_topk(mejor, k)
    if k <= len(mejor) + len(igualado):
        return mejor + igualado[:k - len(mejor)]
    return mejor + igualado + quickselect_topk(peor, k - len(mejor) - len(igualado))


quickselect = quickselect_topk

# Quickselect para mediana 
def quickselect_median(arr):
    """
    Retorna la mediana de una lista de números.
    Si longitud par, retorna el promedio de los dos centrales.
    """
    arr_copy = list(arr)
    n = len(arr_copy)
    if n % 2 == 1:
        return _quickselect_median(arr_copy, 0, n-1, n//2)
    else:
        left = _quickselect_median(arr_copy, 0, n-1, n//2 - 1)
        right = _quickselect_median(arr_copy, 0, n-1, n//2)
        return (left + right) / 2.0

def _quickselect_median(arr, left, right, k):
    if left == right:
        return arr[left]
    pivot_index = random.randint(left, right)
    pivot_index = _partition(arr, left, right, pivot_index)
    if k == pivot_index:
        return arr[k]
    elif k < pivot_index:
        return _quickselect_median(arr, left, pivot_index-1, k)
    else:
        return _quickselect_median(arr, pivot_index+1, right, k)

def _partition(arr, left, right, pivot_index):
    pivot_value = arr[pivot_index]
    arr[pivot_index], arr[right] = arr[right], arr[pivot_index]
    store_index = left
    for i in range(left, right):
        if arr[i] < pivot_value:
            arr[store_index], arr[i] = arr[i], arr[store_index]
            store_index += 1
    arr[right], arr[store_index] = arr[store_index], arr[right]
    return store_index