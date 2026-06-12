import heapq

class LossHash:
    """Almacena pérdida por índice de muestra (hash map)"""
    def __init__(self):
        self.losses = {}  # idx -> loss
    def update(self, idx, loss):
        self.losses[idx] = loss
    def get(self, idx):
        return self.losses.get(idx, 0.0)
    def top_k_indices(self, k):
        pass

class HeapTopK:
    """Mantiene los k elementos con mayor valor usando heap mínimo de tamaño k"""
    def __init__(self, k):
        self.k = k
        self.heap = []  # elementos (valor, item)
    def add(self, value, item):
        if len(self.heap) < self.k:
            heapq.heappush(self.heap, (value, item))
        else:
            if value > self.heap[0][0]:
                heapq.heapreplace(self.heap, (value, item))
    def get_top(self):
        # Devuelve lista de items ordenados descendente (mayor a menor)
        return [item for (value, item) in sorted(self.heap, reverse=True)]