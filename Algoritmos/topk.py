import heapq

def top_k_heap(arr, k):
    return heapq.nlargest(k, arr)

def top_k_sort(arr, k):
    return sorted(arr, reverse=True)[:k]