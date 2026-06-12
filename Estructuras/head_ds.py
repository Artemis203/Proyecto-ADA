import heapq

class MinHeap:
    def __init__(self):
        self.data = []

    def push(self, value):
        heapq.heappush(self.data, value)

    def pop(self):
        if not self.data:
            return None
        return heapq.heappop(self.data)

    def peek(self):
        if not self.data:
            return None
        return self.data[0]

    def size(self):
        return len(self.data)

    def is_empty(self):
        return len(self.data) == 0