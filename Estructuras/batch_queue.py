from collections import deque

class BatchQueue:
    def __init__(self):
        self.q = deque()

    def push(self, item):
        self.q.append(item)

    def pop(self):
        return self.q.popleft()

    def empty(self):
        return len(self.q) == 0