import numpy as np
import time
from .batching import create_batches
from Estructuras.batch_queue import BatchQueue

class Trainer:
    def __init__(self, model, lr=0.01):
        self.model = model
        self.lr = lr
        self.loss_history = []
        self.time_history = []
        self.acc_history = []

    def train(self, X, y, epochs=10, batch_size=32, telegram=None, selector=None, use_queue=True, augment=None):
        """
        augment: puede ser 'digits', 'mnist' o None. Si es 'digits' o 'mnist', aplica aumentación a las imágenes.
        """
        for epoch in range(epochs):
            start = time.time()
            total_loss = 0
            num_batches = 0
            correct = 0
            total = 0

            # Usar cola de lotes si se solicita 
            if use_queue:
                batch_queue = BatchQueue()
                for Xb, yb in create_batches(X, y, batch_size):
                    batch_queue.push((Xb, yb))
                # Procesar desde la cola
                while not batch_queue.empty():
                    X_batch, y_batch = batch_queue.pop()

                    # Aumento de Datos
                    if augment in ['digits', 'mnist']:
                        from utils.augmentation import augment_batch
                        # Inferir tamaño de la imagen (raíz cuadrada del número de características)
                        size = int(np.sqrt(X_batch.shape[1]))  # 8 para digits, 28 para mnist
                        # Reformar a (batch, size, size)
                        X_img = X_batch.reshape(-1, size, size)
                        # Aplicar aumentación (probabilidad 0.5 por muestra)
                        X_img_aug = augment_batch(X_img, dataset=augment, prob=0.5)
                        # Volver a aplanar
                        X_batch = X_img_aug.reshape(-1, size * size)
                    

                    # forward + pérdida
                    pred = self.model.forward(X_batch)
                    losses = -np.sum(y_batch * np.log(pred + 1e-9), axis=1)
                    # métricas
                    pred_labels = np.argmax(pred, axis=1)
                    true_labels = np.argmax(y_batch, axis=1)
                    correct += np.sum(pred_labels == true_labels)
                    total += len(y_batch)

                    # Hard mining con selector (usando heap top-k)
                    if selector:
                        # selector espera (lista de (idx, loss), k) y devuelve índices
                        # Se usara HeapTopK si selector es "heap"
                        if selector == "heap":
                            from utils.structures import HeapTopK
                            heap_k = HeapTopK(k=max(1, int(0.3 * len(losses))))
                            for i, loss in enumerate(losses):
                                heap_k.add(loss, i)
                            indices = heap_k.get_top()
                        else:
                            arr = list(enumerate(losses))
                            k = max(1, int(0.3 * len(losses)))
                            if selector.__name__ == "quickselect":
                                top_k = selector(arr, k)
                                indices = [i for i, _ in top_k]
                            else:
                                # sort o heap simples
                                top_k = selector(arr, k) if callable(selector) else arr
                                indices = [i for i, _ in top_k]
                        X_batch = X_batch[indices]
                        y_batch = y_batch[indices]
                        pred = self.model.forward(X_batch)
                        losses = -np.sum(y_batch * np.log(pred + 1e-9), axis=1)

                    # Backward y update
                    self.model.backward(y_batch)
                    self.model.update(self.lr)
                    total_loss += np.sum(losses)
                    num_batches += 1
            else:
                # Sin cola
                for X_batch, y_batch in create_batches(X, y, batch_size):
                    # Aumentación de Datos
                    if augment in ['digits', 'mnist']:
                        from utils.augmentation import augment_batch
                        size = int(np.sqrt(X_batch.shape[1]))
                        X_img = X_batch.reshape(-1, size, size)
                        X_img_aug = augment_batch(X_img, dataset=augment, prob=0.5)
                        X_batch = X_img_aug.reshape(-1, size * size)
                    

                    pred = self.model.forward(X_batch)
                    losses = -np.sum(y_batch * np.log(pred + 1e-9), axis=1)
                    pred_labels = np.argmax(pred, axis=1)
                    true_labels = np.argmax(y_batch, axis=1)
                    correct += np.sum(pred_labels == true_labels)
                    total += len(y_batch)
                    if selector:
                        arr = list(enumerate(losses))
                        k = max(1, int(0.3 * len(losses)))
                        if selector.__name__ == "quickselect":
                            top_k = selector(arr, k)
                            indices = [i for i, _ in top_k]
                        else:
                            top_k = selector(arr, k) if callable(selector) else arr
                            indices = [i for i, _ in top_k]
                        X_batch = X_batch[indices]
                        y_batch = y_batch[indices]
                        pred = self.model.forward(X_batch)
                        losses = -np.sum(y_batch * np.log(pred + 1e-9), axis=1)
                    self.model.backward(y_batch)
                    self.model.update(self.lr)
                    total_loss += np.sum(losses)
                    num_batches += 1

            epoch_time = time.time() - start
            avg_loss = total_loss / num_batches
            accuracy = correct / total
            self.loss_history.append(avg_loss)
            self.time_history.append(epoch_time)
            self.acc_history.append(accuracy)
            msg = f"[{epoch}] Loss={avg_loss:.4f} Acc={accuracy:.4f} Time={epoch_time:.4f}s"
            print(msg)
            if telegram:
                telegram.send(msg)