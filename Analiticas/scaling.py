import numpy as np
import time
import matplotlib.pyplot as plt
from Nucleo.mlp import MLP
from Entreno.trainer import Trainer
from utils.Dataset_Gestor import load_dataset

def run_scaling_experiment():
    X, y, _, _ = load_dataset("mnist", reduce_mnist=True, mnist_samples=2000)  # pequeño para pruebas
    input_size = X.shape[1]
    output_size = y.shape[1]
    
    batch_sizes = [16, 32, 64, 128]
    hidden_sizes = [16, 32, 64, 128]
    epochs_fixed = 5  # pocas épocas para escalado
    
    results = {}
    
    # Variar batch size (hidden fijo)
    hidden_fixed = 32
    times_batch = []
    for bs in batch_sizes:
        model = MLP(input_size, hidden_fixed, output_size)
        trainer = Trainer(model, lr=0.01)
        start = time.time()
        trainer.train(X, y, epochs=epochs_fixed, batch_size=bs, selector=None, use_queue=False)
        elapsed = time.time() - start
        times_batch.append(elapsed)
        print(f"Batch {bs} -> {elapsed:.2f}s")
    
    # Variar hidden size (batch fijo 32)
    batch_fixed = 32
    times_hidden = []
    for hs in hidden_sizes:
        model = MLP(input_size, hs, output_size)
        trainer = Trainer(model, lr=0.01)
        start = time.time()
        trainer.train(X, y, epochs=epochs_fixed, batch_size=batch_fixed, selector=None, use_queue=False)
        elapsed = time.time() - start
        times_hidden.append(elapsed)
        print(f"Hidden {hs} -> {elapsed:.2f}s")
    
    # Gráficas
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(batch_sizes, times_batch, 'o-')
    plt.xlabel("Batch size")
    plt.ylabel("Tiempo total (s)")
    plt.title("Escalado con batch size")
    plt.subplot(1,2,2)
    plt.plot(hidden_sizes, times_hidden, 'o-')
    plt.xlabel("Neuronas ocultas")
    plt.ylabel("Tiempo total (s)")
    plt.title("Escalado con hidden size")
    plt.tight_layout()
    plt.savefig("scaling_ra1.png")
    plt.show()

if __name__ == "__main__":
    run_scaling_experiment()