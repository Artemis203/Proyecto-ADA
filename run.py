import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from Nucleo.mlp import MLP
from Entreno.trainer import Trainer
from utils.Dataset_Gestor import load_dataset
from Algoritmos.topk import top_k_heap, top_k_sort
from Algoritmos.quickselect import quickselect
from Analiticas.grafica import Graficar_datos
from utils.Bot_Telegram import TelegramBot
from utils.config import LOG_TOKEN, LOG_CHAT_ID

# Configuracion Global de los Entrenos
LEARNING_RATE = 0.005       
BATCH_SIZE = 32
EPOCHS = 1000              
HIDDEN_SIZE = 128
USE_QUEUE = True

logger = TelegramBot(token=LOG_TOKEN, chat_id=LOG_CHAT_ID) if LOG_TOKEN else None

datasets_config = [
    {"name": "digits", "reduce_mnist": False, "mnist_samples": None},
    {"name": "iris",   "reduce_mnist": False, "mnist_samples": None},
    {"name": "wine",   "reduce_mnist": False, "mnist_samples": None},
    {"name": "mnist",  "reduce_mnist": True,  "mnist_samples": 5000}
]

experimentos = {
    "baseline": None,
    "heap": top_k_heap,
    "sort": top_k_sort,
    "quick": quickselect
}

for cfg in datasets_config:
    dataset_name = cfg["name"]
    print("\n" + "="*60)
    print(f"PROCESANDO DATASET: {dataset_name.upper()}")
    print("="*60)

    X, y, X_min, X_max = load_dataset(
        dataset_name,
        reduce_mnist=cfg["reduce_mnist"],
        mnist_samples=cfg["mnist_samples"] if cfg["mnist_samples"] else 0
    )
    print(f"Dataset shape: {X.shape}, clases: {y.shape[1]}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # k-NN baseline
    print(f"\nEntrenando k-NN (k=5) para {dataset_name}...")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, np.argmax(y_train, axis=1))
    y_pred_knn = knn.predict(X_test)
    knn_acc = accuracy_score(np.argmax(y_test, axis=1), y_pred_knn)
    print(f"k-NN Test Accuracy: {knn_acc:.4f}")

    resultados_loss, resultados_time, resultados_acc, resultados_test = {}, {}, {}, {}

    # Aumentación solo para imágenes (descomentar None si se desea NO usar la aumentación de Datos)
    augment_dataset = dataset_name if dataset_name in ["digits", "mnist"] else None
    #augment_dataset = None


    for nombre, selector in experimentos.items():
        print(f"\n=== Entrenando MLP con selector {nombre} para {dataset_name} ===")
        model = MLP(input_size=X.shape[1], hidden_size=HIDDEN_SIZE, output_size=y.shape[1])
        model_path = f"model_{nombre}_{dataset_name}.npz"
        if os.path.exists(model_path):
            print(f"Cargando modelo existente: {model_path}")
            model.load(model_path)
        else:
            print(f"No existe {model_path}, se creará uno nuevo.")

        trainer = Trainer(model, lr=LEARNING_RATE)   
        trainer.train(
            X_train, y_train,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            selector=selector,
            use_queue=USE_QUEUE,
            augment=augment_dataset
        )
        model.save(model_path, X_min, X_max)
        print(f"Modelo guardado: {model_path}")

        pred = model.forward(X_test)
        test_acc = np.mean(np.argmax(pred, axis=1) == np.argmax(y_test, axis=1))
        resultados_test[nombre] = test_acc
        resultados_loss[nombre] = trainer.loss_history
        resultados_time[nombre] = trainer.time_history
        resultados_acc[nombre] = trainer.acc_history
        print(f"Test accuracy: {test_acc:.4f}")

        if logger:
            msg = (
                f"Entrenamiento finalizado\n\n"
                f"Dataset: {dataset_name.upper()}\n"
                f"Modelo: {nombre.upper()}\n"
                f"Accuracy maxima: {max(trainer.acc_history):.4f}\n"
                f"Loss minima: {min(trainer.loss_history):.4f}\n"
                f"Tiempo promedio: {np.mean(trainer.time_history):.4f}s\n"
                f"Epocas: {len(trainer.loss_history)}"
            )
            logger.send(msg)

    # Generar gráficas con valores reales
    epocas_reales = len(resultados_loss[next(iter(resultados_loss))])

    Graficar_datos(
        resultados_test,
        resultados_loss,
        resultados_time,
        resultados_acc,
        epochs=epocas_reales,          
        batch_size=BATCH_SIZE,          
        learning_rate=LEARNING_RATE,   
        dataset_name=dataset_name,
        input_mode=dataset_name,
        input_size=X.shape[1],
        output_size=y.shape[1]
    )

    print(f"\n=== RESULTADOS FINALES PARA {dataset_name.upper()} ===")
    print(f"k-NN test accuracy: {knn_acc:.4f}")
    for name, acc in resultados_test.items():
        print(f"MLP + {name:5} test accuracy: {acc:.4f}")

print("\n" + "="*60)
print("ENTRENAMIENTO COMPLETADO PARA TODOS LOS DATASETS")
print("="*60)