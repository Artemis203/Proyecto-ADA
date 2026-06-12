import os
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime
from utils.Entradas import get_input_type
from utils.Complejidad import (
    get_complexity
)


# Suavizado
def smooth(data, window=30):

    smoothed = []

    for i in range(len(data)):

        start = max(0, i - window)

        smoothed.append(
            np.mean(data[start:i + 1])
        )

    return smoothed


# Graficar resultados
def Graficar_datos(
        resultados_test,
        resultados_loss,
        resultados_time,
        resultados_acc,
        epochs,
        batch_size,
        learning_rate,
        dataset_name,
        input_mode,
        input_size,
        output_size):

    # Estilo de gráficas
    plt.style.use("ggplot")

    # Colores de algoritmos
    colors = {
        "baseline": "#222222",
        "heap": "#1f77b4",
        "sort": "#2ca02c",
        "quick": "#d62728"
    }

    # Fecha y hora
    now = datetime.now()

    fecha = now.strftime("%Y-%m-%d")
    hora = now.strftime("%H-%M-%S")

    # Carpeta de resultados
    folder = (
    f"Analiticas/Resultados/"
    f"{dataset_name}_"
    f"{fecha}_{hora}"
    )

    os.makedirs(folder, exist_ok=True)

    # Archivo de reporte
    report_path = f"{folder}/reporte.txt"

    report = []

    # Información general
    report.append("Reporte de entrenamiento")
    report.append("-" * 40)

    report.append(f"Dataset: {dataset_name}")
    report.append(f"Tipo de input: {get_input_type(input_mode)}")

    report.append(f"Fecha: {fecha}")
    report.append(f"Hora: {hora}")

    report.append(f"Épocas: {epochs}")
    report.append(f"Batch size: {batch_size}")
    report.append(f"Learning rate: {learning_rate}")
    report.append(
    f"Dataset: {dataset_name}"
    )
    report.append(
    f"Input size: "
    f"{input_size}"
    )

    report.append(
        f"Output size: "
        f"{output_size}"
    )

    report.append(
        f"Arquitectura: "
        f"{input_size} -> 32 -> {output_size}"
    )

    report.append("")

    # Épocas reales
    epochs_real = len(
        next(iter(resultados_loss.values()))
    )

    x = range(epochs_real)

    # Gráfica de pérdida
    plt.figure(figsize=(12, 6))

    for nombre, loss in resultados_loss.items():

        plt.plot(
            x,
            smooth(loss),
            label=nombre.upper(),
            color=colors[nombre],
            linewidth=2
        )

    plt.title(
        "Pérdida vs Épocas",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel("Épocas")
    plt.ylabel("Loss")

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        f"{folder}/loss.png",
        dpi=300
    )

    plt.show()
    plt.close()
    
    # Gráfica de Precisión/accuracy
    plt.figure(figsize=(12, 6))

    for nombre, acc in resultados_acc.items():

        plt.plot(
            x,
            smooth(acc),
            label=nombre.upper(),
            color=colors[nombre],
            linewidth=2
        )

    plt.title(
        "Accuracy vs Épocas",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel("Épocas")
    plt.ylabel("Accuracy")

    plt.ylim(0, 1)

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        f"{folder}/accuracy.png",
        dpi=300
    )

    plt.show()
    plt.close()

    # Gráfica de tiempo
    plt.figure(figsize=(12, 6))

    for nombre, tiempos in resultados_time.items():

        plt.plot(
            x,
            smooth(tiempos),
            label=nombre.upper(),
            color=colors[nombre],
            linewidth=2
        )

    plt.title(
        "Tiempo vs Épocas",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel("Épocas")
    plt.ylabel("Tiempo (s)")

    plt.legend()

    plt.grid(
        True,
        alpha=0.3
    )

    plt.tight_layout()

    plt.savefig(
        f"{folder}/time.png",
        dpi=300
    )

    plt.show()
    plt.close()

    # Resumen final
    print("\n--------------------------------------")
    print("Resumen final de métricas")
    print("--------------------------------------")

    best_acc_name = None
    best_acc_value = -1

    best_time_name = None
    best_time_value = 999999

    for nombre in resultados_acc.keys():

        max_acc = max(
            resultados_acc[nombre]
        )

        min_loss = min(
            resultados_loss[nombre]
        )

        avg_time = np.mean(
            resultados_time[nombre]
        )

        complexity = get_complexity(
        nombre)
        report.append(f"Complejidad: {complexity}")
        
            

        print(f"\nMétodo: {nombre.upper()}")

        print(
            f"Accuracy máxima : "
            f"{max_acc:.4f}"
        )

        print(
            f"Loss mínima     : "
            f"{min_loss:.4f}"
        )

        print(
            f"Tiempo promedio : "
            f"{avg_time:.4f}s"
        )

        print(
        f"Complejidad     : "
        f"{complexity}"
        )

        # Guardar en TXT
        report.append(f"Método: {nombre.upper()}")

        report.append(
            f"Accuracy máxima : "
            f"{max_acc:.4f}"
        )

        report.append(
            f"Loss mínima     : "
            f"{min_loss:.4f}"
        )

        report.append(
            f"Tiempo promedio : "
            f"{avg_time:.4f}s"
        )

        report.append(
        f"Complejidad     : "
        f"{complexity}"
        )

        report.append("")

        if max_acc > best_acc_value:

            best_acc_value = max_acc
            best_acc_name = nombre

        if avg_time < best_time_value:

            best_time_value = avg_time
            best_time_name = nombre

    # Mejores algoritmos
    print("\n--------------------------------------")
    print("Mejores métodos")
    print("--------------------------------------")

    print(
        f"Mayor accuracy : "
        f"{best_acc_name.upper()} "
        f"({best_acc_value:.4f})"
    )

    print(
        f"Más rápido     : "
        f"{best_time_name.upper()} "
        f"({best_time_value:.4f}s)"
    )

    report.append("Mejores métodos")
    report.append("-" * 40)

    report.append(
        f"Mayor accuracy: "
        f"{best_acc_name.upper()} "
        f"({best_acc_value:.4f})"
    )

    report.append(
        f"Más rápido: "
        f"{best_time_name.upper()} "
        f"({best_time_value:.4f}s)"
    )

    report.append("")

    # Diferencias
    print("\n--------------------------------------")
    print("Diferencias")
    print("--------------------------------------")

    report.append("Diferencias")
    report.append("-" * 40)

    baseline_time = np.mean(
        resultados_time["baseline"]
    )

    for nombre in resultados_time.keys():

        if nombre == "baseline":
            continue

        current = np.mean(
            resultados_time[nombre]
        )

        diff = (
            (baseline_time - current)
            / baseline_time
        ) * 100

        print(
            f"{nombre.upper()} fue "
            f"{diff:.2f}% más rápido "
            f"que baseline"
        )

        report.append(
            f"{nombre.upper()} fue "
            f"{diff:.2f}% más rápido "
            f"que baseline"
        )

    # Guardar TXT
    with open(
            report_path,
            "w",
            encoding="utf-8") as file:

        file.write(
            "\n".join(report)
        )

    print(f"\nGráficas guardadas en:\n{folder}")

    print(f"\nReporte guardado en:\n{report_path}")