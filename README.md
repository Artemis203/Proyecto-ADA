# Proyecto Final – Red Neuronal MLP desde cero

Clasificación de imágenes MNIST con un perceptrón multicapa (MLP) implementado manualmente, sin frameworks de alto nivel. Incluye análisis de complejidad, estructuras de datos (heap, cola, hash) y un bot de Telegram para inferencia.

## Características

- MLP con una capa oculta, activaciones ReLU y softmax, entrenado con SGD.
- Hard mining: selección del top‑k de pérdidas por batch usando **heap**, **sort** o **quickselect**.
- Estructuras auxiliares: cola de lotes (deque), hash de pérdidas y heap para poda.
- Baseline: k‑NN (k=5) para comparación.
- Soporta datasets: MNIST (reducido), Digits, Iris, Wine.
- Bot de Telegram para clasificar nuevas muestras.
- Generación de gráficas de pérdida, accuracy y tiempo de entrenamiento.
- Análisis de recurrencias: Quickselect para mediana (T(n)=T(n/2)+O(n) → O(n)).

## Requisitos

- Python 3.8 o superior
- pip

## Instalación

1. Clona o descarga el proyecto y accede a la carpeta `Proyecto-ADA`.
2. Usar `pip install -r requirements.txt` para instalar los paquetes necesarios.
3. (Opcional) Crea y activa un entorno virtual:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Mac:
   source venv/bin/activate
4. Si ya tienes creado el .venv, entonces solo ejecutar
  ```bash
  .\venv\Scripts\Activate.ps1  
 ```
Siempre teniendo en cuenta que debes de estar en la ruta del proyecto en el punto 1 que se indico.

## Ejecución
1. Usar el comando `python run.py`
2. Recuerda crear tu propio .env para los bots.

## Creación de env
 Usar un formato como el de abajo para conectar los Token
 ```bash
 TELEGRAM_TOKEN=token_de_bot_interactivo
 LOG_TOKEN=token_de_logs
 LOG_CHAT_ID=chat_id
 ```