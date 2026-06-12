import sys
import os
import numpy as np
import time
import psutil
from datetime import datetime
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, filters, ContextTypes
from telegram.request import HTTPXRequest
from PIL import Image
import io

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Nucleo.mlp import MLP
from utils.Bot_Telegram import TelegramBot
from utils.Complejidad import get_complexity
from utils.config import TOKEN, LOG_TOKEN, LOG_CHAT_ID

logger = TelegramBot(token=LOG_TOKEN, chat_id=LOG_CHAT_ID) if LOG_TOKEN else None

current_model = None
current_dataset = None
current_selector = "heap"
input_size = 0

# Utilidades
def is_bot_mentioned(update: Update, bot_username: str) -> bool:
    if update.message.chat.type == "private":
        return True
    if update.message.text and update.message.entities:
        for entity in update.message.entities:
            if entity.type == "mention":
                mention = update.message.text[entity.offset:entity.offset+entity.length]
                if mention == f"@{bot_username}":
                    return True
    if update.message.caption and update.message.caption_entities:
        for entity in update.message.caption_entities:
            if entity.type == "mention":
                mention = update.message.caption[entity.offset:entity.offset+entity.length]
                if mention == f"@{bot_username}":
                    return True
    return False

def is_numeric_list(text):
    text = text.replace(',', ' ').strip()
    parts = text.split()
    if not parts:
        return False
    try:
        for p in parts:
            float(p)
        return True
    except ValueError:
        return False

def load_model(dataset, selector):
    global current_model, current_dataset, current_selector, input_size
    path = f"model_{selector}_{dataset}.npz"
    if not os.path.exists(path):
        return False, f"No existe el archivo {path}. Primero ejecuta run.py para entrenar."
    data = np.load(path)
    input_size = data["W1"].shape[0]
    output_size = data["W2"].shape[1]
    model = MLP(input_size, 32, output_size)
    model.load(path)
    current_model = model
    current_dataset = dataset
    current_selector = selector
    return True, f"Modelo cargado: {dataset.upper()} | selector={selector.upper()} | entradas={input_size}"

def reload_current():
    if current_dataset and current_selector:
        return load_model(current_dataset, current_selector)
    return False, "No hay dataset cargado."

def preprocess_image(image_bytes, target_size):
    from PIL import Image, ImageOps, ImageFilter
    import numpy as np

    #  Abrir y convertir a grises
    img = Image.open(io.BytesIO(image_bytes)).convert('L')
    
    #  Redimensionar a un tamaño intermedio para mejorar la calidad
    img = img.resize((max(target_size)*4, max(target_size)*4), Image.Resampling.LANCZOS)
    
    #  Convertir a array numpy
    arr = np.array(img, dtype=np.uint8)
    
    #  Invertir si el fondo es claro (media es mayor a 127)
    if arr.mean() > 127:
        arr = 255 - arr
    
    # Aplicar un umbral adaptativo o fijo para binarizar
    # Usamos un umbral fijo (50) para eliminar ruido de fondo
    thresh = 50
    arr = (arr > thresh).astype(np.uint8) * 255
    
    #  Encontrar el bounding box del dígito
    coords = np.argwhere(arr > 0)
    if len(coords) == 0:
        return np.zeros(target_size[0] * target_size[1], dtype=np.float32)
    
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Añadir un pequeño margen (5 píxeles)
    margin = 5
    y_min = max(0, y_min - margin)
    y_max = min(arr.shape[0], y_max + margin)
    x_min = max(0, x_min - margin)
    x_max = min(arr.shape[1], x_max + margin)
    
    digit = arr[y_min:y_max, x_min:x_max]
    
    #  Convertir a PIL y aplicar padding para centrar el dígito
    pil_digit = Image.fromarray(digit)
    
    # Calcular la relación de aspecto
    w, h = pil_digit.size
    if w > h:
        new_w = target_size[0]
        new_h = int(h * (new_w / w))
    else:
        new_h = target_size[1]
        new_w = int(w * (new_h / h))
    
    pil_digit = pil_digit.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Crear un fondo negro del tamaño objetivo
    padded = Image.new("L", target_size, 0)
    # Pegar la imagen centrada
    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2
    padded.paste(pil_digit, (x_offset, y_offset))
    
    #  Normalizar a [0,1]
    final_arr = np.array(padded, dtype=np.float32) / 255.0
    return final_arr.flatten()


# Comandos
async def start_command(update: Update):
    await update.message.reply_text(
        "Hola, Soy Cypher, un bot de clasificacion multi-dataset.\n\n"
        "Estoy entrenado para reconocer digitos escritos a mano (Digits y MNIST) "
        "y clasificar flores (Iris) y vinos (Wine) usando una red neuronal MLP implementada desde cero.\n\n"
        "Mis capacidades:\n"
        "- Clasificar imagenes (envia una foto y te dire que numero es)\n"
        "- Clasificar listas de numeros (pixeles o caracteristicas numericas)\n"
        "- Usar diferentes selectores de hard mining (heap, sort, quickselect)\n"
        "- Cambiar entre datasets: Digits (64 numeros), Iris (4), Wine (13), MNIST (784)\n\n"
        "Usa /help para ver la lista completa de comandos.\n"
        "Empieza cargando un dataset con /digits, /iris, /wine o /mnist"
    )

async def help_command(update: Update):
    await update.message.reply_text(
        "Cypher - Clasificacion multi-dataset\n\n"
        "Comandos:\n"
        "/digits   - Cargar dataset Digits (64 numeros)\n"
        "/iris     - Cargar dataset Iris (4 numeros)\n"
        "/wine     - Cargar dataset Wine (13 numeros)\n"
        "/mnist    - Cargar dataset MNIST (784 numeros)\n"
        "/heap     - Usar selector HEAP (hard mining)\n"
        "/sort     - Usar selector SORT\n"
        "/quick    - Usar selector QUICK\n"
        "/baseline - Usar selector BASELINE\n"
        "/status   - Mostrar modelo actual\n"
        "/reset    - Reiniciar el bot\n"
        "/info     - Informacion del proyecto\n"
        "/about    - Creditos\n"
        "/metrics  - Estadisticas del sistema\n"
        "/chatid   - Obtener ID de este chat\n"
        "/help     - Esta ayuda\n\n"
        "Despues de cargar un dataset, puedes enviarme una lista de numeros o una foto (Digits/MNIST)."
    )

async def reset_command(update: Update):
    global current_model, current_dataset, current_selector, input_size
    current_model = None
    current_dataset = None
    current_selector = "heap"
    input_size = 0
    await update.message.reply_text(
        "Estado reiniciado. Ya no hay ningun modelo cargado.\n"
        "Usa /digits, /iris, /wine o /mnist para cargar un dataset."
    )

async def status_command(update: Update):
    if current_model is None:
        await update.message.reply_text("No hay modelo cargado. Usa /digits, /iris, /wine o /mnist")
    else:
        await update.message.reply_text(
            f"Modelo actual:\n"
            f" - Dataset: {current_dataset.upper()}\n"
            f" - Selector: {current_selector.upper()}\n"
            f" - Caracteristicas: {input_size}\n"
            f" - Clases: {current_model.W2.shape[1]}"
        )

async def chatid_command(update: Update):
    chat_id = update.effective_chat.id
    await update.message.reply_text(f"ID de este chat: {chat_id}")

async def info_command(update: Update):
    await update.message.reply_text(
        "Proyecto final ADA\n"
        "Red neuronal MLP desde cero con:\n"
        " - Hard mining (heap, sort, quickselect)\n"
        " - Estructuras: cola de lotes, hash de perdidas, heap de poda\n"
        " - Baseline k-NN\n"
        " - Analisis de complejidad y recurrencias"
    )

async def about_command(update: Update):
    await update.message.reply_text(
        "Cypher v1.0\n"
        "Desarrollado para Analisis y Diseno de Algoritmos 1\n"
        "Universidad del Valle\n"
        "(c) 2026"
    )

async def metrics_command(update: Update):
    cpu = psutil.cpu_percent(interval=0.5)
    mem = psutil.virtual_memory()
    await update.message.reply_text(
        f"Sistema\n"
        f" - CPU: {cpu}%\n"
        f" - RAM: {mem.percent}% (usada {mem.used//(1024**2)} MB / {mem.total//(1024**2)} MB)\n"
        f" - Tiempo activo: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

async def digits_command(update: Update):
    ok, msg = load_model("digits", current_selector)
    await update.message.reply_text(msg)

async def iris_command(update: Update):
    ok, msg = load_model("iris", current_selector)
    await update.message.reply_text(msg)

async def wine_command(update: Update):
    ok, msg = load_model("wine", current_selector)
    await update.message.reply_text(msg)

async def mnist_command(update: Update):
    ok, msg = load_model("mnist", current_selector)
    await update.message.reply_text(msg)

async def heap_command(update: Update):
    global current_selector
    current_selector = "heap"
    ok, msg = reload_current()
    await update.message.reply_text(f"Selector cambiado a HEAP.\n{msg}" if ok else "Selector cambiado a HEAP. Carga un dataset.")

async def sort_command(update: Update):
    global current_selector
    current_selector = "sort"
    ok, msg = reload_current()
    await update.message.reply_text(f"Selector cambiado a SORT.\n{msg}" if ok else "Selector cambiado a SORT. Carga un dataset.")

async def quick_command(update: Update):
    global current_selector
    current_selector = "quick"
    ok, msg = reload_current()
    await update.message.reply_text(f"Selector cambiado a QUICK.\n{msg}" if ok else "Selector cambiado a QUICK. Carga a dataset.")

async def baseline_command(update: Update):
    global current_selector
    current_selector = "baseline"
    ok, msg = reload_current()
    await update.message.reply_text(f"Selector cambiado a BASELINE.\n{msg}" if ok else "Selector cambiado a BASELINE. Carga un dataset.")

# Handler principal
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"MENSAJE RECIBIDO: {update.message.text}")
    if not is_bot_mentioned(update, context.bot.username):
        print("No mencionado o no es privado, ignorando")
        return
    print("Continuando")

    if update.message.text:
        text = update.message.text
    elif update.message.caption:
        text = update.message.caption
    else:
        return

    if text.startswith(f"@{context.bot.username}"):
        text = text[len(f"@{context.bot.username}"):].strip()

    if not text:
        return

    # Procesar comandos
    if text.startswith('/'):
        parts = text.split()
        command = parts[0].lower()
        if command == '/start':
            await start_command(update)
        elif command == '/help':
            await help_command(update)
        elif command == '/reset':
            await reset_command(update)
        elif command == '/status':
            await status_command(update)
        elif command == '/chatid':
            await chatid_command(update)
        elif command == '/info':
            await info_command(update)
        elif command == '/about':
            await about_command(update)
        elif command == '/metrics':
            await metrics_command(update)
        elif command == '/digits':
            await digits_command(update)
        elif command == '/iris':
            await iris_command(update)
        elif command == '/wine':
            await wine_command(update)
        elif command == '/mnist':
            await mnist_command(update)
        elif command == '/heap':
            await heap_command(update)
        elif command == '/sort':
            await sort_command(update)
        elif command == '/quick':
            await quick_command(update)
        elif command == '/baseline':
            await baseline_command(update)
        else:
            await update.message.reply_text("Comando no reconocido. Usa /help para ver comandos.")
        return

    # Si no es comando, tratar como lista de números
    if current_model is None:
        await update.message.reply_text("Primero carga un dataset con /digits, /iris, /wine o /mnist")
        return

    if not is_numeric_list(text):
        await update.message.reply_text(
            "Envia una lista de numeros (separados por espacios o comas) para clasificar.\n"
            "Ejemplo: 0.1 0.2 0.3 ...\n"
            "O usa /help para ver comandos."
        )
        return

    try:
        numbers = [float(x) for x in text.replace(',', ' ').split()]
        if len(numbers) != input_size:
            await update.message.reply_text(f"Se requieren {input_size} numeros. Enviaste {len(numbers)}.")
            return
        x = np.array(numbers).reshape(1, -1)
        x = (x - current_model.x_min) / (current_model.x_max - current_model.x_min + 1e-8)
        start_time = time.time()
        pred = current_model.forward(x)
        clase = int(np.argmax(pred))
        elapsed = time.time() - start_time
        complexity = get_complexity(current_selector)
        reply = f"Prediccion: {clase}\nTiempo: {elapsed:.4f}s\nComplejidad: {complexity}"
        await update.message.reply_text(reply)
        if logger:
            logger.send(f"Texto - {current_selector}_{current_dataset}: {clase} en {elapsed:.4f}s")
    except Exception as e:
        await update.message.reply_text(f"Error: {str(e)}")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_bot_mentioned(update, context.bot.username):
        return
    if current_model is None:
        await update.message.reply_text("Primero carga un dataset con /digits o /mnist")
        return
    if current_dataset not in ["digits", "mnist"]:
        await update.message.reply_text(f"El dataset {current_dataset} no soporta imagenes. Usa numeros.")
        return
    photo_file = await update.message.photo[-1].get_file()
    image_bytes = await photo_file.download_as_bytearray()
    target_size = (8, 8) if current_dataset == "digits" else (28, 28)
    try:
        arr = preprocess_image(image_bytes, target_size)
        if len(arr) != input_size:
            await update.message.reply_text(f"Error: imagen procesada tiene {len(arr)} pixeles, modelo espera {input_size}.")
            return
        x = arr.reshape(1, -1)
        x = (x - current_model.x_min) / (current_model.x_max - current_model.x_min + 1e-8)
        start_time = time.time()
        pred = current_model.forward(x)
        clase = int(np.argmax(pred))
        elapsed = time.time() - start_time
        complexity = get_complexity(current_selector)
        reply = f"Imagen clasificada como: {clase}\nTiempo: {elapsed:.4f}s\nComplejidad: {complexity}"
        await update.message.reply_text(reply)
        if logger:
            logger.send(f"Imagen - {current_selector}_{current_dataset}: {clase}")
    except Exception as e:
        await update.message.reply_text(f"Error al procesar imagen: {str(e)}")

# Main
def main():
    request = HTTPXRequest(connect_timeout=30.0, read_timeout=30.0, write_timeout=30.0, pool_timeout=30.0)
    app = ApplicationBuilder().token(TOKEN).request(request).build()

    # Todos los mensajes de texto (incluyendo comandos) van a handle_message
    app.add_handler(MessageHandler(filters.TEXT, handle_message))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    print("Cypher iniciado. Presiona Ctrl + C para detener.")
    app.run_polling()

if __name__ == "__main__":
    main()