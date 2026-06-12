def get_input_type(mode):

    return {
        "digits": "Tabla (UCI / sklearn)",
        "iris": "Tabla (UCI / sklearn)",
        "wine": "Tabla (UCI / sklearn)",
        "bot": "Texto (Telegram input)",
        "future_image": "Imagen (futuro CNN)"
    }.get(mode, "Desconocido")