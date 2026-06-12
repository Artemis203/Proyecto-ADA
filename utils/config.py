import os
from dotenv import load_dotenv

load_dotenv()

def get_env(name: str, required=True):
    value = os.getenv(name)

    if required and not value:
        raise ValueError(f"Falta variable de entorno: {name}")

    return value


TOKEN = get_env("TELEGRAM_TOKEN")

LOG_TOKEN = get_env("LOG_TOKEN", required=False)
LOG_CHAT_ID = get_env("LOG_CHAT_ID", required=False)