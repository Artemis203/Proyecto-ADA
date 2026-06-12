import requests
from datetime import datetime

from utils.config import (
    LOG_TOKEN,
    LOG_CHAT_ID
)


class TelegramBot:

    def __init__(
        self,
        token=LOG_TOKEN,
        chat_id=LOG_CHAT_ID):

        self.token = token
        self.chat_id = chat_id

    # Validar configuración
    def configured(self):

        return bool(
            self.token and self.chat_id
        )

    # Enviar mensaje
    def send(self, msg):

        # Verificar configuración
        if not self.configured():

            print(
                "[TelegramBot] Bot no configurado"
            )

            return False

        url = (
            f"https://api.telegram.org/"
            f"bot{self.token}/sendMessage"
        )

        timestamp = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        formatted_msg = (
            f"[{timestamp}]\n\n{msg}"
        )

        try:

            response = requests.post(
                url,
                data={
                    "chat_id": self.chat_id,
                    "text": formatted_msg,
                    #"parse_mode": "Markdown"
                },
                timeout=5
            )

            if response.status_code != 200:

                print("[TelegramBot] Error HTTP:")
                print(response.text)

                return False

            data = response.json()

            if not data.get("ok"):

                print("[TelegramBot] Error API Telegram:")
                print(data)

                return False

            return True

        except requests.exceptions.Timeout:

            print("[TelegramBot] Timeout de conexión")
            return False

        except requests.exceptions.ConnectionError:

            print("[TelegramBot] Sin conexión a internet")
            return False

        except requests.exceptions.RequestException as e:

            print(f"[TelegramBot] Error request: {e}")
            return False

        except Exception as e:

            print(f"[TelegramBot] Error inesperado: {e}")
            return False

    # Probar conexión
    def test(self):

        return self.send(
            "Bot conectado correctamente"
        )