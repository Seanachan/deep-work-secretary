import requests
from config import NTFY_SERVER, NTFY_TOPIC


def send_notification(title: str, message: str, priority: str = "high") -> bool:
    url = f"{NTFY_SERVER}/{NTFY_TOPIC}"
    headers = {
        "Title": title,
        "Priority": priority,
        "Tags": "email,urgent",
    }

    try:
        response = requests.post(url, data=message.encode('utf-8'), headers=headers)
        return response.status_code == 200
    except Exception:
        return False
