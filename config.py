import os
from dotenv import load_dotenv

load_dotenv()

GMAIL_CREDENTIALS_FILE = os.getenv("GMAIL_CREDENTIALS_FILE", "credentials.json")
GMAIL_TOKEN_FILE = os.getenv("GMAIL_TOKEN_FILE", "token.json")
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "deep-work-secretary")
NTFY_SERVER = os.getenv("NTFY_SERVER", "https://ntfy.sh")
URGENCY_THRESHOLD = float(os.getenv("URGENCY_THRESHOLD", "0.7"))
FOCUS_URGENCY_THRESHOLD = float(os.getenv("FOCUS_URGENCY_THRESHOLD", "0.8"))
GOOGLE_CALENDAR_ID = os.getenv("GOOGLE_CALENDAR_ID", "primary")
MLP_MODEL_PATH = os.getenv("MLP_MODEL_PATH", "mlp_model.pt")
TEXT_MODEL_PATH = os.getenv("TEXT_MODEL_PATH", "text_model.pt")
