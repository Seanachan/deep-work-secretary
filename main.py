import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from config import URGENCY_THRESHOLD, FOCUS_URGENCY_THRESHOLD, MLP_MODEL_PATH
from email_fetcher import build_gmail_service, fetch_email_snippets
from models.mlp import load_or_init_model
from calendar_checker import build_calendar_service, is_focus_block_now
from scorer import get_final_score
from notifier import send_notification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Deep Work Secretary started")
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/triage")
def triage():
    gmail_service = build_gmail_service()
    emails = fetch_email_snippets(gmail_service)

    mlp_model = load_or_init_model(MLP_MODEL_PATH)

    calendar_service = build_calendar_service()
    focus_block = is_focus_block_now(calendar_service)

    threshold = FOCUS_URGENCY_THRESHOLD if focus_block else URGENCY_THRESHOLD

    results = []
    urgent_count = 0

    for email in emails:
        score = get_final_score(email, mlp_model)
        notified = False

        if score >= threshold * 100:
            notified = send_notification(
                title=f"Urgent Email from {email['from']}",
                message=email['snippet'],
            )
            urgent_count += 1

        results.append({
            "id": email['id'],
            "from": email['from'],
            "score": score,
            "notified": notified,
        })

    return {
        "processed": len(emails),
        "urgent": urgent_count,
        "focus_block": focus_block,
        "results": results,
    }
