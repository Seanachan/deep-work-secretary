import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from config import (
    URGENCY_THRESHOLD,
    FOCUS_URGENCY_THRESHOLD,
    MLP_MODEL_PATH,
    POLL_INTERVAL,
)
from email_fetcher import build_gmail_service, fetch_email_snippets
from models.mlp import load_or_init_model
from calendar_checker import build_calendar_service, is_focus_block_now
from scorer import get_final_score
from notifier import send_notification

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Track email IDs we've already processed to avoid duplicate notifications
seen_ids: set[str] = set()


def run_triage(only_new: bool = False):
    """Run a single triage pass. If only_new=True, skip already-seen emails."""
    gmail_service = build_gmail_service()
    emails = fetch_email_snippets(gmail_service)

    mlp_model = load_or_init_model(MLP_MODEL_PATH)

    calendar_service = build_calendar_service()
    focus_block = is_focus_block_now(calendar_service)

    threshold = FOCUS_URGENCY_THRESHOLD if focus_block else URGENCY_THRESHOLD

    results = []
    urgent_count = 0

    for email in emails:
        is_new = email['id'] not in seen_ids
        seen_ids.add(email['id'])

        if only_new and not is_new:
            continue

        score = get_final_score(email, mlp_model)
        notified = False

        sender = email['from'].lower()
        is_noreply = 'noreply' in sender or 'no-reply' in sender

        if score >= threshold * 100 and not is_noreply:
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
            "new": is_new,
        })

    return {
        "processed": len(results),
        "urgent": urgent_count,
        "focus_block": focus_block,
        "results": results,
    }


async def poll_loop():
    """Background loop that checks for new emails every POLL_INTERVAL seconds."""
    logger.info(f"Email polling started (every {POLL_INTERVAL}s)")

    # Seed seen_ids with current unread emails so we don't re-notify on startup
    try:
        initial = run_triage(only_new=False)
        logger.info(f"Seeded {len(initial['results'])} existing emails")
    except Exception as e:
        logger.error(f"Failed to seed emails: {e}")

    while True:
        await asyncio.sleep(POLL_INTERVAL)
        try:
            result = run_triage(only_new=True)
            if result['urgent'] > 0:
                logger.info(f"Notified {result['urgent']} urgent email(s)")
            elif result['processed'] > 0:
                logger.info(f"Checked {result['processed']} new email(s), none urgent")
        except Exception as e:
            logger.error(f"Poll error: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Deep Work Secretary started")
    task = asyncio.create_task(poll_loop())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "seen_count": len(seen_ids), "poll_interval": POLL_INTERVAL}


@app.post("/triage")
def triage():
    """Manual triage — processes all unread emails regardless of seen status."""
    return run_triage(only_new=False)
