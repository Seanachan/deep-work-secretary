# deep-work-secretary

An AI-powered email triage system that protects focus time by intercepting unread emails, scoring their urgency with neural models, and only sending push notifications for emails that truly matter — while you're in a deep work block.

## How it works

1. **Automatic Gmail polling** — background loop checks for new unread emails every 30 seconds (configurable), tracking seen IDs to avoid duplicate notifications
2. **Urgency scoring** — combines two models:
   - **Behavioral MLP** (64→32→1): scores metadata features — sender domain, urgency keywords, snippet length, message frequency, recency
   - **Text Transformer** (d_model=32, 4 heads, 2 layers): scores the email text semantically using a fixed urgency vocabulary
   - Final score = `0.4 × MLP + 0.6 × Transformer`
3. **Focus block detection** — checks Google Calendar for active focus/lecture/coding events and raises the urgency threshold dynamically
4. **Notification** — urgent emails from real senders punch through via [ntfy.sh](https://ntfy.sh), bypassing iOS/Android Do Not Disturb (noreply senders are automatically filtered out)

## Project structure

```
main.py              # FastAPI app — auto-polling, POST /triage, GET /health
email_fetcher.py     # Gmail API integration + thread history enrichment
scorer.py            # Combined MLP + Transformer scoring
text_analyzer.py     # Transformer inference wrapper
llm_analyzer.py      # Shim (delegates to text_analyzer)
calendar_checker.py  # Google Calendar focus block detection
notifier.py          # ntfy.sh push notification sender
config.py            # Environment-based configuration
train.py             # Training pipeline (Enron dataset + synthetic data)
monitor.py           # Live training monitor
models/
  mlp.py             # EmailUrgencyMLP + feature extraction
  text_model.py      # EmailTextTransformer + tokenizer
```

## Setup

```bash
pip install -r requirements.txt
```

Place your Google OAuth credentials at `credentials.json` (needs Gmail + Calendar read scopes), then create a `.env`:

```env
NTFY_TOPIC=your-topic
URGENCY_THRESHOLD=0.7
FOCUS_URGENCY_THRESHOLD=0.8
POLL_INTERVAL=30
```

## Running

```bash
# Start the server — automatic polling begins immediately
python3 -m uvicorn main:app --reload --port 8000

# Manual triage (processes all unread, not just new)
curl -X POST http://localhost:8000/triage

# Health check (shows seen email count and poll interval)
curl http://localhost:8000/health
```

Once running, the server automatically polls Gmail every `POLL_INTERVAL` seconds and sends ntfy.sh notifications for urgent new emails. No manual triggering needed.

## Training

Models are pre-trained on the [Enron Email Dataset](https://www.cs.cmu.edu/~./enron/) using folder labels (`action_items`, `important_*`) as urgency ground truth.

```bash
# Train on Enron CSV (auto-detected by column names)
python3 train.py --data path/to/emails.csv --epochs 50 --lr 3e-4

# Resume from saved checkpoints
python3 train.py --data path/to/emails.csv --epochs 30 --resume

# Watch training live in another terminal
python3 monitor.py
```

Trained weights are saved to `mlp_model.pt` and `text_model.pt` (excluded from git).

## Results (Checkpoint 1)

| Model | Best Val Loss | Val Accuracy |
|-------|-------------|-------------|
| Behavioral MLP | 0.618 | 54% |
| Text Transformer | 0.549 | 74% |

The MLP accuracy is limited by the Enron dataset lacking real thread-history metadata. Accuracy will improve as the system collects real Gmail usage data.
