"""
Live training monitor. Run alongside train.py:
    python3 monitor.py
Refreshes every 2 seconds. Press Ctrl+C to exit.
"""
import json
import os
import time

METRICS_FILE = "train_metrics.json"
REFRESH_SECS = 2
BAR_WIDTH = 30


def bar(value, max_value=1.0, width=BAR_WIDTH):
    filled = int(value / max_value * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def render(metrics):
    os.system("clear")
    print("=" * 65)
    print("  Deep Work Secretary — Training Monitor")
    print("=" * 65)

    for model_key, label in [("mlp", "Behavioral MLP"), ("text", "Text Transformer")]:
        rows = metrics.get(model_key, [])
        if not rows:
            print(f"\n{label}: waiting for first epoch...\n")
            continue

        last = rows[-1]
        best = min(rows, key=lambda r: r["val_loss"])
        total_epochs = last["epoch"]

        print(f"\n── {label} ──")
        print(f"  Epoch      : {total_epochs}")
        print(f"  Train loss : {last['train_loss']:.4f}  {bar(last['train_loss'])}")
        print(f"  Val loss   : {last['val_loss']:.4f}  {bar(last['val_loss'])}")
        print(f"  Val acc    : {last['val_acc']:.2%}  {bar(last['val_acc'])}")
        print(f"  Best       : epoch {best['epoch']}  val_loss={best['val_loss']:.4f}  acc={best['val_acc']:.2%}")

        # Mini sparkline of last 10 val losses
        recent = rows[-10:]
        lo = min(r["val_loss"] for r in recent)
        hi = max(r["val_loss"] for r in recent) or 1.0
        levels = " ▁▂▃▄▅▆▇█"
        spark = "".join(
            levels[int((r["val_loss"] - lo) / max(hi - lo, 1e-9) * 8)]
            for r in recent
        )
        trend = "↓ improving" if len(recent) >= 2 and recent[-1]["val_loss"] < recent[0]["val_loss"] else "↑ diverging"
        print(f"  Val trend  : {spark}  {trend}")

    print("\n" + "=" * 65)
    print(f"  Refreshing every {REFRESH_SECS}s — Ctrl+C to exit")
    print("=" * 65)


def main():
    print(f"Watching {METRICS_FILE} ...")
    while True:
        try:
            if os.path.exists(METRICS_FILE):
                with open(METRICS_FILE) as f:
                    metrics = json.load(f)
                render(metrics)
            else:
                print(f"Waiting for {METRICS_FILE} to appear...")
            time.sleep(REFRESH_SECS)
        except KeyboardInterrupt:
            print("\nMonitor stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(REFRESH_SECS)


if __name__ == "__main__":
    main()
