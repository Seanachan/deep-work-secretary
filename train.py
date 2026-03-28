import argparse
import json
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iter(iterable)

from models.mlp import EmailUrgencyMLP
from models.text_model import EmailTextTransformer
from data.synthetic import generate_synthetic_emails
from data.dataset import EmailDataset


def train_model(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        if isinstance(model, EmailUrgencyMLP):
            metadata, _, label = batch
            metadata, label = metadata.to(device), label.to(device)
            output = model(metadata)
        else:
            _, text, label = batch
            text, label = text.to(device), label.to(device)
            output = model(text)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            if isinstance(model, EmailUrgencyMLP):
                metadata, _, label = batch
                metadata, label = metadata.to(device), label.to(device)
                output = model(metadata)
            else:
                _, text, label = batch
                text, label = text.to(device), label.to(device)
                output = model(text)
            loss = criterion(output, label)
            total_loss += loss.item()
            preds = (output >= 0.5).float()
            correct += (preds == label).sum().item()
            total += label.size(0)
    avg_loss = total_loss / len(loader)
    accuracy = correct / total if total > 0 else 0.0
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description='Train email urgency models')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--mlp-out', type=str, default='mlp_model.pt')
    parser.add_argument('--text-out', type=str, default='text_model.pt')
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--resume', action='store_true', help='Resume from saved checkpoints')
    parser.add_argument('--metrics-out', type=str, default='train_metrics.json')
    args = parser.parse_args()

    if args.data is not None:
        import csv
        csv.field_size_limit(10_000_000)
        with open(args.data, newline='', encoding='utf-8', errors='replace') as f:
            header = next(csv.reader(f))
        if 'file' in header and 'message' in header:
            # Enron-style CSV
            from data.preprocess import load_enron_csv
            emails = load_enron_csv(args.data)
        else:
            # Standard pre-processed CSV
            emails = []
            with open(args.data, newline='', encoding='utf-8', errors='replace') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    emails.append({
                        'snippet': row['snippet'],
                        'from': row['from'],
                        'message_frequency': float(row['message_frequency']),
                        'time_since_last_reply': float(row['time_since_last_reply']),
                        'label': float(row['label']),
                    })
    else:
        emails = generate_synthetic_emails(500)

    train_emails, val_emails = train_test_split(emails, test_size=0.2, random_state=42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Oversample minority class via WeightedRandomSampler
    n_pos = sum(1 for e in train_emails if e['label'] == 1.0)
    n_neg = len(train_emails) - n_pos
    sample_weights = [
        (1.0 / n_pos) if e['label'] == 1.0 else (1.0 / n_neg)
        for e in train_emails
    ]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_emails), replacement=True)
    criterion = nn.BCELoss()

    train_dataset = EmailDataset(train_emails)
    val_dataset = EmailDataset(val_emails)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Train MLP
    print(f"Class ratio — pos: {n_pos}, neg: {n_neg}")

    # Initialise metrics file
    metrics = {"mlp": [], "text": []}
    if args.resume and os.path.exists(args.metrics_out):
        with open(args.metrics_out) as f:
            metrics = json.load(f)

    def save_metrics():
        with open(args.metrics_out, "w") as f:
            json.dump(metrics, f, indent=2)

    print("=== Training MLP ===")
    mlp_model = EmailUrgencyMLP(input_size=7).to(device)
    if args.resume and os.path.exists(args.mlp_out):
        mlp_model.load_state_dict(torch.load(args.mlp_out, weights_only=True))
        print(f"Resumed MLP from {args.mlp_out}")
    mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_mlp_val_loss = min((e["val_loss"] for e in metrics["mlp"]), default=float('inf'))

    for epoch in range(1, args.epochs + 1):
        train_loss = train_model(mlp_model, train_loader, mlp_optimizer, criterion, device)
        val_loss, val_acc = evaluate(mlp_model, val_loader, criterion, device)
        improved = val_loss < best_mlp_val_loss
        if improved:
            best_mlp_val_loss = val_loss
            torch.save(mlp_model.state_dict(), args.mlp_out)
        metrics["mlp"].append({
            "epoch": len(metrics["mlp"]) + 1,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "best": improved,
        })
        save_metrics()
        print(f"MLP  [{epoch:>3}/{args.epochs}]  train={train_loss:.4f}  val={val_loss:.4f}  acc={val_acc:.2%}{'  ✓' if improved else ''}", flush=True)

    # Train Text model
    print("=== Training Text Transformer ===")
    text_model = EmailTextTransformer().to(device)
    if args.resume and os.path.exists(args.text_out):
        text_model.load_state_dict(torch.load(args.text_out, weights_only=True))
        print(f"Resumed Text model from {args.text_out}")
    text_optimizer = optim.Adam(text_model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_text_val_loss = min((e["val_loss"] for e in metrics["text"]), default=float('inf'))

    for epoch in range(1, args.epochs + 1):
        train_loss = train_model(text_model, train_loader, text_optimizer, criterion, device)
        val_loss, val_acc = evaluate(text_model, val_loader, criterion, device)
        improved = val_loss < best_text_val_loss
        if improved:
            best_text_val_loss = val_loss
            torch.save(text_model.state_dict(), args.text_out)
        metrics["text"].append({
            "epoch": len(metrics["text"]) + 1,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "best": improved,
        })
        save_metrics()
        print(f"Text [{epoch:>3}/{args.epochs}]  train={train_loss:.4f}  val={val_loss:.4f}  acc={val_acc:.2%}{'  ✓' if improved else ''}", flush=True)

    print("Training complete.")


if __name__ == '__main__':
    main()
