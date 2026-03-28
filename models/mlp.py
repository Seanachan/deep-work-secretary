import os
import torch
import torch.nn as nn


class EmailUrgencyMLP(nn.Module):
    def __init__(self, input_size=7):
        super(EmailUrgencyMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


def extract_metadata_features(email: dict) -> torch.Tensor:
    snippet = email.get('snippet', '')
    from_field = email.get('from', '')

    snippet_length = min(len(snippet) / 500.0, 1.0)

    from_lower = from_field.lower()
    sender_has_edu = 1.0 if ('@edu' in from_lower or '.edu' in from_lower) else 0.0
    sender_has_noreply = 1.0 if ('noreply' in from_lower or 'no-reply' in from_lower) else 0.0

    snippet_lower = snippet.lower()
    has_urgent_keyword = 1.0 if any(
        kw in snippet_lower for kw in ['urgent', 'deadline', 'asap', 'important']
    ) else 0.0

    has_question = 1.0 if '?' in snippet else 0.0

    message_frequency_raw = email.get('message_frequency', 0.0)
    message_frequency_norm = min(max(message_frequency_raw / 20.0, 0.0), 1.0)

    time_since_last_reply_raw = email.get('time_since_last_reply', 168.0)
    recency_norm = 1.0 - min(max(time_since_last_reply_raw / 168.0, 0.0), 1.0)

    features = [
        snippet_length,
        sender_has_edu,
        sender_has_noreply,
        has_urgent_keyword,
        has_question,
        message_frequency_norm,
        recency_norm,
    ]

    return torch.tensor([features], dtype=torch.float32)


def load_or_init_model(path: str) -> EmailUrgencyMLP:
    model = EmailUrgencyMLP(input_size=7)

    if os.path.exists(path):
        model.load_state_dict(torch.load(path, weights_only=True))
    else:
        torch.save(model.state_dict(), path)

    model.eval()
    return model
