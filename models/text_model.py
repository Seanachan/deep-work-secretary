import os
import re
import torch
import torch.nn as nn

MAX_LEN = 64
D_MODEL = 32
NHEAD = 4
NUM_LAYERS = 2

# Fixed vocabulary of common email-relevant words.
# Index 0 = <PAD>, index 1 = <UNK>.
_VOCAB_WORDS = [
    "<PAD>", "<UNK>",
    # urgency signals
    "urgent", "urgently", "asap", "immediately", "deadline", "due", "overdue",
    "critical", "important", "priority", "high", "emergency", "alert",
    "action", "required", "needed", "respond", "response", "reply",
    "please", "kindly", "attention", "note", "reminder", "follow", "up",
    # time signals
    "today", "tomorrow", "tonight", "now", "soon", "morning", "afternoon",
    "evening", "week", "month", "year", "date", "time", "schedule",
    "meeting", "appointment", "conference", "call", "zoom", "teams",
    # academic / professional context
    "professor", "instructor", "course", "class", "lecture", "lab",
    "assignment", "homework", "project", "exam", "test", "quiz", "grade",
    "submit", "submission", "report", "paper", "thesis", "research",
    "office", "hours", "department", "faculty", "student", "advisor",
    # email structure words
    "from", "to", "subject", "re", "fwd", "dear", "hi", "hello",
    "regards", "sincerely", "thanks", "thank", "you", "best",
    # action verbs
    "need", "want", "must", "should", "would", "could", "help",
    "send", "receive", "confirm", "check", "review", "approve",
    "update", "complete", "finish", "start", "begin", "sign",
    "read", "write", "prepare", "attend", "join", "reschedule",
    # negations / qualifiers
    "not", "no", "never", "late", "early", "before", "after",
    "please", "if", "when", "by", "until", "within", "only",
    # numbers / dates (keep raw tokens)
    "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
    "monday", "tuesday", "wednesday", "thursday", "friday",
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    # misc common words
    "team", "group", "work", "office", "email", "message", "issue",
    "problem", "question", "request", "approval", "feedback",
]

VOCAB: dict[str, int] = {w: i for i, w in enumerate(_VOCAB_WORDS)}
VOCAB_SIZE = len(_VOCAB_WORDS)


def tokenize(text: str, max_len: int = MAX_LEN) -> torch.Tensor:
    """Lowercase, split on non-alphanumeric, map to vocab indices, pad/truncate."""
    tokens = re.findall(r"[a-z0-9]+", text.lower())[:max_len]
    indices = [VOCAB.get(t, VOCAB["<UNK>"]) for t in tokens]
    indices += [0] * (max_len - len(indices))  # pad
    return torch.tensor([indices], dtype=torch.long)  # [1, max_len]


class EmailTextTransformer(nn.Module):
    """
    Small Transformer encoder for email text urgency scoring.

    Architecture:
        Token Embedding (vocab_size → d_model)
        + Positional Embedding (max_len → d_model)
        → TransformerEncoder (d_model=32, nhead=4, 2 layers, ffn_dim=64)
        → Masked mean pooling
        → Linear(d_model, 16) → ReLU → Linear(16, 1) → Sigmoid
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        num_layers: int = NUM_LAYERS,
        max_len: int = MAX_LEN,
    ):
        super().__init__()
        self.max_len = max_len
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=64,
            dropout=0.2,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: LongTensor of shape [batch, seq_len] with token indices (0 = pad)
        Returns:
            FloatTensor of shape [batch, 1] with urgency probability in [0, 1]
        """
        batch, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, seq_len]
        emb = self.token_emb(x) + self.pos_emb(positions)               # [B, S, D]

        padding_mask = (x == 0)                                          # [B, S] True = pad
        out = self.transformer(emb, src_key_padding_mask=padding_mask)   # [B, S, D]

        # Masked mean pool: average only over non-padding positions
        mask = (~padding_mask).float().unsqueeze(-1)                     # [B, S, 1]
        pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1) # [B, D]

        return self.classifier(pooled)                                   # [B, 1]


def load_or_init_text_model(path: str) -> EmailTextTransformer:
    model = EmailTextTransformer()
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, weights_only=True))
    else:
        torch.save(model.state_dict(), path)
    model.eval()
    return model
