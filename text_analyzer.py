import torch
from config import TEXT_MODEL_PATH
from models.text_model import load_or_init_text_model, tokenize

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = load_or_init_text_model(TEXT_MODEL_PATH)
    return _model


def text_analyze(email_text: str) -> float:
    """
    Run the Transformer text model on an email string.
    Returns a urgency score in [0, 100].
    """
    model = _get_model()
    tokens = tokenize(email_text)
    with torch.no_grad():
        score = model(tokens).item()  # [0, 1]
    return score * 100
