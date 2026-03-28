from models.mlp import extract_metadata_features
from text_analyzer import text_analyze


def get_final_score(email: dict, mlp_model) -> float:
    metadata_tensor = extract_metadata_features(email)
    behavioral_score = mlp_model(metadata_tensor).item() * 100

    email_text = f"From: {email['from']}\n{email['snippet']}"
    semantic_score = text_analyze(email_text)

    return (behavioral_score * 0.4) + (semantic_score * 0.6)
