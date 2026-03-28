"""
Kept for backwards compatibility — delegates to the neural text model.
Use text_analyzer.text_analyze() directly for new code.
"""
from text_analyzer import text_analyze


def llm_analyze(email_text: str) -> float:
    return text_analyze(email_text)
