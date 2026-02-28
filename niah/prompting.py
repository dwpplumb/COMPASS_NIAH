from __future__ import annotations

from niah.config import AppConfig


def load_compass_text(cfg: AppConfig) -> str:
    if not cfg.compass_prompt_file:
        return ""
    try:
        with open(cfg.compass_prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def build_system_prompt(*, cfg: AppConfig, compass_text: str) -> str:
    base = str(cfg.system_prompt or "").strip()
    if compass_text:
        return (f"{base}\n\nCOMPASS (internal):\n{compass_text}").strip()
    return base


def build_full_context_user_prompt(*, context_text: str, question: str) -> str:
    return (
        "Context (full, uncompressed):\n"
        f"{context_text}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Task:\n"
        "- Find the sentence in the context that answers the question.\n"
        "- Return the full original sentence exactly as written.\n"
        "- If no supporting sentence exists, answer: UNSURE"
    )


def build_rag_user_prompt(*, retrieved_context: str, question: str) -> str:
    return (
        "Retrieved context:\n"
        f"{retrieved_context}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Task:\n"
        "- Find the sentence in the retrieved context that answers the question.\n"
        "- Return the full original sentence exactly as written.\n"
        "- If no supporting sentence exists, answer: UNSURE"
    )
