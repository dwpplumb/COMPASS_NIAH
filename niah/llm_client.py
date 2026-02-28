from __future__ import annotations

import json
from typing import Any

from niah.config import AppConfig
from niah.models import LLMResponse


def call_chat_completion(*, cfg: AppConfig, system_text: str, user_text: str) -> LLMResponse:
    import requests

    if not cfg.llm_api_key:
        raise RuntimeError("Missing NIAH_LLM_API_KEY")
    if not cfg.llm_model:
        raise RuntimeError("Missing NIAH_LLM_MODEL")
    if not cfg.llm_endpoint_url:
        raise RuntimeError("Missing NIAH_LLM_ENDPOINT_URL")

    body: dict[str, Any] = {
        "model": cfg.llm_model,
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
        ],
    }
    headers = {
        "Authorization": f"Bearer {cfg.llm_api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(
        cfg.llm_endpoint_url,
        headers=headers,
        data=json.dumps(body),
        timeout=float(cfg.llm_timeout_s),
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"LLM HTTP {resp.status_code}: {resp.text[:500]}")
    obj = resp.json()
    try:
        text = str(obj["choices"][0]["message"]["content"])
    except Exception as e:
        raise RuntimeError(f"Unexpected chat response schema: {e}; body={resp.text[:500]}") from e
    return LLMResponse(text=text, raw=obj)
