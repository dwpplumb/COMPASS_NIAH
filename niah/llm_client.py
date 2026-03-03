from __future__ import annotations

import json
import time
from typing import Any

from niah.config import AppConfig
from niah.models import LLMResponse


def call_chat_completion(
    *,
    cfg: AppConfig,
    system_text: str,
    user_text: str,
    temperature: float | None = None,
) -> LLMResponse:
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
        "temperature": float(cfg.llm_temperature if temperature is None else temperature),
        "max_tokens": int(cfg.llm_max_output_tokens),
    }
    headers = {
        "Authorization": f"Bearer {cfg.llm_api_key}",
        "Content-Type": "application/json",
    }
    max_retries = max(0, int(cfg.llm_max_retries))
    backoff_s = max(0.1, float(cfg.llm_retry_backoff_s))
    retry_statuses = {429, 502, 503, 504}

    last_error: str = ""
    resp = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(
                cfg.llm_endpoint_url,
                headers=headers,
                data=json.dumps(body),
                timeout=float(cfg.llm_timeout_s),
            )
        except requests.RequestException as e:
            last_error = f"LLM request error: {e}"
            if attempt < max_retries:
                time.sleep(backoff_s * (2 ** attempt))
                continue
            raise RuntimeError(last_error) from e

        if resp.status_code < 400:
            break

        last_error = f"LLM HTTP {resp.status_code}: {resp.text[:500]}"
        if resp.status_code in retry_statuses and attempt < max_retries:
            time.sleep(backoff_s * (2 ** attempt))
            continue
        raise RuntimeError(last_error)

    if resp is None:
        raise RuntimeError(last_error or "LLM call failed without response")

    obj = resp.json()
    try:
        text = str(obj["choices"][0]["message"]["content"])
    except Exception as e:
        raise RuntimeError(f"Unexpected chat response schema: {e}; body={resp.text[:500]}") from e
    return LLMResponse(text=text, raw=obj)
