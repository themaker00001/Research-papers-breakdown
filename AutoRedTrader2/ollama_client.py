"""
Thin wrapper around the Ollama chat API, tuned for qwen3:14b.

Handles:
  - disabling / stripping qwen3's <think> reasoning stream,
  - robust JSON extraction from model output,
  - retries on transient failures.
"""

import json
import re
import time
from typing import Optional, Dict, Any

import requests

from config import CONFIG


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


class OllamaClient:
    def __init__(self, cfg=CONFIG):
        self.cfg = cfg
        self.url = f"{cfg.ollama_url}/api/chat"

    def chat(self, system: str, user: str,
             temperature: Optional[float] = None) -> str:
        """Send a single-turn chat and return the raw text content."""
        payload: Dict[str, Any] = {
            "model": self.cfg.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature
                if temperature is None else temperature
            },
        }
        # qwen3 supports a top-level "think" flag in recent Ollama versions.
        if self.cfg.disable_thinking:
            payload["think"] = False

        last_err = None
        for attempt in range(3):
            try:
                r = requests.post(
                    self.url, json=payload, timeout=self.cfg.request_timeout
                )
                r.raise_for_status()
                data = r.json()
                content = data.get("message", {}).get("content", "")
                return _THINK_RE.sub("", content).strip()
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"Ollama chat failed after retries: {last_err}")

    def chat_json(self, system: str, user: str,
                  temperature: Optional[float] = None) -> Optional[dict]:
        """Chat and parse a JSON object out of the response, or None."""
        raw = self.chat(system, user, temperature=temperature)
        return _extract_json(raw)


def _extract_json(text: str) -> Optional[dict]:
    """Best-effort extraction of the first JSON object from model text."""
    if not text:
        return None
    # strip markdown fences
    cleaned = text.replace("```json", "").replace("```", "").strip()
    # try direct parse first
    try:
        return json.loads(cleaned)
    except Exception:  # noqa: BLE001
        pass
    # fall back to the first {...} block
    m = _JSON_RE.search(cleaned)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        return None


def check_ollama(cfg=CONFIG) -> bool:
    """Verify the Ollama server is up and the model is present."""
    try:
        r = requests.get(f"{cfg.ollama_url}/api/tags", timeout=10)
        r.raise_for_status()
        names = [m.get("name", "") for m in r.json().get("models", [])]
        ok = any(cfg.model in n for n in names)
        if not ok:
            print(f"[warn] model '{cfg.model}' not found in Ollama. "
                  f"Available: {names}")
        return ok
    except Exception as e:  # noqa: BLE001
        print(f"[error] cannot reach Ollama at {cfg.ollama_url}: {e}")
        return False
