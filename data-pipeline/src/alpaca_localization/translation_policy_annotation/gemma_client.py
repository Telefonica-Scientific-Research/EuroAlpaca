"""Gemma API and JSON parsing helpers."""

import argparse
import json
import re
from typing import Optional

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)

class GemmaJSONParseError(RuntimeError):
    """Raised when Gemma is reachable but its answer is not valid JSON."""

    def __init__(self, raw_response: str):
        """Handle init."""
        super().__init__(f"Gemma returned invalid JSON: {raw_response[:300]}")
        self.raw_response = raw_response

def extract_first_json_object(text: str) -> Optional[dict]:
    """Extract the first JSON object from model text."""
    # Model occasionally emits extra wrappers/text; recover by extracting
    # the first {...} block when direct parse fails.
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = _JSON_OBJ_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

def query_gemma_server(
    prompt: str,
    endpoint: str,
    timeout: int,
    max_tokens: int,
    temperature: float,
) -> str:
    """Query gemma server."""
    # Returns raw assistant text. Gemma is required for fresh annotation runs:
    # if the endpoint is unreachable or returns an invalid response, fail fast
    # instead of silently falling back to permissive defaults.
    try:
        import requests
    except Exception as e:
        raise RuntimeError(f"Gemma query unavailable (requests import failed): {e}") from e

    try:
        resp = requests.post(
            endpoint,
            headers={"Content-Type": "application/json"},
            json={
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are a strict data curation assistant. "
                            "Return JSON only."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False,
            },
            timeout=timeout,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Gemma server error {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            if "choices" in data and data["choices"] and "text" in data["choices"][0]:
                return str(data["choices"][0]["text"]).strip()
            raise RuntimeError(f"Gemma response missing assistant content: {str(data)[:300]}")
    except Exception as e:
        if isinstance(e, RuntimeError):
            raise
        raise RuntimeError(f"Gemma request failed: {type(e).__name__}: {e}") from e


def query_gemma_json(prompt: str, args: argparse.Namespace) -> dict:
    """Run one Gemma JSON call and return the parsed object.

    Both annotation passes use the same request settings and JSON recovery
    strategy. Keeping that here makes the pass functions focus on policy logic
    instead of transport/parsing boilerplate.
    """
    raw = query_gemma_server(
        prompt=prompt,
        endpoint=args.gemma_endpoint,
        timeout=args.gemma_timeout,
        max_tokens=args.gemma_max_tokens,
        temperature=args.gemma_temperature,
    )
    parsed = extract_first_json_object(raw)
    if parsed is None:
        raise GemmaJSONParseError(raw)
    return parsed

__all__ = [
    "GemmaJSONParseError",
    "extract_first_json_object",
    "query_gemma_server",
    "query_gemma_json",
]
