"""Vision utilities for the Web3 agent."""

from __future__ import annotations

import base64
import os
from pathlib import Path

from openai import OpenAI


def _encode_image(image_path: str | Path) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found at {path}")
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def verify_image_consistency(text_claim: str, image_path: str, model: str | None = None) -> bool:
    """
    Send text + image to a vision-capable model and return True if they are consistent.

    The function intentionally expects a strict "CONSISTENT"/"INCONSISTENT" response so the
    caller can treat False as a potential visual anomaly.
    """
    api_key = os.getenv("VISION_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("VISION_API_BASE") or os.getenv("LLM_API_BASE") or os.getenv("OPENAI_BASE_URL")
    model_name = model or os.getenv("VISION_MODEL") or "gpt-4o-mini"

    client = OpenAI(api_key=api_key, base_url=base_url)
    encoded_image = _encode_image(image_path)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a security checker looking for image-text mismatches. "
                        "Reply with exactly one word: CONSISTENT if the image matches the text, "
                        "otherwise INCONSISTENT."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Text claim: {text_claim}"},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{encoded_image}"},
                        },
                    ],
                },
            ],
            temperature=0,
        )
    except Exception:
        return False

    result_text = response.choices[0].message.content.strip().upper()
    return result_text.startswith("CONSISTENT")
