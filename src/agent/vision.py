"""Vision utilities for the Web3 agent."""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

_LOCAL_VISION_MODEL = None


def _encode_image(image_path: str | Path) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found at {path}")
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _local_clip_consistency(text_claim: str, image_path: str, model_path: str, threshold: float = 0.5) -> Optional[bool]:
    """Use a local vision-text embedding model (e.g., CLIP/SigLIP) to score similarity."""
    global _LOCAL_VISION_MODEL
    try:
        if _LOCAL_VISION_MODEL is None:
            from sentence_transformers import SentenceTransformer

            _LOCAL_VISION_MODEL = SentenceTransformer(model_path)
            logger.info("Loaded local vision model: %s", model_path)
        from PIL import Image
        import torch

        text_emb = _LOCAL_VISION_MODEL.encode(
            [text_claim],
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        image_emb = _LOCAL_VISION_MODEL.encode(
            [Image.open(image_path)],
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        # cosine similarity with normalized embeddings = dot product
        sim = float(torch.sum(text_emb * image_emb))
        logger.info("Local vision similarity=%.4f (threshold=%.4f)", sim, threshold)
        return sim >= threshold
    except Exception as exc:  # noqa: BLE001
        logger.exception("Local vision check failed: %s", exc)
        return None


def verify_image_consistency(text_claim: str, image_path: str, model: str | None = None) -> Optional[bool]:
    """
    Send text + image to a vision-capable model and return:
    - True: consistent
    - False: inconsistent
    - None: call/processing failed or not executed
    """
    # Try local lightweight consistency check first if enabled
    use_local = os.getenv("VISION_LOCAL_ENABLED", "true").lower() == "true"
    local_model = os.getenv("VISION_LOCAL_MODEL") or "sentence-transformers/clip-ViT-B-32"
    local_threshold = float(os.getenv("VISION_LOCAL_THRESHOLD", "0.5"))
    if use_local:
        local_result = _local_clip_consistency(text_claim, image_path, local_model, threshold=local_threshold)
        if local_result is not None:
            return bool(local_result)

    # Optional remote multimodal QA/description (OpenAI-compatible)
    use_remote = os.getenv("VISION_REMOTE_ENABLED", "false").lower() == "true"
    if not use_remote:
        return None

    api_key = os.getenv("VISION_API_KEY") or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("VISION_API_BASE") or os.getenv("LLM_API_BASE") or os.getenv("OPENAI_BASE_URL")
    model_name = model or os.getenv("VISION_MODEL") or "gpt-4o-mini"

    client = OpenAI(api_key=api_key, base_url=base_url)
    encoded_image = _encode_image(image_path)

    try:
        logger.info("vision check start model=%s image=%s", model_name, image_path)
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
    except Exception as exc:  # noqa: BLE001
        logger.exception("vision check failed: %s", exc)
        return None

    result_text = response.choices[0].message.content.strip().upper()
    is_consistent = result_text.startswith("CONSISTENT")
    logger.info("vision check result=%s", result_text)
    return is_consistent
