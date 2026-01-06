"""Vision utilities for the Web3 agent."""

from __future__ import annotations

import base64
import logging
import os
from pathlib import Path
from typing import Optional

from openai import OpenAI

logger = logging.getLogger(__name__)

_FLORENCE_MODEL = None
_FLORENCE_PROCESSOR = None
_FLORENCE_NAME = None


def _encode_image(image_path: str | Path) -> str:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found at {path}")
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def _llm_text_consistency(text_claim: str, caption: str) -> Optional[bool]:
    """Ask a remote LLM to judge consistency between user text and generated caption."""
    api_key = os.getenv("VISION_REMOTE_TEXT_API_KEY")
    base_url = os.getenv("VISION_REMOTE_TEXT_API_BASE")
    model_name = os.getenv("VISION_REMOTE_TEXT_MODEL") or "gpt-4o-mini"
    if not api_key or not base_url:
        logger.warning("LLM text consistency skipped: missing VISION_REMOTE_TEXT_API_KEY or VISION_REMOTE_TEXT_API_BASE")
        return None
    logger.info("LLM text consistency request model=%s claim=%.80s caption=%.80s", model_name, text_claim, caption)
    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a security checker. Judge whether the following two statements describe the same image. "
                        "Reply with exactly one word: CONSISTENT or INCONSISTENT."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Text claim: {text_claim}\nImage caption: {caption}",
                },
            ],
            temperature=0,
        )
        result = resp.choices[0].message.content.strip().upper()
        logger.info("LLM text consistency result=%s", result)
        if result.startswith("CONSISTENT"):
            return True
        if result.startswith("INCONSISTENT"):
            return False
        return None
    except Exception as exc:  # noqa: BLE001
        logger.exception("LLM text consistency check failed: %s", exc)
        return None


def _caption_consistency(
    text_claim: str, image_path: str, model_path: str, threshold: float | None = None
) -> tuple[Optional[bool], Optional[float], Optional[str]]:
    """
    Use a caption/VLM model (Florence-2) to describe the image, then ask a remote LLM to judge consistency.
    Returns (bool, score=None, caption) or (None, None, caption/None) on failure.
    """
    global _FLORENCE_MODEL, _FLORENCE_PROCESSOR, _FLORENCE_NAME

    try:
        from PIL import Image
        import torch
        import transformers  # type: ignore
        from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore

        if _FLORENCE_MODEL is None or _FLORENCE_NAME != model_path:
            # Some Florence checkpoints expect sdpa flags; patch to disable if missing
            if not hasattr(transformers.modeling_utils.PreTrainedModel, "_supports_sdpa"):  # type: ignore
                transformers.modeling_utils.PreTrainedModel._supports_sdpa = False  # type: ignore[attr-defined]
            _FLORENCE_MODEL = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                attn_implementation="eager",
                low_cpu_mem_usage=False,
                dtype="float32",
            ).to("cpu").eval()
            _FLORENCE_PROCESSOR = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            _FLORENCE_NAME = model_path
            logger.info("Loaded caption model: %s", model_path)

        assert _FLORENCE_PROCESSOR is not None
        image = Image.open(image_path).convert("RGB")
        prompt = "<MORE_DETAILED_CAPTION>"
        inputs = _FLORENCE_PROCESSOR(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to("cpu") for k, v in inputs.items()}
        with torch.no_grad():
            generated_ids = _FLORENCE_MODEL.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=128,
                do_sample=False,
                num_beams=1,
                use_cache=False,
            )
        decoded = _FLORENCE_PROCESSOR.batch_decode(generated_ids, skip_special_tokens=True)[0]
        logger.info("Caption text=%s", decoded)

        llm_result = _llm_text_consistency(text_claim, decoded)
        if llm_result is not None:
            return llm_result, None, decoded
        return None, None, decoded
    except Exception as exc:  # noqa: BLE001
        logger.exception("Caption-based vision check failed: %s", exc)
        return None, None, None


def verify_image_consistency(
    text_claim: str, image_path: str, model: str | None = None
) -> tuple[Optional[bool], Optional[float], Optional[str]]:
    """
    Send text + image to a vision-capable model and return:
    - (True, score, caption): consistent
    - (False, score, caption): inconsistent
    - (None, None, caption/None): call/processing failed or not executed
    """
    logger.info("vision input text=%s image=%s", text_claim, image_path)
    mode = (os.getenv("VISION_PIPELINE_MODE") or "caption_text").lower()

    if mode == "caption_text":
        caption_model = os.getenv("VISION_LOCAL_CAPTION_MODEL") or "microsoft/Florence-2-base"
        caption_result, caption_score, caption_text = _caption_consistency(text_claim, image_path, caption_model)
        if caption_result is not None:
            return bool(caption_result), caption_score, caption_text
        return None, None, caption_text

    if mode == "multimodal":
        return _remote_multimodal_consistency(text_claim, image_path)

    logger.warning("Unknown VISION_PIPELINE_MODE=%s", mode)
    return None, None, None


def _remote_multimodal_consistency(text_claim: str, image_path: str) -> tuple[Optional[bool], Optional[float], Optional[str]]:
    """Use remote multimodal model (glm-4v-flash) to judge consistency."""
    api_key = os.getenv("VISION_REMOTE_MM_API_KEY")
    model_name = os.getenv("VISION_REMOTE_MM_MODEL") or "glm-4v-flash"
    if not api_key:
        logger.warning("Vision remote multimodal skipped: missing VISION_REMOTE_MM_API_KEY")
        return None, None, None
    try:
        from zhipuai import ZhipuAI  # type: ignore
        import json
    except Exception as exc:  # noqa: BLE001
        logger.exception("Vision remote multimodal import failed: %s", exc)
        return None, None, None

    client = ZhipuAI(api_key=api_key)
    encoded_image = _encode_image(image_path)
    prompt = (
        "你是一名严格的 Web3 安全审计员。任务：检测图文不一致的欺诈行为。\n"
        "步骤：1) 独立分析图表趋势/关键数值；2) 将图表事实与文本声明比对；\n"
        "若图表趋势与声明相反或严重不符，判定为不一致（INCONSISTENT）。\n"
        "请仅以 JSON 格式返回结果，不要包含 Markdown：\n"
        '{\"is_consistent\": bool, \"chart_trend\": \"string\", \"reason\": \"string\"}'
    )
    logger.info("Vision remote multimodal request model=%s claim=%.80s image=%s", model_name, text_claim, image_path)

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"{prompt}\n\n文本声明: \"{text_claim}\""},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}},
                    ],
                }
            ],
            temperature=0,
        )
        raw_content = resp.choices[0].message.content
        logger.info("Vision remote multimodal raw content=%.200s", raw_content)
        clean = raw_content.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean)
        is_consistent = bool(data.get("is_consistent"))
        chart_trend = data.get("chart_trend")
        reason = data.get("reason")
        logger.info("Vision remote multimodal result=%s trend=%s reason=%s", is_consistent, chart_trend, reason)
        return is_consistent, None, chart_trend
    except Exception as exc:  # noqa: BLE001
        logger.exception("Vision remote multimodal check failed: %s", exc)
        return None, None, None
