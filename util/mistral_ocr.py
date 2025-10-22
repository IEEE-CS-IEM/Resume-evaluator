import base64
import os
from typing import Optional


def _load_client():
    try:
        from mistralai import Mistral  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "mistralai is not installed. Add it to requirements or pip install mistralai."
        ) from exc
    return Mistral


def ocr_image(image_bytes: bytes) -> Optional[str]:
    """
    Attempts to OCR the provided image bytes using Mistral's OCR API.
    Returns Markdown text when successful, otherwise None.
    """
    api_key = os.getenv("MISTRAL_API_KEY") or os.getenv("mistral_api_key")
    if not api_key:
        return None

    model_name = os.getenv("MISTRAL_OCR_MODEL", "CX-9").strip()
    mistral_cls = _load_client()
    payload = {
        "type": "base64",
        "base64": base64.b64encode(image_bytes).decode("ascii"),
        "mime_type": "image/png",
    }

    try:
        with mistral_cls(api_key=api_key) as client:
            response = client.ocr.process(model=model_name, document=payload)
    except Exception:
        return None

    if not response:
        return None

    pages = response.get("pages", [])
    markdown_segments = []
    for page in pages or []:
        markdown = page.get("markdown")
        if isinstance(markdown, str) and markdown.strip():
            markdown_segments.append(markdown.strip())

    if not markdown_segments:
        return None

    return "\n\n".join(markdown_segments)
