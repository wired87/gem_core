"""Model chain and image-generation config helpers."""
import os

from google.genai import types


def imagen_model_chain() -> list:
    """Primary Imagen model plus ordered fallbacks (env GEM_IMAGE_MODEL_FALLBACKS)."""
    primary = (os.getenv("GEM_IMAGE_MODEL") or "imagen-4.0-ultra-generate-001").strip()
    raw_fallbacks = (os.getenv("GEM_IMAGE_MODEL_FALLBACKS") or "").strip()
    if raw_fallbacks:
        fallbacks = [m.strip() for m in raw_fallbacks.split(",") if m.strip()]
    else:
        fallbacks = ["imagen-4.0-fast-generate-001", "imagen-4.0-generate-001"]
    chain: list = []
    for model_id in [primary, *fallbacks]:
        if model_id and model_id not in chain:
            chain.append(model_id)
    return chain


def gemini_text_fallback_model() -> str:
    """Last-resort text-to-image model when every Imagen tier fails."""
    return (os.getenv("GEM_IMAGE_TEXT_FALLBACK_MODEL") or "gemini-2.5-flash-image").strip()


def gemini_image_generate_config():
    """Developer-API-safe image config (output_mime_type breaks on non-Enterprise keys)."""
    return types.GenerateContentConfig(response_modalities=["IMAGE"])
