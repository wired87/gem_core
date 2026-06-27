"""Imagen text-to-image generation with model fallbacks."""
import os
import time

from google.genai import types

from gem_core.client import genai_client
from gem_core.image.config import imagen_model_chain
from gem_core.image.errors import is_permanent_model_error, is_transient_generation_error


def generate_cover_image_imagen_once(image_prompt: str, model: str):
    """Single Imagen generate_images call for *model*."""
    client = genai_client(os.getenv("GEM_API_KEY"))
    return client.models.generate_images(
        model=model,
        prompt=image_prompt,
        config=types.GenerateImagesConfig(
            number_of_images=1,
            output_mime_type="image/jpeg",
            aspect_ratio="1:1",
        ),
    )


def generate_cover_image_imagen(image_prompt: str):
    """Imagen with ordered model fallbacks and light per-model retries on transient errors."""
    per_model_retries = max(1, int(os.getenv("GEM_IMAGE_RETRY_PER_MODEL") or "1"))
    retry_delays = (2, 5, 10)
    collected_errors: list = []

    for model in imagen_model_chain():
        for attempt in range(per_model_retries):
            try:
                print(f"Generating image (attempt {attempt + 1}/{per_model_retries}).")
                return generate_cover_image_imagen_once(image_prompt, model)
            except Exception as exc:
                collected_errors.append(f"{model}: {exc}")
                print(f"Image generation attempt failed: {exc}")
                if is_permanent_model_error(exc):
                    break
                if not is_transient_generation_error(exc):
                    raise
                if attempt + 1 < per_model_retries:
                    delay = retry_delays[min(attempt, len(retry_delays) - 1)]
                    print(f"Temporary error. Retrying in {delay}s.")
                    time.sleep(delay)
        print("Primary image model exhausted. Trying a fallback.")

    raise RuntimeError("All Imagen models failed: " + " | ".join(collected_errors[-3:]))
