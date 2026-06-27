"""Gemini native image output (text-only and reference-aware)."""
import os
from types import SimpleNamespace


from file.image.bytes_from_image import extract_image_bytes_from_content_response
from gem_core.client import genai_client
from gem_core.image.config import gemini_image_generate_config, gemini_text_fallback_model
import dotenv
dotenv.load_dotenv()

def wrap_generated_image_bytes(image_bytes_list):
    """Adapt Gemini bytes into the same shape Imagen returns (generated_images[].image.image_bytes)."""
    wrapped = []
    for raw in image_bytes_list or []:
        if not raw:
            continue
        wrapped.append(SimpleNamespace(image=SimpleNamespace(image_bytes=raw)))
    return SimpleNamespace(generated_images=wrapped)


def generate_cover_image_gemini_text(image_prompt: str):
    """Text-only fallback via Gemini native image output when Imagen tiers are down."""
    model = gemini_text_fallback_model()
    client = genai_client(os.getenv("GEM_API_KEY"))
    print("Trying Gemini text-only fallback.")
    response = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=image_prompt)])],
        config=gemini_image_generate_config(),
    )
    image_bytes_list = extract_image_bytes_from_content_response(response)
    if not image_bytes_list:
        raise RuntimeError(f"Gemini fallback model={model} returned no image bytes")
    return wrap_generated_image_bytes(image_bytes_list)


from google.genai import types
import mimetypes

def generate_cover_image_with_references(image_prompt: str, images):
    client = genai_client(os.getenv("GEM_API_KEY"))
    model = (os.getenv("GEM_IMAGE_MODEL_WITH_REFS") or "gemini-2.5-flash-image").strip()

    print("Generating with reference images...")

    content_parts = []

    for image_path in images:
        mime_type = mimetypes.guess_type(image_path)[0] or "image/png"

        with open(image_path, "rb") as f:
            content_parts.append(
                types.Part.from_bytes(
                    data=f.read(),
                    mime_type=mime_type,
                )
            )

    content_parts.append(
        types.Part.from_text(text=image_prompt)
    )

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Content(
                role="user",
                parts=content_parts,
            )
        ],
        config=gemini_image_generate_config(),
    )
    image_bytes_list = extract_image_bytes_from_content_response(response)
    if not image_bytes_list:
        raise RuntimeError("Reference-aware generation returned no image bytes")
    return wrap_generated_image_bytes(image_bytes_list)

