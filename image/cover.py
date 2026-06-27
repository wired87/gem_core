"""Top-level cover image generation orchestrator."""
from gem_core.image.gemini import (
    generate_cover_image_gemini_text,
    generate_cover_image_with_references,
)
from gem_core.image.imagen import generate_cover_image_imagen


def generate_cover_image(image_prompt: str, images=None):
    """
    Generate cover art. Uses Gemini reference-image model when uploads are present,
    otherwise Imagen text-to-image with multi-model + Gemini text fallback.
    """
    print("Generating the image.")
    try:
        if images:
            try:
                return generate_cover_image_with_references(image_prompt, images)
            except Exception as ref_error:
                print(f"Reference-aware generation failed. Falling back: {ref_error}")
        try:
            return generate_cover_image_imagen(image_prompt)
        except Exception as imagen_error:
            print(f"Imagen chain failed. Trying fallback: {imagen_error}")
            return generate_cover_image_gemini_text(image_prompt)
    except Exception as e:
        print(f"Image generation failed: {e}")
        raise
