"""Provider error classification and user-facing copy."""
from typing import Optional


def generation_error_code(exc: Exception) -> Optional[int]:
    """Best-effort HTTP/gRPC status code from google.genai errors."""
    for attr in ("status_code", "code"):
        raw = getattr(exc, attr, None)
        if raw is None:
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    text = str(exc).upper()
    for code in (503, 429, 404, 400, 401, 403):
        if str(code) in text:
            return code
    return None


def is_transient_generation_error(exc: Exception) -> bool:
    """True for overload / capacity errors where another model or retry may succeed."""
    code = generation_error_code(exc)
    if code in (429, 503, 504):
        return True
    text = str(exc).upper()
    return any(
        token in text
        for token in (
            "UNAVAILABLE",
            "RESOURCE_EXHAUSTED",
            "OVERLOADED",
            "DEADLINE",
            "TIMEOUT",
            "TRY AGAIN",
            "CURRENTLY UNAVAILABLE",
        )
    )


def is_permanent_model_error(exc: Exception) -> bool:
    """404/400 on a model id — skip retries and try the next model in the chain."""
    code = generation_error_code(exc)
    if code in (404, 400):
        return True
    text = str(exc).upper()
    return "NOT_FOUND" in text or "IS NOT FOUND" in text


def user_facing_generation_error(exc: Exception) -> str:
    """Map provider failures to actionable UI copy (no raw stack traces)."""
    if is_transient_generation_error(exc):
        return (
            "Google image generation is temporarily overloaded. "
            "The server tried multiple models — please wait a moment and try again."
        )
    text = str(exc).strip()
    if "API key" in text or generation_error_code(exc) == 401:
        return "Gemini API key is missing or invalid. Check GEM_API_KEY in your server configuration."
    if len(text) > 240:
        text = text[:237] + "..."
    return text or "Image generation failed. Please try again."
