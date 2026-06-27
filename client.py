"""Shared Gemini / Imagen client factory."""
import os

from google import genai


def genai_client(gem_api_key=None):
    """Shared Gemini/Imagen client for cover generation."""
    return genai.Client(api_key=os.getenv("GEM_API_KEY"))
