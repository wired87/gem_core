"""Debug helpers — list available Gemini models."""
from gem_core.client import genai_client


def list_available_models(gem_api_key):
    print("Fetching available Gemini models.")
    client = genai_client(gem_api_key)
    try:
        models = client.models.list()
        print("--- ALLE VERFÜGBAREN MODELLE ---")
        for m in models:
            print(f"Name: {m.name}")
            if m.supported_actions:
                print(f"  -> Aktionen: {', '.join(m.supported_actions)}")
            print("-" * 30)
    except Exception as e:
        print(f"Failed to fetch available models: {e}")
