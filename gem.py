import os
from google import genai

import dotenv
from google.genai import types

dotenv.load_dotenv()

class Gem:

    def __init__(self, model="gemini-2.0-flash"):
        self.model = model
        api_key = os.environ.get("GEMINI_API_KEY")
        
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY environment variable is not set. "
                "Please set it in your .env file or environment:\n"
                "  - Create a .env file in the project root with: GEMINI_API_KEY=your_api_key_here\n"
                "  - Or export it: export GEMINI_API_KEY=your_api_key_here\n"
                "  - Get your API key from: https://makersuite.google.com/app/apikey"
            )
        
        if not api_key.startswith(("AIza", "AI")) or len(api_key) < 20:
            print(f"[Gem] WARNING: GEMINI_API_KEY looks invalid (starts with: {api_key[:5] if api_key else 'None'}...). "
                  f"Expected format: AIza... (39 characters)")
        
        self.client = genai.Client(api_key=api_key)
        print(f"[Gem] Initialized with model={model}, API key present (length={len(api_key)})")
        self.max_try = 10

    def ask(self, content, config:dict=None):
        print("================== ASK GEM ===============")
        for i in range(self.max_try):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=content,
                    config=config if config else {},
                )
                text = response.text
                print(f"[Gem] ask: success, response_length={len(text)}")
                return text
            except Exception as e:
                error_msg = str(e)
                print(f"[Gem] ask: attempt {i+1}/{self.max_try} failed: {error_msg}")
                
                # Check for specific API key errors
                if "API key" in error_msg or "INVALID_ARGUMENT" in error_msg or "API_KEY_INVALID" in error_msg:
                    print("[Gem] ERROR: Invalid or missing GEMINI_API_KEY.")
                    print("[Gem] Please check:")
                    print("  1. Your .env file contains: GEMINI_API_KEY=your_actual_key")
                    print("  2. The API key is valid and not expired")
                    print("  3. The API key has proper permissions for Gemini API")
                    print("  4. Get a new key from: https://makersuite.google.com/app/apikey")
                    # Don't retry on API key errors
                    raise ValueError(f"Invalid API key: {error_msg}")
                
                if i == self.max_try - 1:
                    print(f"[Gem] ask: all {self.max_try} attempts failed")
                    raise


    def ask_mm(self, file_content_str: str, prompt: str, config: dict = None):
        """Multimodal: file (PDF/image) + prompt. config can include response_mime_type, response_json_schema for structured output."""
        print("================== ASK GEM MultiModal ===============")
        cfg = dict(config) if config else {}
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=[file_content_str, prompt],
                config=cfg,
            )
            text = response.text
            print(f"[Gem] ask_mm: success, response_length={len(text)}")
            return text
        except Exception as e:
            error_msg = str(e)
            print(f"[Gem] ask_mm: error: {error_msg}")
            if "API key" in error_msg or "INVALID_ARGUMENT" in error_msg:
                raise ValueError(f"Invalid API key in multimodal request: {error_msg}")
            raise

    def ask_rag(
            self,
            prompt: str,
            corpus_name: str,
            rag_file_ids: list = None,
            top_k: int = 5,
            response_schema: dict = None
    ):
        """
        Executes a RAG query using the GenAI SDK.
        Combines retrieval and generation into one schematic output.
        """
        print(f"================== ASK RAG (Corpus: {corpus_name}) ===============")

        # 1. Configure the RAG Tool using the GenAI SDK syntax
        # Note: 'rag_resources' expects a list of RagResource objects
        rag_resource = types.RagResource(
            rag_corpus=corpus_name,
            rag_file_ids=rag_file_ids or []
        )

        # Using the Google Search / Retrieval tool structure for GenAI SDK
        retrieval_tool = types.Tool(
            vertex_rag_store=types.VertexRagStore(
                rag_resources=[rag_resource],
                similarity_top_k=top_k,
            )
        )

        # 2. Define the Schema (using a default if none provided)
        if not response_schema:
            response_schema = {
                "type": "OBJECT",
                "properties": {
                    "answer": {"type": "STRING"},
                    "sources": {"type": "ARRAY", "items": {"type": "STRING"}},
                    "confidence": {"type": "NUMBER"}
                },
                "required": ["answer", "sources"]
            }

        # 3. Build the config
        config = types.GenerateContentConfig(
            tools=[retrieval_tool],
            response_mime_type="application/json",
            response_schema=response_schema,
        )

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config,
            )

            # The response.text will be a JSON string thanks to response_mime_type
            print(f"[Gem] ask_rag: success")
            return response.text

        except Exception as e:
            print(f"[Gem] ask_rag: error: {str(e)}")
            raise


class Gemma:
    """
    Standalone Gemma using Hugging Face transformers (runs locally, no API key).
    Same method names and parameters as Gem: ask, ask_mm, ask_rag.
    Uses a local Gemma model via transformers; works on CPU/GPU/MPS, any OS.
    """
    # model name -> Hugging Face model id
    _MODEL_MAP = {
        "gemma2_2b_it": "google/gemma-2-2b-it",
        "gemma2_2b": "google/gemma-2-2b",
        "gemma2_9b_it": "google/gemma-2-9b-it",
        "gemma2_9b": "google/gemma-2-9b",
        "gemma3_270m_it": "google/gemma-3-270m-it",
        "gemma3_1b_it": "google/gemma-3-1b-it",
    }
    DEFAULT_MODEL = "gemma2_2b_it"
    DEFAULT_MAX_NEW_TOKENS = 1024

    def __init__(self, model: str = None):
        self._model_name = (model or self.DEFAULT_MODEL).strip().lower()
        self._pipe = None
        self._hf_model_id = self._MODEL_MAP.get(self._model_name) or self._MODEL_MAP[self.DEFAULT_MODEL]
        print(f"[Gemma] Initialized (transformers), model={self._model_name} -> {self._hf_model_id} (lazy load on first ask)")

    def _ensure_pipeline(self):
        if self._pipe is not None:
            return
        try:
            import torch
            from transformers import pipeline
        except ImportError as e:
            raise ImportError(
                "Gemma class requires: pip install transformers torch\n"
                "Runs Gemma locally with no API key. Optional: accelerate for device_map='auto'."
            ) from e
        import torch
        from transformers import pipeline
        if torch.cuda.is_available():
            device = 0
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = -1  # CPU
        dtype = getattr(torch, "bfloat16", torch.float16)
        self._pipe = pipeline(
            "text-generation",
            model=self._hf_model_id,
            model_kwargs={"torch_dtype": dtype},
            device=device,
            trust_remote_code=True,
        )
        print(f"[Gemma] Loaded {self._hf_model_id} on device={device}")

    def ask(self, content, config: dict = None):
        """Same signature as Gem.ask: content (str), optional config. Returns generated text."""
        print("================== ASK GEMMA ===============")
        self._ensure_pipeline()
        prompt = content if isinstance(content, str) else str(content)
        cfg = config or {}
        max_new_tokens = cfg.get("max_new_tokens", self.DEFAULT_MAX_NEW_TOKENS)
        try:
            out = self._pipe(
                prompt,
                max_new_tokens=max_new_tokens,
                return_full_text=False,
                do_sample=cfg.get("do_sample", False),
            )
            text = (out[0].get("generated_text", "") or "") if out else ""
            if isinstance(text, list):
                text = text[0].get("content", "") if text else ""
            text = (text or "").strip()
            print(f"[Gemma] ask: success, response_length={len(text)}")
            return text
        except Exception as e:
            print(f"[Gemma] ask: error: {e}")
            raise

    def ask_mm(self, file_content_str: str, prompt: str, config: dict = None):
        """Same signature as Gem.ask_mm. Treats file_content_str as text context."""
        print("================== ASK GEMMA MultiModal ===============")
        combined = f"{file_content_str}\n\n{prompt}" if file_content_str else prompt
        return self.ask(combined, config)

    def ask_rag(
        self,
        prompt: str,
        corpus_name: str,
        rag_file_ids: list = None,
        top_k: int = 5,
        response_schema: dict = None,
    ):
        """Same signature as Gem.ask_rag. No RAG: runs ask(prompt) and returns JSON-shaped response."""
        print(f"================== ASK GEMMA RAG (no retrieval, corpus={corpus_name}) ===============")
        answer = self.ask(prompt)
        import json
        response_schema = response_schema or {}
        required = response_schema.get("required", ["answer", "sources"])
        out = {"answer": answer, "sources": [], "confidence": 1.0}
        if "sources" not in out and "sources" in required:
            out["sources"] = []
        if "confidence" not in out and "confidence" in required:
            out["confidence"] = 1.0
        print("[Gemma] ask_rag: success")
        return json.dumps(out)


class GoogleIntelligent:
    """
    Wrapper that includes Gem and Gemma; model param chooses backend.
    Same method names, args and functionality as Gem.
    Default model='gemma' uses Gemma; use 'gemini' or 'gem' for Gem.
    """
    def __init__(self, model: str = "gemma"):
        self._model = (model or "gemma").strip().lower()
        print(f"[GoogleIntelligent] model={self._model} (backend={'Gemma' if self._model == 'gemma' else 'Gem'})")

    def _backend(self):
        """Returns Gem or Gemma depending on model."""
        return Gemma() if self._model == "gemma" else Gem()

    def ask(self, content, config: dict = None):
        return self._backend().ask(content, config)

    def ask_mm(self, file_content_str: str, prompt: str, config: dict = None):
        return self._backend().ask_mm(file_content_str, prompt, config)

    def ask_rag(
        self,
        prompt: str,
        corpus_name: str,
        rag_file_ids: list = None,
        top_k: int = 5,
        response_schema: dict = None,
    ):
        return self._backend().ask_rag(
            prompt=prompt,
            corpus_name=corpus_name,
            rag_file_ids=rag_file_ids,
            top_k=top_k,
            response_schema=response_schema,
        )
