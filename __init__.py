import os
from google import genai

import dotenv
from google.genai import types

dotenv.load_dotenv()

class Gem:

    def __init__(self, model="gemini-2.5-pro"):
        self.model = model
        self.token_count = 0
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
                    config=config
                )
                text = response.text
                print(f"[Gem] ask: success, response_length={text}")

                if response.usage_metadata and response.usage_metadata.total_token_count:
                    self.token_count +=response.usage_metadata.total_token_count
                    print("TOTAL TOKENS USED:", self.token_count)

                return text
            except Exception as e:
                error_msg = str(e)
                print(f"[Gem] ask: attempt {i+1}/{self.max_try} failed: {error_msg}")
                
                # Check for specific API key errors

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

