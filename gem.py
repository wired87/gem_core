import os
from google import genai

import dotenv
dotenv.load_dotenv()

class Gem:

    def __init__(self, model="gemini-2.5-flash"):
        self.model = model
        self.client = genai.Client(
            api_key=os.environ.get("GEMINI_API_KEY")
        )
        print("GEMW INITIALIZED")
        self.max_try =10

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
                return text
            except Exception as e:
                print("ERR REQUEST", e)


    def ask_mm(self, file_content_str: str, prompt: str, config: dict = None):
        """Multimodal: file (PDF/image) + prompt. config can include response_mime_type, response_json_schema for structured output."""
        print("================== ASK GEM MultiModal ===============")
        cfg = dict(config) if config else {}
        response = self.client.models.generate_content(
            model=self.model,
            contents=[file_content_str, prompt],
            config=cfg,
        )
        text = response.text
        return text
