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


    def ask_mm(self, file_content_str:str, prompt:str):
        print("================== ASK GEM MultiModal ===============")
        print("file_content_str", file_content_str)
        print("prompt", prompt)
        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                file_content_str,
                prompt,
            ],
        )
        text = response.text
        return text
