import os

import ray
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


    def ask(self, content):
        print("================== ASK GEM ===============")
        response = self.client.models.generate_content(
            model=self.model,
            contents=content,
        )
        text = response.text
        obj_ref = ray.put(text)
        return obj_ref

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
        obj_ref = ray.put(text)
        return obj_ref
