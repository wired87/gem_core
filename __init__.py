import os

from google import genai
import dotenv

dotenv.load_dotenv()


GAIC=genai.Client(api_key=os.getenv("GEMINI_API_KEY"))