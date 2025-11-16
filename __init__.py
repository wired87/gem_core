import os

from google import genai
import dotenv as dotenv
dotenv.load_dotenv()

GAIC=genai.Client(api_key=os.environ["GEMINI_API_KEY"])