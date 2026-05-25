from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

response = client.models.generate_content(
   model="models/gemini-2.5-flash",
    contents="Say hello , I am testing the gemini api with langsmith evaluation framework"
)

print(response.text)
