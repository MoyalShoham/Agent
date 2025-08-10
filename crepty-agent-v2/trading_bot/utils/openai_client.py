import os
import openai
from dotenv import load_dotenv

load_dotenv()

class OpenAIClient:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        openai.api_key = self.api_key

    def ask(self, prompt: str, model: str = 'gpt-3.5-turbo', **kwargs):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            raise RuntimeError(f"OpenAI API error: {e}")
