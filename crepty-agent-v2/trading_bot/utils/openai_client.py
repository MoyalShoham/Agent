import os
import json
import time
import random
import openai
from dotenv import load_dotenv
from typing import Any, Dict, Optional, Callable

load_dotenv()

class OpenAIClient:
    def __init__(self, api_key=None, max_retries: int = 3, backoff_base: float = 1.5):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        openai.api_key = self.api_key
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    def ask(self, prompt: str, model: str = 'gpt-4o-mini', **kwargs):
        # Default model is 'gpt-4o-mini'.
        attempt = 0
        last_err = None
        while attempt < self.max_retries:
            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=kwargs.get('temperature', 0.2),
                    max_tokens=kwargs.get('max_tokens', 800)
                )
                return response['choices'][0]['message']['content']
            except Exception as e:
                last_err = e
                sleep_for = (self.backoff_base ** attempt) + random.uniform(0, 0.25)
                time.sleep(sleep_for)
                attempt += 1
        raise RuntimeError(f"OpenAI API error after {self.max_retries} retries: {last_err}")

    def ask_json(self, prompt: str, schema_validator: Optional[Callable[[Dict[str, Any]], bool]] = None, model: str = 'gpt-4o-mini', max_attempts: int = 3) -> Dict[str, Any]:
        """Request JSON structured output; retry parse/validation if malformed.
        schema_validator: returns True if schema OK.
        """
        prompt = prompt + "\nRespond ONLY with a single minified JSON object. No markdown fences, no prose."
        for attempt in range(1, max_attempts + 1):
            raw = self.ask(prompt, model=model)
            json_text = raw.strip()
            # Extract fenced block if present
            if '```' in raw:
                parts = raw.split('```')
                for p in parts[1:]:
                    pt = p.strip()
                    if pt.startswith('{'):
                        json_text = pt
                        break
            # Fallback: locate outermost braces
            if not json_text.strip().startswith('{'):
                start = raw.find('{')
                end = raw.rfind('}')
                if start != -1 and end != -1 and end > start:
                    json_text = raw[start:end+1]
            json_text = json_text.strip().lstrip('`').rstrip('`')
            try:
                data = json.loads(json_text)
            except Exception:
                if attempt == max_attempts:
                    raise RuntimeError(f"Failed to parse JSON from model response: {raw[:200]}...")
                continue
            if schema_validator and not schema_validator(data):
                if attempt == max_attempts:
                    raise RuntimeError(f"Model JSON schema invalid after {max_attempts} attempts: {data}")
                prompt += "\nThe previous JSON did not match the required schema. Output ONLY corrected JSON."
                continue
            return data
        return {}
