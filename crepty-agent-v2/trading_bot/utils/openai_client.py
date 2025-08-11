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
        for attempt in range(1, max_attempts + 1):
            raw = self.ask(prompt, model=model)
            # Try to locate JSON block
            json_text = raw
            if '```' in raw:
                # Extract first fenced block
                parts = raw.split('```')
                if len(parts) >= 2:
                    # skip possible language hint
                    candidate = parts[1]
                    if candidate.strip().startswith('{'):
                        json_text = candidate
                    else:
                        # maybe third part
                        for p in parts[1:]:
                            if p.strip().startswith('{'):
                                json_text = p
                                break
            json_text = json_text.strip().strip('`')
            try:
                data = json.loads(json_text)
            except Exception:
                if attempt == max_attempts:
                    raise RuntimeError(f"Failed to parse JSON from model response: {raw[:200]}...")
                continue
            if schema_validator and not schema_validator(data):
                if attempt == max_attempts:
                    raise RuntimeError(f"Model JSON schema invalid after {max_attempts} attempts: {data}")
                # Append correction instruction
                prompt = prompt + "\nThe previous JSON did not follow the required schema. Please output ONLY a corrected JSON object."
                continue
            return data
        return {}
