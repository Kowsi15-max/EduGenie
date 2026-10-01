import json
import os
from functools import lru_cache

from dotenv import load_dotenv
from google import genai

load_dotenv()

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
API_KEY = os.getenv("GEMINI_API_KEY")


class GeminiConfigurationError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def get_client():
    if not API_KEY:
        raise GeminiConfigurationError(
            "GEMINI_API_KEY is missing in .env"
        )

    return genai.Client(api_key=API_KEY)


def generate_text(
    prompt: str,
    *,
    temperature: float = 0.4,
    max_output_tokens: int = 1500,
) -> str:

    client = get_client()

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
    )

    text = response.text.strip()

    if not text:
        raise RuntimeError("Gemini returned an empty response.")

    return text


def generate_json(
    prompt: str,
    response_schema,
    *,
    temperature: float = 0.3,
    max_output_tokens: int = 2000,
) -> str:

    client = get_client()

    schema = response_schema.model_json_schema()

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": schema,
        },
    )

    text = response.text.strip()

    if not text:
        raise RuntimeError("Gemini returned an empty JSON response.")

    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            "Gemini returned invalid JSON."
        ) from exc

    return text