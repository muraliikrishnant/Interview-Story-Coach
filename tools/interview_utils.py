import json
import os
import re
from pathlib import Path

from google import genai
from google.genai import types

_PROJECT  = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
_MODEL    = "gemini-2.0-flash-001"

_client = genai.Client(
    vertexai=True,
    project=_PROJECT,
    location=_LOCATION,
)

_SYSTEM_PROMPT = (
    "You are an expert behavioral interview coach who helps students, new grads, and "
    "international candidates prepare for interviews using the STAR framework. "
    "Return only valid JSON — no markdown, no explanation."
)


def call_llm(prompt: str) -> str | None:
    try:
        response = _client.models.generate_content(
            model=_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM_PROMPT,
                max_output_tokens=1500,
                temperature=0.7,
            ),
        )
        return response.text
    except Exception as e:
        print(f"LLM call failed: {e}")
        return None


def extract_json_object(text: str) -> dict | None:
    clean = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(clean)
    except Exception:
        pass
    match = re.search(r"\{.*\}", clean, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass
    return None


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def words(text: str) -> list[str]:
    return text.split()


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))
