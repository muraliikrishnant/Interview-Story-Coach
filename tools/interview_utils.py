import json
import os
import re
from pathlib import Path

import anthropic

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

_SYSTEM_PROMPT = (
    "You are an expert behavioral interview coach who helps students, new grads, and "
    "international candidates prepare for interviews using the STAR framework. "
    "Return only valid JSON — no markdown, no explanation."
)


def call_llm(prompt: str) -> str | None:
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1500,
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
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
