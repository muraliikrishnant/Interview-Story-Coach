import datetime as dt
import json
from pathlib import Path
from typing import Any

from crewai.tools import tool

from tools.interview_utils import call_llm, extract_json_object, slugify, words, write_json

STORY2STAR_DIR = Path("output/story2star")

SKILL_FOCUSES = {
    "leadership": "Leadership",
    "teamwork": "Teamwork & Collaboration",
    "teamwork & collaboration": "Teamwork & Collaboration",
    "problem solving": "Problem Solving",
    "conflict resolution": "Conflict Resolution",
    "ownership": "Ownership & Initiative",
    "ownership & initiative": "Ownership & Initiative",
    "communication": "Communication",
    "adaptability": "Adaptability",
    "technical skills": "Technical Skills",
}

ROLE_PRESETS = {
    "software engineer intern",
    "product/business analyst",
    "marketing/comms",
    "general / any role",
    "general",
}


def _normalize_skill(skill_focus: str) -> str:
    key = skill_focus.strip().lower()
    return SKILL_FOCUSES.get(key, skill_focus.strip() or "Problem Solving")


def _normalize_length(answer_length: str) -> str:
    length = answer_length.strip().lower()
    if "90" in length or "short" in length or "concise" in length:
        return "Concise (90 sec)"
    return "Detailed (3-4 min)"


def _as_list(value: Any, fallback: list[str], *, limit: int = 4) -> list[str]:
    if not isinstance(value, list):
        return fallback[:limit]
    cleaned = [str(item).strip() for item in value if str(item).strip()]
    return (cleaned or fallback)[:limit]


def _as_text(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    return text or fallback


def _clamp_score(value: Any) -> int:
    try:
        return max(1, min(10, int(value)))
    except (TypeError, ValueError):
        return 6


def _infer_action(story: str, skill_focus: str) -> str:
    lower = story.lower()
    if any(term in lower for term in ("bug", "debug", "production", "fix")):
        return (
            "I investigated the issue, isolated the highest-risk failure path, tested a focused fix, "
            "and communicated progress so the team could keep moving."
        )
    if any(term in lower for term in ("conflict", "disagree", "timeline", "manager")):
        return (
            "I listened to each concern, reframed the discussion around shared goals, proposed a "
            "practical compromise, and kept the conversation focused on evidence."
        )
    if any(term in lower for term in ("team", "group", "club", "class")):
        return (
            "I helped clarify responsibilities, broke the work into visible milestones, checked in "
            "with teammates, and stepped in where progress was blocked."
        )
    if "technical" in skill_focus.lower():
        return (
            "I broke the technical problem into smaller parts, built a first version, validated it, "
            "and improved it based on what I learned."
        )
    return (
        "I took ownership of the next step, organized the work, asked for feedback at the right "
        "moments, and followed through until there was a usable outcome."
    )


def _fallback_story2star(story: str, skill_focus: str, target_role: str, answer_length: str) -> dict:
    story_words = words(story)
    concise = _normalize_length(answer_length).startswith("Concise")
    role_context = target_role.strip() or "the target role"
    metric_hint = " If you can, add a number such as time saved, grade impact, users helped, or defects reduced."
    action = _infer_action(story, skill_focus)
    confidence = 7 if len(story_words) >= 50 else 5
    if any(char.isdigit() for char in story):
        confidence += 1

    situation = (
        f"In a recent experience related to {skill_focus.lower()}, I was working in a context where "
        f"the outcome mattered for my team and connected to skills needed for {role_context}."
    )
    task_text = (
        "My responsibility was to turn a rough or blocked situation into a clear plan, align the "
        "people involved, and make sure the final result was useful."
    )
    result = (
        "The work gave the team a clearer path forward and produced a stronger final outcome."
        + metric_hint
    )
    if concise:
        situation = (
            f"I was in a {skill_focus.lower()} situation where the outcome mattered for my team "
            f"and connected to skills needed for {role_context}."
        )

    short_version = (
        f"I handled a {skill_focus.lower()} situation where the initial experience was messy and "
        f"needed structure. My task was to clarify the goal, take useful action, and keep the work "
        f"moving. {action} The result was a better outcome for the team, and I learned how to turn "
        f"ambiguity into progress."
    )

    return {
        "star": {
            "situation": situation,
            "task": task_text,
            "action": action,
            "result": result,
        },
        "shortVersion": short_version,
        "bullets": [
            f"Structured an ambiguous {skill_focus.lower()} experience into a clear plan.",
            "Took ownership of communication, priorities, and execution.",
            "Delivered a stronger final outcome and identified a metric to quantify impact.",
        ],
        "followUpQuestions": [
            "What would you do differently now?",
            "How did the people involved respond to your approach?",
            "Can you quantify the impact or describe how success was measured?",
        ],
        "improvementTips": [
            "Add one specific metric to the Result section, even if it is an estimate.",
            "Name the stakeholder or audience so the Situation has clearer context.",
            "Make your personal Action more specific than the team's general effort.",
        ],
        "confidenceScore": min(10, confidence),
        "confidenceReason": (
            "The story has enough direction to become a STAR answer, but it will be stronger with "
            "specific metrics, stakeholders, and a sharper personal contribution."
        ),
    }


def _story2star_prompt(story: str, skill_focus: str, target_role: str, answer_length: str) -> str:
    return f"""
You are an expert interview coach. A user has provided a rough work or school experience.
Transform it into a polished behavioral interview answer using the STAR framework.

User's raw story: {story}
Target skill/competency: {skill_focus}
Target role: {target_role or "General / Any Role"}
Desired length: {answer_length}

Rules:
- Keep the user's facts intact. Do not invent company names, grades, numbers, or outcomes.
- If a metric is missing, suggest where to add one instead of fabricating it.
- Make the Action section the strongest and most personal section.
- Write for students, new grads, and international students preparing for interviews.

Return ONLY a JSON object with this exact structure:
{{
  "star": {{
    "situation": "...",
    "task": "...",
    "action": "...",
    "result": "..."
  }},
  "shortVersion": "...",
  "bullets": ["...", "...", "..."],
  "followUpQuestions": ["...", "...", "..."],
  "improvementTips": ["...", "...", "..."],
  "confidenceScore": 7,
  "confidenceReason": "..."
}}
"""


def _validate_story2star(parsed: dict | None, fallback: dict) -> dict:
    if not parsed:
        return fallback

    parsed_star = parsed.get("star") if isinstance(parsed.get("star"), dict) else {}
    star = {
        "situation": _as_text(parsed_star.get("situation"), fallback["star"]["situation"]),
        "task": _as_text(parsed_star.get("task"), fallback["star"]["task"]),
        "action": _as_text(parsed_star.get("action"), fallback["star"]["action"]),
        "result": _as_text(parsed_star.get("result"), fallback["star"]["result"]),
    }
    return {
        "star": star,
        "shortVersion": _as_text(parsed.get("shortVersion"), fallback["shortVersion"]),
        "bullets": _as_list(parsed.get("bullets"), fallback["bullets"]),
        "followUpQuestions": _as_list(
            parsed.get("followUpQuestions"), fallback["followUpQuestions"], limit=3
        ),
        "improvementTips": _as_list(parsed.get("improvementTips"), fallback["improvementTips"], limit=3),
        "confidenceScore": _clamp_score(parsed.get("confidenceScore")),
        "confidenceReason": _as_text(parsed.get("confidenceReason"), fallback["confidenceReason"]),
    }


def build_story2star_coaching(
    story: str,
    skill_focus: str,
    target_role: str = "General / Any Role",
    answer_length: str = "Detailed (3-4 min)",
    *,
    save: bool = True,
) -> dict:
    clean_story = story.strip()
    if not clean_story:
        raise ValueError("story is required")

    skill = _normalize_skill(skill_focus)
    role = target_role.strip() or "General / Any Role"
    length = _normalize_length(answer_length)
    fallback = _fallback_story2star(clean_story, skill, role, length)
    parsed = extract_json_object(call_llm(_story2star_prompt(clean_story, skill, role, length)) or "")
    result = _validate_story2star(parsed, fallback)
    result["metadata"] = {
        "app": "Story2STAR",
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "skillFocus": skill,
        "targetRole": role,
        "answerLength": length,
        "sourceStory": clean_story,
    }

    if save:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = STORY2STAR_DIR / f"{timestamp}_{slugify(skill)}.json"
        write_json(path, result)
        result["metadata"]["savedPath"] = str(path)
    return result


def render_story2star_markdown(result: dict) -> str:
    star = result["star"]
    metadata = result.get("metadata", {})
    bullets = "\n".join(f"- {item}" for item in result["bullets"])
    questions = "\n".join(
        f"{index}. {item}" for index, item in enumerate(result["followUpQuestions"], start=1)
    )
    tips = "\n".join(f"- {item}" for item in result["improvementTips"])
    return f"""# Story2STAR Coaching

**Skill focus:** {metadata.get("skillFocus", "")}
**Target role:** {metadata.get("targetRole", "")}
**Answer length:** {metadata.get("answerLength", "")}

## STAR Answer

**Situation:** {star["situation"]}

**Task:** {star["task"]}

**Action:** {star["action"]}

**Result:** {star["result"]}

## Quick Bullets

{bullets}

## Shorter Version

{result["shortVersion"]}

## Follow-up Questions

{questions}

## Make It Stronger

{tips}

## Confidence Score

**{result["confidenceScore"]}/10** - {result["confidenceReason"]}
"""


@tool("story2star_coach")
def story2star_coach(
    story: str,
    skill_focus: str,
    target_role: str = "General / Any Role",
    answer_length: str = "Detailed (3-4 min)",
) -> str:
    """
    Turn a rough school or work experience into a full Story2STAR coaching result.
    """
    result = build_story2star_coaching(story, skill_focus, target_role, answer_length)
    return json.dumps(result, indent=2)
