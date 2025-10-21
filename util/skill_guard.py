import json
import re
from functools import lru_cache
from typing import List, Set

from util import constants
from util.llm_helpers import coerce_json
from util.simpleagent import MyAgent
from util.system_prompt import prompt_skill_guard

_MAX_SKILLS_FOR_LLM = 40  # Keeps LLM payload sizes manageable.


@lru_cache(maxsize=1)
def _get_agent() -> MyAgent:
    return MyAgent(system_prompt=prompt_skill_guard)


@lru_cache(maxsize=256)
def _filter_tuple(skills_tuple: tuple[str, ...]) -> tuple[str, ...]:
    if not skills_tuple:
        return ()

    agent = _get_agent()
    payload = {"skills": list(skills_tuple)}
    response = agent(message=json.dumps(payload), temperature=0.0)
    parsed, raw_text = coerce_json(response)

    if isinstance(parsed, dict):
        cleaned = parsed.get("skills")
        if isinstance(cleaned, list):
            return tuple(str(item).strip() for item in cleaned if str(item).strip())
    return skills_tuple


def _flatten_taxonomy() -> Set[str]:
    flattened: Set[str] = set()
    for keywords in constants.SKILL_CATEGORIES.values():
        for keyword in keywords:
            flattened.add(keyword.lower())
    return flattened


_TAXONOMY_CACHE = _flatten_taxonomy()
_LOCATION_TERMS = {
    "india",
    "bengaluru",
    "bangalore",
    "kolkata",
    "delhi",
    "west",
    "bengal",
    "usa",
    "london",
    "remote",
}
_GENERIC_TERMS = {
    "developer",
    "development",
    "applications",
    "application",
    "team",
    "teams",
    "project",
    "projects",
    "experience",
    "customers",
    "users",
    "solutions",
    "building",
    "cutting-edge",
    "cutting",
    "edge",
    "mobile",
    "devices",
    "mission",
    "impact",
    "growth",
}

_TECH_KEYWORDS = {
    "spline",
    "spline3d",
    "framer",
    "framer motion",
    "vercel",
    "netlify",
    "firebase",
    "huggingface",
    "hugging",
    "cloudflare",
    "gcp",
    "google cloud",
    "rag",
    "retrieval",
    "augmented",
}


def _looks_like_location(skill: str) -> bool:
    tokens = [token.lower() for token in re.split(r"[^\w]+", skill) if token]
    return any(token in _LOCATION_TERMS for token in tokens)


def _looks_generic(skill: str) -> bool:
    tokens = [token.lower() for token in re.split(r"[^\w]+", skill) if token]
    return all(token in _GENERIC_TERMS or len(token) <= 2 for token in tokens)


def _is_technical_hint(skill: str) -> bool:
    lower = skill.lower()
    if lower in _TAXONOMY_CACHE:
        return True
    tokens = [token.lower() for token in re.split(r"[^\w]+", lower) if token]
    if any(token in _TAXONOMY_CACHE for token in tokens):
        return True
    if any(token in _TECH_KEYWORDS for token in tokens):
        return True
    if re.search(r"\b(api|cloud|framework|studio|platform|pipeline|dataset|ml|ai|model|analytics|sql|devops|testing|design|render|3d)\b", lower):
        return True
    if re.search(r"\d", skill):
        return True
    if any(symbol in skill for symbol in ("/", "-", "(", ")", "+", "#")):
        return True
    if lower.endswith(("js", "sql")):
        return True
    return False


def filter_skills_via_llm(skills: List[str]) -> List[str]:
    if not skills:
        return []

    unique_skills = []
    seen = set()
    for skill in skills:
        normalized = skill.strip()
        if normalized and normalized.lower() not in seen:
            unique_skills.append(normalized)
            seen.add(normalized.lower())

    llm_candidates = unique_skills[:_MAX_SKILLS_FOR_LLM]
    filtered = _filter_tuple(tuple(llm_candidates))
    filtered_set = set(filtered)

    recovered: List[str] = []
    for skill in unique_skills:
        if skill in filtered_set:
            recovered.append(skill)
            continue
        if _looks_like_location(skill) or _looks_generic(skill):
            continue
        if _is_technical_hint(skill):
            recovered.append(skill)

    deduped: List[str] = []
    seen_final = set()
    combined_sequence = list(filtered) + recovered
    for skill in combined_sequence:
        if skill and skill.lower() not in seen_final:
            deduped.append(skill)
            seen_final.add(skill.lower())

    return deduped
