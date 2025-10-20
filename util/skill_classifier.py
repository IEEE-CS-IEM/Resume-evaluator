import json
from functools import lru_cache
from typing import Dict, List

from util.llm_helpers import coerce_json
from util.simpleagent import MyAgent
from util.system_prompt import prompt_skill_classifier

_CATEGORIES = [
    "programming_languages",
    "frameworks",
    "libraries_packages",
    "cloud_platforms",
    "developer_tools",
    "ml_ai_tools",
    "design_tools",
    "devops_infra",
    "databases",
    "frontend_tooling",
    "mobile_tooling",
    "other",
]


@lru_cache(maxsize=1)
def _get_agent() -> MyAgent:
    return MyAgent(system_prompt=prompt_skill_classifier)


def classify_skills(skills: List[str]) -> Dict[str, List[str]]:
    if not skills:
        return {category: [] for category in _CATEGORIES}

    agent = _get_agent()
    payload = {"skills": skills}
    response = agent(message=json.dumps(payload), temperature=0.0)
    parsed, raw_text = coerce_json(response)

    if not isinstance(parsed, dict):
        return {category: [] for category in _CATEGORIES}

    result: Dict[str, List[str]] = {}
    for category in _CATEGORIES:
        values = parsed.get(category, [])
        if isinstance(values, list):
            cleaned = [str(item).strip() for item in values if str(item).strip()]
            result[category] = cleaned
        else:
            result[category] = []

    return result
