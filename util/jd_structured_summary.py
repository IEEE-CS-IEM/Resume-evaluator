import re
from typing import Any, Dict, List, Optional

from util import constants
from util.llm_helpers import coerce_json
from util.simpleagent import MyAgent
from util.system_prompt import prompt_jd_structured


def _experience_band(years: Optional[float]) -> str:
    if years is None:
        return "unspecified"
    if years >= 10:
        return "principal"
    if years >= 7:
        return "senior"
    if years >= 4:
        return "mid-level"
    if years >= 1:
        return "junior"
    return "entry"


def _categorise_skills(skills: List[str]) -> Dict[str, List[str]]:
    categories: Dict[str, List[str]] = {}
    lower_skills = {skill.lower(): skill for skill in skills}
    for category, keywords in constants.SKILL_CATEGORIES.items():
        hits = []
        for keyword in keywords:
            if keyword in lower_skills:
                hits.append(lower_skills[keyword])
        if hits:
            categories[category] = sorted(set(hits))
    return categories


def _derive_years(requirement: Dict[str, Any], jd_text: str) -> Optional[float]:
    minimum = requirement.get("minimum_years")
    maximum = requirement.get("maximum_years")
    candidates = []
    for value in (minimum, maximum):
        if isinstance(value, (int, float)):
            candidates.append(float(value))
        elif isinstance(value, str):
            try:
                candidates.append(float(value))
            except ValueError:
                continue
    if candidates:
        return max(candidates)

    matches = re.findall(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)", jd_text.lower())
    if matches:
        try:
            return max(float(match) for match in matches)
        except ValueError:
            return None
    return None


def _normalise_list(values: Optional[List[Any]]) -> List[str]:
    if not values:
        return []
    result = []
    for value in values:
        if isinstance(value, str):
            cleaned = value.strip()
            if cleaned:
                result.append(cleaned)
    return result


def summarise_job_description(jd_text: str, temperature: float = 0.0) -> Dict[str, Any]:
    """
    Summarise a job description into structured data using the configured LLM.
    """
    agent = MyAgent(system_prompt=prompt_jd_structured)
    response = agent(message=jd_text, temperature=temperature)
    payload, raw_text = coerce_json(response)

    if not payload:
        payload = {
            "role_title": None,
            "seniority_level": "unspecified",
            "summary": raw_text.strip(),
            "must_have_skills": [],
            "nice_to_have_skills": [],
            "tooling": [],
            "experience_requirement": {
                "minimum_years": None,
                "maximum_years": None,
                "described_range": None,
            },
            "knowledge_expectations": [],
            "domains": [],
        }

    role_title = payload.get("role_title")
    seniority = payload.get("seniority_level", "unspecified")
    summary_text = payload.get("summary") or raw_text.strip()
    must_have = _normalise_list(payload.get("must_have_skills"))
    nice_to_have = _normalise_list(payload.get("nice_to_have_skills"))
    tooling = _normalise_list(payload.get("tooling"))
    knowledge_expectations = _normalise_list(payload.get("knowledge_expectations"))
    domains = _normalise_list(payload.get("domains"))
    experience_requirement = payload.get("experience_requirement") or {}

    combined_skills = sorted(set(must_have + nice_to_have + tooling))
    skills_by_category = _categorise_skills(combined_skills)
    knowledge_structured: List[Dict[str, Any]] = []
    for statement in knowledge_expectations:
        lower_statement = statement.lower()
        statement_skills = []
        for category_keywords in constants.SKILL_CATEGORIES.values():
            for keyword in category_keywords:
                if keyword in lower_statement:
                    statement_skills.append(keyword)
        knowledge_structured.append(
            {
                "text": statement,
                "skills": sorted(set(statement_skills)),
                "categories": _categorise_skills(sorted(set(statement_skills))),
            }
        )

    required_years = _derive_years(experience_requirement, jd_text)

    return {
        "raw_text": jd_text,
        "role_title": role_title,
        "seniority_level": seniority,
        "llm_summary": summary_text,
        "summary": summary_text,
        "skills": combined_skills,
        "skills_by_category": skills_by_category,
        "must_have_skills": must_have,
        "nice_to_have_skills": nice_to_have,
        "tooling": tooling,
        "domains": domains,
        "required_years": required_years,
        "required_experience_band": _experience_band(required_years),
        "knowledge_requirements": knowledge_structured,
        "experience_requirement": experience_requirement,
    }
