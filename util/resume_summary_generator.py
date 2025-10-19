from typing import Any, Dict

from util.llm_helpers import coerce_json
from util.simpleagent import MyAgent
from util.system_prompt import prompt_resume_summary


def generate_resume_summary(resume_text: str, temperature: float = 0.2) -> Dict[str, Any]:
    """
    Uses the configured LLM provider to distil a resume into structured JSON data.
    """
    agent = MyAgent(system_prompt=prompt_resume_summary)
    response = agent(message=resume_text, temperature=temperature)
    payload, raw_text = coerce_json(response)

    if not payload:
        # Basic fallback if JSON parsing failed.
        payload = {
            "summary_text": raw_text.strip(),
            "core_skills": [],
            "tooling": [],
            "domain_experience": [],
            "quantifiable_highlights": [],
            "leadership_experience": [],
            "total_years_experience": None,
            "seniority": None,
            "knowledge_statements": [],
            "quantification_suggestions": [],
        }

    # Ensure mandatory keys exist.
    payload.setdefault("summary_text", raw_text.strip())
    payload.setdefault("core_skills", [])
    payload.setdefault("tooling", [])
    payload.setdefault("domain_experience", [])
    payload.setdefault("quantifiable_highlights", [])
    payload.setdefault("leadership_experience", [])
    payload.setdefault("total_years_experience", None)
    payload.setdefault("seniority", None)
    payload.setdefault("knowledge_statements", [])
    payload.setdefault("quantification_suggestions", [])

    return payload
