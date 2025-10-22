import json
from typing import Any, Dict

from util.llm_helpers import coerce_json
from util.simpleagent import MyAgent
from util.system_prompt import prompt_fit_evaluation


def compare_resume_to_jd(
    resume_summary: Dict[str, Any],
    jd_summary: Dict[str, Any],
    scoring_breakdown: Dict[str, Any],
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    Ask the LLM to provide a narrative fit evaluation using the structured
    summaries and the computed scoring data.
    """
    agent = MyAgent(system_prompt=prompt_fit_evaluation)
    payload = {
        "resume_summary": resume_summary,
        "jd_summary": jd_summary,
        "scoring_breakdown": scoring_breakdown,
    }
    response = agent(message=json.dumps(payload), temperature=temperature)
    parsed, raw_text = coerce_json(response)

    if not parsed:
        parsed = {
            "overall_fit": "Almost Ready",
            "ats_score": scoring_breakdown.get("final_ats_score"),
            "strengths": [],
            "gaps": [],
            "recommendations": [],
            "narrative": raw_text.strip(),
        }

    parsed.setdefault("overall_fit", "Almost Ready")
    parsed.setdefault("ats_score", scoring_breakdown.get("final_ats_score"))
    parsed.setdefault("strengths", [])
    parsed.setdefault("gaps", [])
    parsed.setdefault("recommendations", [])
    parsed.setdefault("narrative", raw_text.strip())

    return parsed
