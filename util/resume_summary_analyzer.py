import re
from typing import Any, Dict, List, Optional, Set, Tuple

from util import constants


def _normalize_text(text: Optional[str]) -> str:
    return text or ""


def _detect_skills(text: str) -> Tuple[Set[str], Dict[str, List[str]]]:
    lower_text = text.lower()
    category_hits: Dict[str, Set[str]] = {key: set() for key in constants.SKILL_CATEGORIES}

    for category, keywords in constants.SKILL_CATEGORIES.items():
        for keyword in keywords:
            if keyword in lower_text:
                category_hits[category].add(keyword)

    flat_skills = set().union(*category_hits.values())
    ordered_categories = {
        category: sorted(values) for category, values in category_hits.items() if values
    }
    return flat_skills, ordered_categories


def _extract_years_of_experience(text: str) -> Optional[float]:
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)", text.lower())
    if not matches:
        return None
    try:
        return max(float(match) for match in matches)
    except ValueError:
        return None


def _experience_band(years: Optional[float]) -> str:
    if years is None:
        return "unknown"
    if years >= 10:
        return "principal"
    if years >= 7:
        return "senior"
    if years >= 4:
        return "mid-level"
    if years >= 1:
        return "junior"
    return "entry"


def _extract_knowledge_mentions(text: str) -> List[Dict[str, str]]:
    sentences = re.split(r"(?:\.\s+|\n+)", text)
    mentions: List[Dict[str, str]] = []
    for sentence in sentences:
        stripped = sentence.strip()
        if not stripped:
            continue
        lower_sentence = stripped.lower()
        level_tag: Optional[str] = None
        for level, keywords in constants.KNOWLEDGE_LEVEL_KEYWORDS.items():
            if any(keyword in lower_sentence for keyword in keywords):
                level_tag = level
                break
        if not level_tag:
            continue
        skills, _ = _detect_skills(lower_sentence)
        mentions.append(
            {
                "level": level_tag,
                "excerpt": stripped,
                "skills": sorted(skills),
            }
        )
    return mentions


def _quantification_gaps(text: str) -> List[str]:
    suggestions: List[str] = []
    sentences = [segment.strip() for segment in re.split(r"[.\n]", text) if segment.strip()]
    for sentence in sentences:
        lower_sentence = sentence.lower()
        if any(verb in lower_sentence for verb in constants.QUANTIFIABLE_VERBS):
            if not re.search(r"\b\d+%|\b\d+(?:\.\d+)?\b", sentence):
                suggestions.append(
                    "Add a measurable outcome to: '{}'".format(sentence[:120].strip())
                )
    return suggestions


def _extract_llm_score(text: str) -> Optional[float]:
    pattern = re.search(
        r"(?:ats\s*score|overall\s*score)\s*[:\-]\s*(\d{1,3})", text.lower()
    )
    if not pattern:
        return None
    try:
        score = float(pattern.group(1))
        return max(0.0, min(score, 100.0))
    except ValueError:
        return None


def _estimate_llm_score(
    summary_text: str,
    years_experience: Optional[float],
    skills: List[str],
    quantifiable_highlights: List[str],
    quantification_suggestions: List[str],
    leadership_signals: List[str],
) -> float:
    """
    Heuristic fallback that approximates an ATS-style LLM score when the upstream
    model does not provide one explicitly.
    """
    whitespace_stripped = summary_text.strip()
    score = 55.0 if whitespace_stripped else 42.0

    if years_experience is not None:
        if years_experience >= 10:
            score += 12.0
        elif years_experience >= 7:
            score += 10.0
        elif years_experience >= 4:
            score += 7.0
        elif years_experience >= 1:
            score += 4.0
        else:
            score += 2.0
    else:
        score -= 6.0

    skill_bonus = min(15.0, len(skills) * 0.6)
    score += skill_bonus

    metrics_present = bool(
        re.search(r"\b\d+(?:\.\d+)?\s*(?:%|x|k|m|million|billion)?", summary_text.lower())
    ) or bool(quantifiable_highlights)
    if metrics_present:
        score += 8.0
    else:
        score -= 5.0

    if quantification_suggestions:
        score -= min(10.0, len(quantification_suggestions) * 2.0)

    summary_lower = summary_text.lower()
    leadership_keywords = constants.EXPERIENCE_LEVEL_KEYWORDS.get("leadership", set())
    impact_keywords = constants.EXPERIENCE_LEVEL_KEYWORDS.get("impact", set())
    delivery_keywords = constants.EXPERIENCE_LEVEL_KEYWORDS.get("delivery", set())

    if any(keyword in summary_lower for keyword in leadership_keywords):
        score += 4.0
    if leadership_signals:
        score += min(6.0, len(leadership_signals) * 2.0)

    impact_hits = sum(1 for keyword in impact_keywords if keyword in summary_lower)
    delivery_hits = sum(1 for keyword in delivery_keywords if keyword in summary_lower)
    if impact_hits:
        score += min(6.0, impact_hits * 1.5)
    if delivery_hits:
        score += min(4.0, delivery_hits * 1.5)

    return round(max(0.0, min(score, 100.0)), 2)


def analyze_resume_summary(summary_payload: Any) -> Dict[str, object]:
    """
    Analyse the LLM generated resume summary to extract structured signals
    used downstream during JD comparison.
    """
    direct_skills: Set[str] = set()
    direct_years: Optional[float] = None
    direct_knowledge: List[Dict[str, str]] = []
    direct_quant_suggestions: List[str] = []
    direct_quant_highlights: List[str] = []
    direct_leadership_signals: List[str] = []
    summary_text: str

    if isinstance(summary_payload, dict):
        summary_text = _normalize_text(summary_payload.get("summary_text"))
        for collection_key in ("core_skills", "tooling", "domain_experience"):
            direct_skills.update(
                {
                    str(value).lower()
                    for value in summary_payload.get(collection_key, [])
                    if isinstance(value, str) and value.strip()
                }
            )
        direct_years = summary_payload.get("total_years_experience")
        if isinstance(direct_years, str):
            try:
                direct_years = float(direct_years)
            except ValueError:
                direct_years = None
        direct_quant_suggestions = [
            str(value).strip()
            for value in summary_payload.get("quantification_suggestions", [])
            if isinstance(value, str) and value.strip()
        ]
        knowledge_statements = summary_payload.get("knowledge_statements", [])
        if isinstance(knowledge_statements, list):
            for statement in knowledge_statements:
                if isinstance(statement, str) and statement.strip():
                    skills_hit, _ = _detect_skills(statement)
                    direct_knowledge.append(
                        {
                            "level": "unspecified",
                            "excerpt": statement.strip(),
                            "skills": sorted(skills_hit),
                        }
                    )
        direct_quant_highlights = [
            str(value).strip()
            for value in summary_payload.get("quantifiable_highlights", [])
            if isinstance(value, str) and value.strip()
        ]
        direct_leadership_signals = [
            str(value).strip()
            for value in summary_payload.get("leadership_experience", [])
            if isinstance(value, str) and value.strip()
        ]
    else:
        summary_text = _normalize_text(summary_payload)

    skills, _ = _detect_skills(summary_text)
    years_of_experience = _extract_years_of_experience(summary_text)
    if direct_years is not None:
        years_of_experience = direct_years

    combined_skills = skills.union(direct_skills)
    skills_list = sorted(combined_skills)

    skills_by_category: Dict[str, List[str]] = {}
    for category, keywords in constants.SKILL_CATEGORIES.items():
        hits = sorted({skill for skill in combined_skills if skill in keywords})
        if hits:
            skills_by_category[category] = hits

    knowledge_mentions = _extract_knowledge_mentions(summary_text)
    if direct_knowledge:
        knowledge_mentions.extend(direct_knowledge)

    quantification_suggestions = _quantification_gaps(summary_text)
    if direct_quant_suggestions:
        quantification_suggestions.extend(
            suggestion
            for suggestion in direct_quant_suggestions
            if suggestion not in quantification_suggestions
        )

    llm_score_raw = _extract_llm_score(summary_text)
    if llm_score_raw is not None:
        llm_score = llm_score_raw
        llm_score_available = True
    else:
        llm_score = _estimate_llm_score(
            summary_text=summary_text,
            years_experience=years_of_experience,
            skills=skills_list,
            quantifiable_highlights=direct_quant_highlights,
            quantification_suggestions=quantification_suggestions,
            leadership_signals=direct_leadership_signals,
        )
        llm_score_available = True

    return {
        "raw_summary": summary_text,
        "skills": skills_list,
        "skills_by_category": skills_by_category,
        "years_experience": years_of_experience,
        "experience_band": _experience_band(years_of_experience),
        "knowledge_mentions": knowledge_mentions,
        "quantification_suggestions": quantification_suggestions,
        "llm_score": llm_score,
        "llm_score_available": llm_score_available,
    }
