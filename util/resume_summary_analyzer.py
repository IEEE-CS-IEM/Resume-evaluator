import re
from typing import Any, Dict, List, Optional

from util import constants
from util.keyword_extractor import extract_keywords
from util.skill_guard import filter_skills_via_llm
from util.skill_classifier import classify_skills


def _normalize_text(text: Optional[str]) -> str:
    return text or ""


def _keyword_present(text: str, keyword: str) -> bool:
    pattern = rf"(?<!\w){re.escape(keyword)}(?!\w)"
    return re.search(pattern, text) is not None


def _find_raw_occurrence(candidate: str, text: str) -> Optional[str]:
    pattern = re.compile(rf"(?<!\w){re.escape(candidate)}(?!\w)", re.IGNORECASE)
    match = pattern.search(text)
    if match:
        return text[match.start() : match.end()]
    return None


def _looks_like_person_name(raw: Optional[str]) -> bool:
    if not raw:
        return False
    tokens = [token for token in re.split(r"[^\w]+", raw) if token]
    if not tokens or len(tokens) > 3:
        return False
    alphabetical = all(token.isalpha() for token in tokens)
    title_case_tokens = sum(token[:1].isupper() for token in tokens)
    if alphabetical and title_case_tokens == len(tokens):
        return True
    return False


def _canonical_skill_key(value: str) -> str:
    lowered = value.lower()
    lowered = re.sub(r"\(.*?\)", " ", lowered)
    lowered = lowered.replace("enabled", "")
    lowered = re.sub(r"[^a-z0-9+#]+", " ", lowered)
    lowered = re.sub(r"\b(retrieval augmented generation|retrieval-augmented generation|rag)\b", "rag", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _detect_skills(text: str) -> List[str]:
    lower_text = text.lower()
    extracted_keywords = set(extract_keywords(text))
    canonical_lookup: Dict[str, str] = {}
    for keywords in constants.SKILL_CATEGORIES.values():
        for keyword in keywords:
            canonical_lookup.setdefault(keyword.lower(), keyword)
            if _keyword_present(lower_text, keyword.lower()):
                extracted_keywords.add(keyword.lower())

    normalized: Dict[str, str] = {}
    for candidate in extracted_keywords:
        original = canonical_lookup.get(candidate, candidate)
        raw = _find_raw_occurrence(original, text)
        if not canonical_lookup.get(candidate) and _looks_like_person_name(raw):
            continue
        chosen = (raw or original or candidate).strip()
        key = _canonical_skill_key(chosen)
        if not key:
            continue
        current = normalized.get(key)
        if current is None or len(chosen) < len(current):
            normalized[key] = chosen

    filtered_skills = filter_skills_via_llm(list(normalized.values()))

    deduped: Dict[str, str] = {}
    for skill in filtered_skills:
        key = _canonical_skill_key(skill)
        if not key:
            continue
        current = deduped.get(key)
        if current is None or len(skill) < len(current):
            deduped[key] = skill

    return list(deduped.values())


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
        skills = sorted(_detect_skills(lower_sentence))
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
    direct_skills: List[str] = []
    direct_years: Optional[float] = None
    direct_knowledge: List[Dict[str, str]] = []
    direct_quant_suggestions: List[str] = []
    direct_quant_highlights: List[str] = []
    direct_leadership_signals: List[str] = []
    summary_text: str

    if isinstance(summary_payload, dict):
        summary_text = _normalize_text(summary_payload.get("summary_text"))
        for collection_key in ("core_skills", "tooling", "domain_experience"):
            for value in summary_payload.get(collection_key, []) or []:
                if isinstance(value, str) and value.strip():
                    direct_skills.append(value.strip())
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
                    skills_hit = _detect_skills(statement)
                    direct_knowledge.append(
                        {
                            "level": "unspecified",
                            "excerpt": statement.strip(),
                            "skills": sorted(skills_hit, key=lambda val: val.lower()),
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

    detected_skills = _detect_skills(summary_text)
    years_of_experience = _extract_years_of_experience(summary_text)
    if direct_years is not None:
        years_of_experience = direct_years

    candidate_skills: List[str] = []
    candidate_skills.extend(detected_skills)
    candidate_skills.extend([skill for skill in direct_skills if skill])
    candidate_skills = filter_skills_via_llm(candidate_skills)

    deduped: Dict[str, str] = {}
    for skill in candidate_skills:
        key = _canonical_skill_key(skill)
        if not key:
            continue
        current = deduped.get(key)
        if current is None or len(skill) < len(current):
            deduped[key] = skill

    skills_list = sorted(deduped.values(), key=lambda value: value.lower())

    classification = classify_skills(skills_list)
    if any(classification.values()):
        skills_by_category = {
            category: values for category, values in classification.items() if values
        }
    else:
        fallback: Dict[str, List[str]] = {}
        for category, keywords in constants.SKILL_CATEGORIES.items():
            hits = [
                skill
                for skill in skills_list
                if _canonical_skill_key(skill) in {
                    _canonical_skill_key(keyword) for keyword in keywords
                }
            ]
            if hits:
                fallback[category] = sorted(set(hits), key=lambda val: val.lower())
        skills_by_category = fallback

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
