import os
import re
from typing import Dict, List, Optional, Tuple, Union

import docx
import pdfplumber

from util import constants
from util.resume_summary_analyzer import analyze_resume_summary


def extract_text_from_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        with pdfplumber.open(file_path) as pdf:
            return "\n".join(
                page.extract_text() for page in pdf.pages if page.extract_text()
            )
    if ext in [".docx", ".doc"]:
        doc = docx.Document(file_path)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)
    if ext in [".txt"]:
        with open(file_path, "r", encoding="utf-8") as handle:
            return handle.read()
    raise ValueError(f"Unsupported file type: {ext}")


def _normalize(text: Optional[str]) -> str:
    return text or ""


def _detect_skills(text: str) -> Tuple[List[str], Dict[str, List[str]]]:
    lower_text = text.lower()
    hits: Dict[str, List[str]] = {}
    flat_skills: set[str] = set()

    for category, keywords in constants.SKILL_CATEGORIES.items():
        matched = sorted({keyword for keyword in keywords if keyword in lower_text})
        if matched:
            hits[category] = matched
            flat_skills.update(matched)
    return sorted(flat_skills), hits


def _extract_required_experience(text: str) -> Optional[float]:
    pattern = re.findall(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)", text.lower())
    if not pattern:
        return None
    try:
        return max(float(match) for match in pattern)
    except ValueError:
        return None


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


def _extract_knowledge_requirements(text: str) -> List[Dict[str, object]]:
    sentences = re.split(r"(?:\.\s+|\n+)", text)
    keywords = (
        "familiar",
        "experience with",
        "knowledge of",
        "exposure to",
        "proficient in",
    )
    requirements: List[Dict[str, object]] = []
    for sentence in sentences:
        stripped = sentence.strip()
        if not stripped:
            continue
        lower_sentence = stripped.lower()
        if not any(keyword in lower_sentence for keyword in keywords):
            continue
        skills, skills_by_category = _detect_skills(lower_sentence)
        requirements.append(
            {
                "text": stripped,
                "skills": skills,
                "categories": skills_by_category,
            }
        )
    return requirements


def _skill_overlap(
    jd_skills: List[str], resume_skills: List[str]
) -> Tuple[List[str], List[str]]:
    jd_set = set(jd_skills)
    resume_set = set(resume_skills)
    return sorted(jd_set & resume_set), sorted(jd_set - resume_set)


def _category_overlap(
    jd_categories: Dict[str, List[str]], resume_categories: Dict[str, List[str]]
) -> Dict[str, Dict[str, List[str]]]:
    breakdown: Dict[str, Dict[str, List[str]]] = {}
    for category, jd_items in jd_categories.items():
        resume_items = set(resume_categories.get(category, []))
        jd_items_set = set(jd_items)
        breakdown[category] = {
            "matched": sorted(jd_items_set & resume_items),
            "missing": sorted(jd_items_set - resume_items),
        }
    return breakdown


def _calculate_keyword_score(
    matched_skills: List[str],
    total_required_skills: int,
    experience_alignment: float,
    knowledge_alignment: float,
    category_coverage: float,
) -> float:
    skill_score = (
        len(matched_skills) / total_required_skills if total_required_skills else 0.0
    )
    keyword_score = (
        constants.SCORING_WEIGHTS["skill_match"] * skill_score
        + constants.SCORING_WEIGHTS["experience_alignment"] * experience_alignment
        + constants.SCORING_WEIGHTS["knowledge_level"] * knowledge_alignment
        + constants.SCORING_WEIGHTS.get("category_coverage", 0.0) * category_coverage
    )
    return round(keyword_score * 100, 2)


def _experience_alignment(required: Optional[float], candidate: Optional[float]) -> float:
    if required is None or candidate is None:
        # Neutral stance when either side is unspecified.
        return 0.5
    if candidate >= required:
        return 1.0
    return max(0.0, candidate / required)


def _knowledge_alignment(
    knowledge_requirements: List[Dict[str, object]],
    knowledge_mentions: List[Dict[str, object]],
    resume_skills: List[str],
) -> float:
    if not knowledge_requirements:
        return 1.0
    coverage = 0
    resume_skill_set = set(resume_skills)
    mention_skill_set = set()
    for mention in knowledge_mentions:
        mention_skill_set.update(mention.get("skills", []))
        excerpt = mention.get("excerpt", "").lower()
        for category_keywords in constants.SKILL_CATEGORIES.values():
            for keyword in category_keywords:
                if keyword in excerpt:
                    mention_skill_set.add(keyword)
    for requirement in knowledge_requirements:
        requirement_skills = set(requirement.get("skills", []))
        if not requirement_skills:
            # fallback: look for any overlap between requirement text and skills mentioned
            requirement_skills = {
                keyword
                for keyword_set in constants.SKILL_CATEGORIES.values()
                for keyword in keyword_set
                if keyword in requirement.get("text", "").lower()
            }
        if requirement_skills & resume_skill_set:
            coverage += 1
            continue
        if requirement_skills & mention_skill_set:
            coverage += 1
    score = coverage / len(knowledge_requirements)
    if score == 0 and resume_skills:
        # provide a small base score when the resume exhibits relevant skills,
        # but phrasing does not match the JD requirement sentences verbatim.
        return 0.2
    return score


def _final_ats_score(keyword_score: float, llm_score: Optional[float]) -> float:
    if llm_score is None:
        return keyword_score
    # Blend keyword score (objective) with LLM score (subjective narrative).
    final_score = 0.6 * keyword_score + 0.4 * llm_score
    return round(final_score, 2)


def _experience_relevance_statement(
    required: Optional[float], candidate: Optional[float]
) -> str:
    if required is None and candidate is None:
        return "Experience level not specified in both resume and job description."
    if required is None:
        return f"Job description does not specify required years; candidate reports ~{candidate} years."
    if candidate is None:
        return f"Candidate experience not detected; role calls for ~{required} years."
    gap = candidate - required
    if gap >= 1.0:
        return f"Experience exceeds requirement by approximately {gap:.1f} years."
    if gap >= 0:
        return "Experience closely matches the requirement."
    return f"Experience shortfall of approximately {abs(gap):.1f} years."


def parse_jd(jd_text: str) -> Dict[str, object]:
    jd_text = _normalize(jd_text)
    skills, skills_by_category = _detect_skills(jd_text)
    required_years = _extract_required_experience(jd_text)
    knowledge_requirements = _extract_knowledge_requirements(jd_text)

    return {
        "raw_text": jd_text,
        "skills": skills,
        "skills_by_category": skills_by_category,
        "required_years": required_years,
        "required_experience_band": _experience_band(required_years),
        "knowledge_requirements": knowledge_requirements,
    }


def process_jd_and_resume(
    jd_input: Union[str, os.PathLike],
    resume_data: Union[str, Dict[str, object]],
) -> Dict[str, object]:
    """
    jd_input: Either a raw job description string or a file path.
    resume_data: Either a raw LLM summary string or the structured output from
                 resume_summary_analyzer.analyze_resume_summary.
    """
    # Step 1: resolve texts.
    jd_text = jd_input
    if isinstance(jd_input, (str, os.PathLike)) and os.path.exists(str(jd_input)):
        jd_text = extract_text_from_file(str(jd_input))

    if isinstance(jd_text, dict):
        jd_struct = jd_text
    else:
        jd_struct = parse_jd(jd_text)

    if isinstance(resume_data, str):
        resume_struct = analyze_resume_summary(resume_data)
    else:
        resume_struct = resume_data

    matched_skills, missing_skills = _skill_overlap(
        jd_struct["skills"], resume_struct.get("skills", [])
    )
    category_breakdown = _category_overlap(
        jd_struct["skills_by_category"],
        resume_struct.get("skills_by_category", {}),
    )
    if jd_struct["skills_by_category"]:
        matched_categories = sum(
            1 for data in category_breakdown.values() if data["matched"]
        )
        total_categories = len(jd_struct["skills_by_category"])
        category_coverage = matched_categories / total_categories
    else:
        category_coverage = 1.0

    experience_alignment = _experience_alignment(
        jd_struct["required_years"], resume_struct.get("years_experience")
    )
    knowledge_alignment = _knowledge_alignment(
        jd_struct["knowledge_requirements"],
        resume_struct.get("knowledge_mentions", []),
        resume_struct.get("skills", []),
    )

    keyword_score = _calculate_keyword_score(
        matched_skills,
        len(jd_struct["skills"]),
        experience_alignment,
        knowledge_alignment,
        category_coverage,
    )
    llm_score = resume_struct.get("llm_score", 0.0)
    final_score = _final_ats_score(keyword_score, llm_score)
    llm_score_available = bool(resume_struct.get("llm_score_available"))

    return {
        "jd_insights": {
            "role_title": jd_struct.get("role_title"),
            "seniority_level": jd_struct.get("seniority_level"),
            "summary": jd_struct.get("summary") or jd_struct.get("llm_summary"),
            "domains": jd_struct.get("domains", []),
            "required_skills": jd_struct["skills"],
            "skills_by_category": jd_struct["skills_by_category"],
            "required_years": jd_struct["required_years"],
            "experience_band": jd_struct["required_experience_band"],
            "knowledge_requirements": jd_struct["knowledge_requirements"],
        },
        "resume_insights": {
            "skills": resume_struct.get("skills", []),
            "skills_by_category": resume_struct.get("skills_by_category", {}),
            "years_experience": resume_struct.get("years_experience"),
            "experience_band": resume_struct.get("experience_band"),
            "knowledge_mentions": resume_struct.get("knowledge_mentions", []),
            "quantification_suggestions": resume_struct.get(
                "quantification_suggestions", []
            ),
            "llm_score": llm_score,
            "llm_score_available": llm_score_available,
        },
        "matching_summary": {
            "matched_skills": matched_skills,
            "missing_skills": missing_skills,
            "category_breakdown": category_breakdown,
            "experience_alignment": round(experience_alignment, 2),
            "knowledge_alignment": round(knowledge_alignment, 2),
            "category_coverage": round(category_coverage, 2),
            "experience_relevance": _experience_relevance_statement(
                jd_struct["required_years"],
                resume_struct.get("years_experience"),
            ),
        },
        "scores": {
            "keyword_score": keyword_score,
            "llm_score": llm_score,
            "llm_score_used": llm_score_available,
            "final_ats_score": final_score,
            "subscores": {
                "skill_coverage": round(
                    len(matched_skills) / len(jd_struct["skills"]), 2
                )
                if jd_struct["skills"]
                else 1.0,
                "experience_alignment": round(experience_alignment, 2),
                "knowledge_alignment": round(knowledge_alignment, 2),
                "category_coverage": round(category_coverage, 2),
            },
        },
        "recommendations": {
            "quantification": resume_struct.get("quantification_suggestions", []),
            "next_skills_to_learn": missing_skills[:5],
        },
    }
