import os
import re
from typing import Dict, List, Optional, Tuple, Union

import docx
import pdfplumber
import pytesseract

from util import constants
from util.resume_summary_analyzer import analyze_resume_summary, _canonical_skill_key
from util.keyword_extractor import extract_keywords
from util.skill_guard import filter_skills_via_llm


def extract_text_from_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        pages_text: List[str] = []
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                if text.strip():
                    pages_text.append(text)
                    continue
                try:
                    pil_image = page.to_image(resolution=300).original
                    if pil_image.mode not in ("RGB", "RGBA"):
                        pil_image = pil_image.convert("RGB")
                    ocr_text = pytesseract.image_to_string(pil_image)
                    if ocr_text.strip():
                        pages_text.append(ocr_text)
                except Exception:
                    continue
        return "\n".join(segment for segment in pages_text if segment.strip())
    if ext in [".docx", ".doc"]:
        doc = docx.Document(file_path)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)
    if ext in [".txt"]:
        with open(file_path, "r", encoding="utf-8") as handle:
            return handle.read()
    raise ValueError(f"Unsupported file type: {ext}")


def _normalize(text: Optional[str]) -> str:
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


def _detect_skills(text: str) -> Tuple[List[str], Dict[str, List[str]]]:
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

    final_skills = list(deduped.values())

    category_hits: Dict[str, List[str]] = {}
    for category, keywords in constants.SKILL_CATEGORIES.items():
        matches = []
        for keyword in keywords:
            key = _canonical_skill_key(keyword)
            if key in deduped:
                matches.append(deduped[key])
        if matches:
            category_hits[category] = sorted(set(matches))

    return final_skills, category_hits


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
        if not any(_keyword_present(lower_sentence, phrase.lower()) for phrase in keywords):
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
    jd_map = {
        _canonical_skill_key(skill): skill
        for skill in jd_skills
        if isinstance(skill, str) and _canonical_skill_key(skill)
    }
    resume_keys = {
        _canonical_skill_key(skill)
        for skill in resume_skills
        if isinstance(skill, str) and _canonical_skill_key(skill)
    }

    matched_keys = sorted(jd_map.keys() & resume_keys)
    missing_keys = sorted(jd_map.keys() - resume_keys)

    matched = [jd_map[key] for key in matched_keys]
    missing = [jd_map[key] for key in missing_keys]
    return matched, missing


def _category_overlap(
    jd_categories: Dict[str, List[str]], resume_categories: Dict[str, List[str]]
) -> Dict[str, Dict[str, List[str]]]:
    breakdown: Dict[str, Dict[str, List[str]]] = {}
    for category, jd_items in jd_categories.items():
        resume_items = {
            _canonical_skill_key(item)
            for item in resume_categories.get(category, [])
            if isinstance(item, str) and _canonical_skill_key(item)
        }
        jd_items_set = {
            _canonical_skill_key(item)
            for item in jd_items
            if isinstance(item, str) and _canonical_skill_key(item)
        }
        breakdown[category] = {
            "matched": sorted(
                item
                for item in jd_items
                if _canonical_skill_key(item) in resume_items
            ),
            "missing": sorted(
                item
                for item in jd_items
                if _canonical_skill_key(item) not in resume_items
            ),
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
                if _keyword_present(excerpt, keyword.lower()):
                    mention_skill_set.add(keyword)
    for requirement in knowledge_requirements:
        requirement_skills = set(requirement.get("skills", []))
        if not requirement_skills:
            # fallback: look for any overlap between requirement text and skills mentioned
            requirement_text = requirement.get("text", "").lower()
            requirement_skills = {
                keyword
                for keyword_set in constants.SKILL_CATEGORIES.values()
                for keyword in keyword_set
                if _keyword_present(requirement_text, keyword.lower())
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


def _infer_role_title(jd_text: str) -> Optional[str]:
    def _clean_role_title(line: str) -> str:
        trimmed = line.strip()
        if " - " in trimmed:
            left, right = trimmed.split(" - ", 1)
            if not re.search(
                r"(?i)\b(engineer|developer|designer|manager|analyst|scientist|specialist)\b",
                right,
            ):
                return left.strip()
        return trimmed

    lines = [line.strip(" :-\u2022") for line in jd_text.splitlines()]
    cleaned_lines = [line for line in lines if line.strip()]

    for line in cleaned_lines:
        if re.search(r"(?i)\bresponsibilities\b", line):
            candidate = re.split(r"(?i)\bresponsibilities\b", line)[0].strip(" :-")
            if candidate:
                return _clean_role_title(candidate)

    for line in cleaned_lines:
        if re.search(r"(?i)\b(engineer|developer|designer|manager|analyst|scientist|specialist)\b", line):
            return _clean_role_title(line)

    return None


def _infer_seniority(role_title: Optional[str], jd_text: str) -> str:
    text_blob = " ".join(filter(None, [role_title, jd_text])).lower()
    seniority_map = [
        ("principal", ("principal", "distinguished")),
        ("senior", ("senior", "lead", "staff")),
        ("mid-level", ("mid level", "mid-level", "intermediate", "experienced")),
        ("junior", ("junior", "associate")),
        ("entry", ("entry level", "entry-level", "graduate", "university grad", "freshers", "new grad", "intern")),
    ]

    for label, keywords in seniority_map:
        if any(keyword in text_blob for keyword in keywords):
            return label

    return "unspecified"


def _summarise_text(jd_text: str, max_sentences: int = 3) -> str:
    sentences = [segment.strip() for segment in re.split(r"[.\n]", jd_text) if segment.strip()]
    selected = sentences[:max_sentences]
    return ". ".join(selected).strip()


def parse_jd(jd_text: str) -> Dict[str, object]:
    jd_text = _normalize(jd_text)
    skills, skills_by_category = _detect_skills(jd_text)
    required_years = _extract_required_experience(jd_text)
    knowledge_requirements = _extract_knowledge_requirements(jd_text)
    role_title = _infer_role_title(jd_text)
    seniority_level = _infer_seniority(role_title, jd_text)
    summary = _summarise_text(jd_text)

    return {
        "raw_text": jd_text,
        "role_title": role_title,
        "seniority_level": seniority_level,
        "summary": summary,
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
