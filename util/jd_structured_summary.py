import json
import re
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from util import constants
from util.embedding_models import batched_query_similarity, embed_sentences
from util.keyword_extractor import extract_keywords
from util.llm_helpers import coerce_json
from util.skill_guard import filter_skills_via_llm
from util.simpleagent import MyAgent, ProviderCapacityError
from util.system_prompt import (
    prompt_jd_domains,
    prompt_jd_role,
    prompt_jd_role_keywords,
    prompt_jd_seniority,
)

SUMMARY_QUERIES = [
    "role overview",
    "job responsibilities",
    "position summary",
]

KNOWLEDGE_QUERIES = [
    "knowledge of",
    "familiar with",
    "experience with",
]

def _split_sentences(text: str) -> List[str]:
    raw_segments = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences = [segment.strip() for segment in raw_segments if segment.strip()]
    return sentences or [text.strip()] if text.strip() else []


def _ensure_embeddings(sentences: Sequence[str]) -> np.ndarray:
    if not sentences:
        return np.empty((0, 0))
    return embed_sentences(sentences)


def _rank_sentences(
    sentences: Sequence[str],
    embeddings: np.ndarray,
    queries: Sequence[str],
    top_k: int,
    min_score: float,
) -> List[str]:
    if not sentences:
        return []
    scores_matrix = batched_query_similarity(embeddings, list(queries))
    if scores_matrix.size == 0:
        return []
    aggregate = scores_matrix.max(axis=0)
    ranked_indices = np.argsort(-aggregate)

    chosen: List[int] = []
    for idx in ranked_indices:
        if aggregate[idx] < min_score:
            continue
        chosen.append(idx)
        if len(chosen) >= top_k:
            break

    if not chosen:
        return []

    chosen_sorted = sorted(chosen)
    return [sentences[idx] for idx in chosen_sorted]


def _canonical_skill_key(value: str) -> str:
    lowered = value.lower()
    lowered = re.sub(r"\(.*?\)", " ", lowered)
    lowered = re.sub(r"[^a-z0-9+#]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _keyword_present(text: str, keyword: str) -> bool:
    pattern = rf"(?<!\w){re.escape(keyword)}(?!\w)"
    return re.search(pattern, text, re.IGNORECASE) is not None


def _extract_skills(text: str) -> List[str]:
    extracted = set(extract_keywords(text))
    for keywords in constants.SKILL_CATEGORIES.values():
        for keyword in keywords:
            if _keyword_present(text, keyword):
                extracted.add(keyword)
    cleaned = [skill.strip() for skill in extracted if skill.strip()]
    return filter_skills_via_llm(cleaned)


def _categorise_skills(skills: List[str]) -> Dict[str, List[str]]:
    category_hits: Dict[str, List[str]] = {}
    for category, keywords in constants.SKILL_CATEGORIES.items():
        matches = []
        keyword_keys = {_canonical_skill_key(keyword) for keyword in keywords}
        for skill in skills:
            if _canonical_skill_key(skill) in keyword_keys:
                matches.append(skill)
        if matches:
            category_hits[category] = sorted(set(matches), key=lambda value: value.lower())
    return category_hits


def _extract_years(text: str) -> Tuple[Optional[float], Optional[float]]:
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)", text.lower())
    if not matches:
        return (None, None)
    try:
        values = sorted(float(match) for match in matches)
    except ValueError:
        return (None, None)
    return (values[0], values[-1])


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


@lru_cache(maxsize=1)
def _get_role_agent() -> MyAgent:
    return MyAgent(system_prompt=prompt_jd_role)


@lru_cache(maxsize=1)
def _get_role_keyword_agent() -> MyAgent:
    return MyAgent(system_prompt=prompt_jd_role_keywords)


def _role_keywords_from_llm(text: str) -> Tuple[str, ...]:
    agent = _get_role_keyword_agent()
    payload = json.dumps({"job_description": text})
    try:
        response = agent(message=payload, temperature=0.0)
    except ProviderCapacityError:
        return ()

    parsed, raw_text = coerce_json(response)
    keywords: List[str] = []
    seen = set()

    def _add_candidate(value: str) -> None:
        normalized = value.strip().lower()
        if not normalized:
            return
        if not re.search(r"[a-z]", normalized):
            return
        if normalized not in seen:
            seen.add(normalized)
            keywords.append(normalized)

    def _extract_candidates(payload: Any) -> None:
        if isinstance(payload, dict):
            block = payload.get("role_keywords") or payload.get("keywords")
            if isinstance(block, list):
                for entry in block:
                    _extract_candidates(entry)
            elif isinstance(block, str):
                for part in re.split(r"[,\n/]+", block):
                    _add_candidate(part)
        elif isinstance(payload, list):
            for entry in payload:
                _extract_candidates(entry)
        elif isinstance(payload, str):
            for part in re.split(r"[,\n/]+", payload):
                _add_candidate(part)

    _extract_candidates(parsed)
    if not keywords and isinstance(raw_text, str):
        _extract_candidates(raw_text)

    if len(keywords) > 32:
        keywords[:] = keywords[:32]

    return tuple(keywords)


def _role_title_from_llm(text: str) -> Optional[str]:
    agent = _get_role_agent()
    payload = json.dumps({"job_description": text})
    try:
        response = agent(message=payload, temperature=0.0)
    except ProviderCapacityError:
        return None
    parsed, raw_text = coerce_json(response)
    candidate: Optional[str] = None
    if isinstance(parsed, dict):
        value = parsed.get("role_title")
        if isinstance(value, str):
            candidate = value.strip()
    elif isinstance(parsed, str):
        candidate = parsed.strip()
    else:
        candidate = raw_text.strip() if isinstance(raw_text, str) else None
    return candidate or None


def _fallback_role_title(text: str) -> Optional[str]:
    role_keywords = _role_keywords_from_llm(text)
    if role_keywords:
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            lowered = stripped.lower()
            if any(keyword in lowered for keyword in role_keywords) and len(stripped.split()) <= 10:
                return stripped
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return None


@lru_cache(maxsize=1)
def _get_seniority_agent() -> MyAgent:
    return MyAgent(system_prompt=prompt_jd_seniority)


def _normalise_seniority(value: str) -> Optional[str]:
    mapping = {
        "entry": "entry",
        "entry-level": "entry",
        "entry level": "entry",
        "junior": "junior",
        "mid": "mid-level",
        "mid-level": "mid-level",
        "mid level": "mid-level",
        "midlevel": "mid-level",
        "senior": "senior",
        "principal": "principal",
        "mixed": "mixed",
        "unspecified": "unspecified",
    }
    key = value.strip().lower()
    return mapping.get(key)


def _seniority_from_llm(text: str, role_title: Optional[str]) -> Optional[str]:
    agent = _get_seniority_agent()
    payload_dict = {
        "job_description": text,
        "role_title": role_title,
    }
    payload = json.dumps(payload_dict)
    try:
        response = agent(message=payload, temperature=0.0)
    except ProviderCapacityError:
        return None
    parsed, raw_text = coerce_json(response)
    candidate: Optional[str] = None
    if isinstance(parsed, dict):
        value = parsed.get("seniority_level")
        if isinstance(value, str):
            candidate = _normalise_seniority(value)
    elif isinstance(parsed, str):
        candidate = _normalise_seniority(parsed)
    else:
        if isinstance(raw_text, str):
            candidate = _normalise_seniority(raw_text)
    return candidate


def _seniority_from_title(role_title: Optional[str]) -> Optional[str]:
    if not role_title:
        return None
    lowered = role_title.lower()
    if any(keyword in lowered for keyword in ("intern", "graduate", "entry")):
        return "entry"
    if any(keyword in lowered for keyword in ("junior", "associate")):
        return "junior"
    if any(keyword in lowered for keyword in ("mid", "mid-level", "experienced")):
        return "mid-level"
    if any(keyword in lowered for keyword in ("senior", "lead", "staff")):
        return "senior"
    if any(keyword in lowered for keyword in ("principal", "distinguished")):
        return "principal"
    return None


@lru_cache(maxsize=1)
def _get_domain_agent() -> MyAgent:
    return MyAgent(system_prompt=prompt_jd_domains)


def _domains_from_llm(text: str) -> List[str]:
    agent = _get_domain_agent()
    payload = json.dumps({"job_description": text})
    try:
        response = agent(message=payload, temperature=0.0)
    except ProviderCapacityError:
        return []
    parsed, raw_text = coerce_json(response)
    domains: List[str] = []
    if isinstance(parsed, dict):
        values = parsed.get("domains")
        if isinstance(values, list):
            for entry in values:
                if isinstance(entry, str):
                    candidate = entry.strip()
                    if candidate and candidate.lower() not in {d.lower() for d in domains}:
                        domains.append(candidate)
    elif isinstance(parsed, list):
        for entry in parsed:
            if isinstance(entry, str):
                candidate = entry.strip()
                if candidate and candidate.lower() not in {d.lower() for d in domains}:
                    domains.append(candidate)
    elif isinstance(parsed, str):
        candidate = parsed.strip()
        if candidate:
            domains.append(candidate)
    elif isinstance(raw_text, str):
        candidate = raw_text.strip()
        if candidate:
            domains.append(candidate)
    return domains


def _knowledge_requirements(
    sentences: Sequence[str],
    embeddings: np.ndarray,
) -> List[Dict[str, Any]]:
    statements = _rank_sentences(
        sentences=sentences,
        embeddings=embeddings,
        queries=KNOWLEDGE_QUERIES,
        top_k=8,
        min_score=0.35,
    )
    structured: List[Dict[str, Any]] = []
    for statement in statements:
        structured.append(
            {
                "text": statement,
                "skills": [],
                "categories": {},
            }
        )
    return structured


def _partition_skill_lists(sentences: Iterable[str]) -> Tuple[List[str], List[str]]:
    must_have: List[str] = []
    nice_to_have: List[str] = []
    for sentence in sentences:
        lower = sentence.lower()
        skills = _extract_skills(sentence)
        if not skills:
            continue
        if any(phrase in lower for phrase in ("must have", "required", "mandate", "strong experience")):
            must_have.extend(skills)
        elif any(phrase in lower for phrase in ("nice to have", "good to have", "bonus", "plus")):
            nice_to_have.extend(skills)
        elif "tool" in lower or "technology" in lower:
            must_have.extend(skills)
    return (filter_skills_via_llm(must_have), filter_skills_via_llm(nice_to_have))


def _dedupe(sequence: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in sequence:
        lowered = item.lower()
        if lowered not in seen:
            seen.add(lowered)
            ordered.append(item)
    return ordered


def summarise_job_description(jd_text: str, temperature: float = 0.0) -> Dict[str, Any]:
    """
    Produce a structured summary of the job description using a BERT encoder
    instead of relying on an external LLM.
    """
    sentences = _split_sentences(jd_text)
    embeddings = _ensure_embeddings(sentences)

    summary_sentences = _rank_sentences(
        sentences=sentences,
        embeddings=embeddings,
        queries=SUMMARY_QUERIES,
        top_k=5,
        min_score=0.25,
    )
    if not summary_sentences:
        summary_sentences = sentences[:5]
    summary_text = " ".join(summary_sentences).strip()

    skills = _extract_skills(jd_text)
    must_have_skills, nice_to_have_skills = _partition_skill_lists(sentences)
    tooling_keywords = {
        _canonical_skill_key(keyword)
        for keyword in constants.SKILL_CATEGORIES.get("frameworks", set())
        | constants.SKILL_CATEGORIES.get("devops_tools", set())
        | constants.SKILL_CATEGORIES.get("cloud_platforms", set())
    }
    tooling = _dedupe(
        skill for skill in skills if _canonical_skill_key(skill) in tooling_keywords
    )
    domains = _domains_from_llm(jd_text)
    knowledge_requirements = _knowledge_requirements(sentences, embeddings)
    min_years, max_years = _extract_years(jd_text)
    required_years = max_years or min_years
    fallback_role = _fallback_role_title(jd_text)
    role_title = _role_title_from_llm(jd_text) or fallback_role
    seniority_level = (
        _seniority_from_llm(jd_text, role_title)
        or _seniority_from_title(role_title)
        or "unspecified"
    )

    return {
        "raw_text": jd_text,
        "role_title": role_title,
        "seniority_level": seniority_level,
        "llm_summary": summary_text,
        "summary": summary_text,
        "skills": skills,
        "skills_by_category": _categorise_skills(skills),
        "must_have_skills": must_have_skills,
        "nice_to_have_skills": nice_to_have_skills,
        "tooling": tooling,
        "domains": domains,
        "required_years": required_years,
        "required_experience_band": _experience_band(required_years),
        "knowledge_requirements": knowledge_requirements,
        "experience_requirement": {
            "minimum_years": min_years,
            "maximum_years": max_years,
            "described_range": summary_text,
        },
    }
