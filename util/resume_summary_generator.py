import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np

from util import constants
from util.embedding_models import batched_query_similarity, embed_sentences
from util.keyword_extractor import extract_keywords
from util.skill_guard import filter_skills_via_llm

SUMMARY_QUERIES = [
    "professional summary",
    "career overview",
    "high level accomplishments",
]

KNOWLEDGE_QUERIES = [
    "knowledge of technologies",
    "familiar with",
    "experience with",
]

DOMAIN_HINTS = {
    "finance",
    "financial services",
    "banking",
    "healthcare",
    "pharmaceutical",
    "insurance",
    "ecommerce",
    "retail",
    "logistics",
    "supply chain",
    "manufacturing",
    "automotive",
    "education",
    "edtech",
    "gaming",
    "media",
    "telecom",
    "telecommunications",
    "travel",
    "hospitality",
}


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


def _keyword_present(text: str, keyword: str) -> bool:
    pattern = rf"(?<!\\w){re.escape(keyword)}(?!\\w)"
    return re.search(pattern, text, re.IGNORECASE) is not None


def _extract_skills(text: str) -> List[str]:
    extracted = set(extract_keywords(text))
    for keywords in constants.SKILL_CATEGORIES.values():
        for keyword in keywords:
            if _keyword_present(text, keyword):
                extracted.add(keyword)
    cleaned = [skill.strip() for skill in extracted if skill.strip()]
    return filter_skills_via_llm(cleaned)


def _select_domain_sentences(sentences: Iterable[str]) -> List[str]:
    selected: List[str] = []
    seen = set()
    for sentence in sentences:
        lower = sentence.lower()
        if any(hint in lower for hint in DOMAIN_HINTS):
            if sentence not in seen:
                selected.append(sentence)
                seen.add(sentence)
    return selected


def _extract_quantifiable(sentences: Iterable[str]) -> List[str]:
    highlights: List[str] = []
    for sentence in sentences:
        if re.search(r"\b\d+(?:\.\d+)?\s*(?:%|x|k|m|million|billion)?", sentence):
            highlights.append(sentence)
    return highlights


def _extract_leadership(sentences: Iterable[str]) -> List[str]:
    keywords = constants.EXPERIENCE_LEVEL_KEYWORDS.get("leadership", set())
    hits: List[str] = []
    for sentence in sentences:
        lower = sentence.lower()
        if any(keyword in lower for keyword in keywords):
            hits.append(sentence)
    return hits


def _extract_years_of_experience(text: str) -> Optional[float]:
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years|yrs)", text.lower())
    if not matches:
        return None
    try:
        values = [float(match) for match in matches]
    except ValueError:
        return None
    return max(values) if values else None


def _infer_seniority(text: str, years: Optional[float]) -> Optional[str]:
    normalized = text.lower()
    if "principal" in normalized or "distinguished" in normalized:
        return "principal"
    if any(tag in normalized for tag in ("staff", "lead", "senior")):
        return "senior"
    if any(tag in normalized for tag in ("mid level", "mid-level", "experienced")):
        return "mid-level"
    if any(tag in normalized for tag in ("junior", "associate")):
        return "junior"
    if any(tag in normalized for tag in ("entry level", "entry-level", "graduate", "intern")):
        return "entry"

    if years is None:
        return None
    if years >= 10:
        return "principal"
    if years >= 7:
        return "senior"
    if years >= 4:
        return "mid-level"
    if years >= 1:
        return "junior"
    return "entry"


def _knowledge_statements(sentences: Sequence[str], embeddings: np.ndarray) -> List[str]:
    statements = _rank_sentences(
        sentences=sentences,
        embeddings=embeddings,
        queries=KNOWLEDGE_QUERIES,
        top_k=6,
        min_score=0.35,
    )
    return statements


def _quantification_gaps(summary_text: str) -> List[str]:
    suggestions: List[str] = []
    segments = [segment.strip() for segment in re.split(r"[.\n]", summary_text) if segment.strip()]
    for sentence in segments:
        lower_sentence = sentence.lower()
        if any(verb in lower_sentence for verb in constants.QUANTIFIABLE_VERBS):
            if not re.search(r"\b\d+(?:\.\d+)?(?:%|x|k|m|million|billion)?\b", sentence):
                suggestions.append(f"Add a measurable outcome to: '{sentence[:120].strip()}'")
    return suggestions


def _collect_tooling(skills: List[str]) -> List[str]:
    tooling_hits: List[str] = []
    lookup = {
        keyword: category
        for category, keywords in constants.SKILL_CATEGORIES.items()
        for keyword in keywords
        if category in ("frameworks", "devops_tools", "data_platforms", "cloud_platforms")
    }
    for skill in skills:
        if lookup.get(skill.lower()):
            tooling_hits.append(skill)
    return sorted(set(tooling_hits), key=lambda val: val.lower())


def generate_resume_summary(resume_text: str, temperature: float = 0.0) -> Dict[str, Any]:
    """
    Summarise a resume using a BERT encoder to identify the most representative
    sentences and surface structured signals without calling an external LLM.
    """
    sentences = _split_sentences(resume_text)
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
    skills = _extract_skills(resume_text)
    tooling = _collect_tooling(skills)
    domain_sentences = _select_domain_sentences(sentences)
    quantifiable_highlights = _extract_quantifiable(sentences)
    leadership_experience = _extract_leadership(sentences)
    knowledge_statements = _knowledge_statements(sentences, embeddings)
    years_experience = _extract_years_of_experience(resume_text)
    seniority = _infer_seniority(resume_text, years_experience)
    quantification_suggestions = _quantification_gaps(summary_text)

    return {
        "summary_text": summary_text,
        "core_skills": skills,
        "tooling": tooling,
        "domain_experience": domain_sentences,
        "quantifiable_highlights": quantifiable_highlights,
        "leadership_experience": leadership_experience,
        "total_years_experience": years_experience,
        "seniority": seniority,
        "knowledge_statements": knowledge_statements,
        "quantification_suggestions": quantification_suggestions,
    }
