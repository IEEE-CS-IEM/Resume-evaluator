import re
from functools import lru_cache
from typing import Dict, List, Tuple

from util.embedding_models import batched_query_similarity, embed_sentences

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

_CATEGORY_DESCRIPTORS = {
    "programming_languages": [
        "programming language",
        "scripting language",
        "strong coding skills",
    ],
    "frameworks": [
        "software framework",
        "web framework",
        "backend framework",
    ],
    "libraries_packages": [
        "software library",
        "python package",
        "javascript library",
    ],
    "cloud_platforms": [
        "cloud platform",
        "cloud service provider",
        "infrastructure as a service",
    ],
    "developer_tools": [
        "developer tool",
        "version control",
        "productivity tooling",
    ],
    "ml_ai_tools": [
        "machine learning toolkit",
        "ai platform",
        "ml framework",
    ],
    "design_tools": [
        "design tool",
        "ui design software",
        "ux prototyping tool",
    ],
    "devops_infra": [
        "devops platform",
        "infrastructure automation tool",
        "ci cd pipeline",
    ],
    "databases": [
        "database technology",
        "data storage system",
        "sql or nosql database",
    ],
    "frontend_tooling": [
        "frontend tooling",
        "javascript build system",
        "css framework",
    ],
    "mobile_tooling": [
        "mobile development toolkit",
        "ios android framework",
        "mobile app platform",
    ],
    "other": [
        "general technical skill",
    ],
}
_SIMILARITY_THRESHOLD = 0.28


def _canonical_skill_key(value: str) -> str:
    lowered = value.lower()
    lowered = re.sub(r"\(.*?\)", " ", lowered)
    lowered = re.sub(r"[^a-z0-9+#]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


def _category_descriptor_texts() -> Tuple[List[str], Dict[str, Tuple[int, int]]]:
    texts: List[str] = []
    spans: Dict[str, Tuple[int, int]] = {}
    for category in _CATEGORIES:
        descriptors = _CATEGORY_DESCRIPTORS.get(category, []) or ["technical skill"]
        start = len(texts)
        texts.extend(descriptors)
        spans[category] = (start, len(texts))
    return texts, spans


@lru_cache(maxsize=1)
def _descriptor_texts_and_spans() -> Tuple[List[str], Dict[str, Tuple[int, int]]]:
    return _category_descriptor_texts()


def _initialise_result() -> Dict[str, List[str]]:
    return {category: [] for category in _CATEGORIES}


def classify_skills(skills: List[str]) -> Dict[str, List[str]]:
    if not skills:
        return _initialise_result()

    result = _initialise_result()
    canonical_map = {
        category: {_canonical_skill_key(value) for value in descriptors}
        for category, descriptors in _CATEGORY_DESCRIPTORS.items()
    }

    # Direct classification using canonical keywords.
    unresolved_indices: List[int] = []
    cleaned_skills: List[str] = []
    for idx, skill in enumerate(skills):
        cleaned = skill.strip()
        if not cleaned:
            continue
        cleaned_skills.append(cleaned)
        canonical_key = _canonical_skill_key(cleaned)
        matched = False
        for category, keyset in canonical_map.items():
            if canonical_key in keyset:
                result[category].append(cleaned)
                matched = True
        if not matched:
            unresolved_indices.append(len(cleaned_skills) - 1)

    if not unresolved_indices:
        return result

    descriptor_texts, spans = _descriptor_texts_and_spans()
    try:
        skill_embeddings = embed_sentences([cleaned_skills[idx] for idx in unresolved_indices])
        scores_matrix = batched_query_similarity(skill_embeddings, descriptor_texts)
    except (RuntimeError, OSError):
        # If the model cannot load, fall back to marking unresolved skills as "other".
        for idx in unresolved_indices:
            result["other"].append(cleaned_skills[idx])
        return result

    for col, skill_idx in enumerate(unresolved_indices):
        skill = cleaned_skills[skill_idx]
        best_category = None
        best_score = 0.0
        for category, (start, end) in spans.items():
            category_score = scores_matrix[start:end, col].max()
            if category_score > best_score:
                best_score = category_score
                best_category = category
        if best_category and best_score >= _SIMILARITY_THRESHOLD:
            result[best_category].append(skill)
        else:
            result["other"].append(skill)

    return result
