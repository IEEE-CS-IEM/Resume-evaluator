import re
from functools import lru_cache
from typing import List

import yake

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "as",
    "be",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "our",
    "so",
    "that",
    "the",
    "their",
    "this",
    "to",
    "with",
}

_BLACKLIST_TERMS = {
    "responsibilities",
    "responsibility",
    "located",
    "location",
    "mission",
    "team",
    "bengaluru",
    "bengalore",
    "india",
    "company",
    "candidates",
    "culture",
    "customers",
    "users",
    "people",
    "impact",
    "experience",
    "opportunity",
    "developer",
    "development",
    "application",
    "applications",
    "teams",
    "team",
    "products",
    "product",
    "solutions",
    "clients",
    "building",
    "cutting-edge",
    "cutting",
    "edge",
    "mobile",
    "devices",
    "mobile devices",
    "global",
    "connect",
    "connecting",
    "growing",
}


def _is_plausible_keyword(keyword: str) -> bool:
    tokens = [token for token in re.split(r"[^\w+#./\-]+", keyword) if token]
    if not tokens:
        return False
    if any(token in _BLACKLIST_TERMS for token in tokens):
        return False
    meaningful_tokens = [token for token in tokens if token.lower() not in _STOPWORDS]
    if not meaningful_tokens:
        return False
    if len(tokens) > 4:
        return False
    return True


@lru_cache(maxsize=1)
def _get_extractor() -> yake.KeywordExtractor:
    return yake.KeywordExtractor(
        lan="en",
        n=3,
        dedupLim=0.9,
        top=100,
        features=None,
    )


def extract_keywords(text: str, max_keywords: int = 50) -> List[str]:
    if not text or not text.strip():
        return []
    extractor = _get_extractor()
    raw_keywords = extractor.extract_keywords(text)
    ordered = sorted(raw_keywords, key=lambda item: item[1])
    unique: List[str] = []
    seen = set()
    for keyword, _score in ordered:
        cleaned = keyword.strip().lower()
        if not cleaned or cleaned in seen:
            continue
        if not _is_plausible_keyword(cleaned):
            continue
        unique.append(cleaned)
        seen.add(cleaned)
        if len(unique) >= max_keywords:
            break
    return unique
