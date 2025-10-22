import json
from functools import lru_cache
from typing import Dict, Iterable, List, Sequence, Tuple

from util.llm_helpers import coerce_json
from util.simpleagent import MyAgent, ProviderCapacityError
from util.system_prompt import prompt_skill_guard

_MAX_CHUNK_SIZE = 20


def _chunk_sequence(sequence: Sequence[str], size: int) -> Iterable[Tuple[str, ...]]:
    for index in range(0, len(sequence), size):
        yield tuple(sequence[index : index + size])


@lru_cache(maxsize=1)
def _get_agent() -> MyAgent:
    return MyAgent(system_prompt=prompt_skill_guard)


def _extract_positive_skills(
    skills_tuple: Tuple[str, ...], parsed_response
) -> Tuple[str, ...]:
    if not skills_tuple:
        return ()

    positive_keys = set()

    def _normalise(value: str) -> str:
        return value.strip().lower()

    if isinstance(parsed_response, dict):
        skills_block = parsed_response.get("skills")
        if isinstance(skills_block, list):
            for entry in skills_block:
                if isinstance(entry, dict):
                    raw_value = (
                        entry.get("value")
                        or entry.get("skill")
                        or entry.get("name")
                        or entry.get("text")
                    )
                    if isinstance(raw_value, str):
                        value = _normalise(raw_value)
                        flag = entry.get("is_skill")
                        if isinstance(flag, bool):
                            if flag:
                                positive_keys.add(value)
                        elif flag is None:
                            positive_keys.add(value)
                elif isinstance(entry, str):
                    positive_keys.add(_normalise(entry))
        elif isinstance(skills_block, dict):
            for raw_value, flag in skills_block.items():
                value = _normalise(str(raw_value))
                if isinstance(flag, bool):
                    if flag:
                        positive_keys.add(value)
                else:
                    positive_keys.add(value)
    elif isinstance(parsed_response, list):
        for entry in parsed_response:
            if isinstance(entry, str):
                positive_keys.add(_normalise(entry))

    if not positive_keys:
        return skills_tuple

    ordered: List[str] = []
    seen = set()
    for original in skills_tuple:
        lowered = _normalise(original)
        if lowered in positive_keys and lowered not in seen:
            ordered.append(original)
            seen.add(lowered)
    return tuple(ordered)


@lru_cache(maxsize=512)
def _filter_chunk(skills_tuple: Tuple[str, ...]) -> Tuple[str, ...]:
    if not skills_tuple:
        return ()

    agent = _get_agent()
    payload = {"skills": list(skills_tuple)}
    try:
        response = agent(message=json.dumps(payload), temperature=0.0)
    except ProviderCapacityError:
        return ()

    parsed, _raw = coerce_json(response)
    return _extract_positive_skills(skills_tuple, parsed)


def filter_skills_via_llm(skills: List[str]) -> List[str]:
    """
    Use the configured LLM to label each token as a true technical skill or not.
    The model receives explicit yes/no instructions and only approved tokens are retained.
    """
    if not skills:
        return []

    unique: List[str] = []
    seen = set()
    for skill in skills:
        normalized = skill.strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered not in seen:
            unique.append(normalized)
            seen.add(lowered)

    if not unique:
        return []

    filtered: List[str] = []
    for chunk in _chunk_sequence(tuple(unique), _MAX_CHUNK_SIZE):
        filtered.extend(_filter_chunk(chunk))

    deduped: List[str] = []
    final_seen = set()
    for item in filtered:
        lowered = item.lower()
        if lowered not in final_seen:
            final_seen.add(lowered)
            deduped.append(item)

    return deduped
