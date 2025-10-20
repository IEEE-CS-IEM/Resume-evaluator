# Architecture & Algorithm Guide

This document explains how the Resume Evaluator project is organised, what each module does, and the algorithms that power the evaluation pipeline. It is intended for developers extending or integrating the system.

## Overview

The solution is built around a three-stage pipeline:

1. **Resume Summarisation** – An LLM extracts structured, ATS-ready signals from raw resume text.
2. **Job Description Structuring** – Another LLM distils a job description into comparable structured data.
3. **Comparison & Scoring** – Deterministic algorithms align resume vs JD signals, compute ATS-style metrics, and an LLM produces a narrative fit report.

Supporting utilities handle prompt management, response coercion, and configuration.

```
resume_text ──▶ resume_summary_generator ──▶ resume_summary_analyzer ──┐
                                                                      │
jd_text ─────▶ jd_structured_summary ────────────────▶ jd_resume_analyzer ──▶ fit_comparator
                                                                      │
                                                     util/pipeline orchestrates the flow
```

## Module Responsibilities

| Module | Purpose |
| --- | --- |
| `util/system_prompt.py` | Houses system prompts for the resume summariser, JD summariser, fit evaluation, and legacy combined prompt. |
| `util/simpleagent.py` | Provider-agnostic LLM wrapper supporting Gemini and Mistral. Handles API key configuration, model selection, and temperature control. |
| `util/llm_helpers.py` | Normalises LLM responses into plain text or JSON (`coerce_text`, `coerce_json`). |
| `util/resume_summary_generator.py` | Calls the resume summarisation prompt via `MyAgent` and returns structured JSON (with fallbacks if parsing fails). |
| `util/resume_summary_analyzer.py` | Merges LLM JSON with regex heuristics to extract skills, experience, knowledge signals, quantification gaps, and ATS score hints. |
| `util/jd_structured_summary.py` | Uses an LLM prompt to capture JD metadata (role, skills, tooling, experience requirements) and post-processes results into canonical categories. |
| `util/jd_resume_analyzer.py` | Deterministic matcher that aligns resume and JD signals, computing keyword coverage, experience alignment, knowledge alignment, category coverage, and blended ATS score. |
| `util/fit_comparator.py` | Asks the LLM to craft a narrative fit report using structured resume, JD, and score data. |
| `util/pipeline.py` | High-level orchestrator returning a dictionary with all intermediate artefacts (resume summary, resume signals, JD summary, scoring output, LLM fit report). |
| `util/constants.py` | Stores skill taxonomies, knowledge keywords, scoring weights, and default model names. |
| `docs/scoring.md` | Deep dive into the ATS scoring formula. |
| `LLMTest.py` | Example script that runs the pipeline on sample text and saves the ATS evaluation (`testresult.json`). |

## Stage 1 – Resume Summarisation

1. **Prompt**: `prompt_resume_summary` instructs the LLM to deliver a JSON payload with:
   - `summary_text`
   - lists of `core_skills`, `tooling`, `domain_experience`
   - `quantifiable_highlights`, `leadership_experience`
   - `total_years_experience`, `seniority`, and knowledge statements
2. **LLM Invocation**: `util/resume_summary_generator.generate_resume_summary`
   - Uses `MyAgent` (Gemini/Mistral) with low temperature to encourage deterministic output.
   - Parses the response via `coerce_json`, falling back to a template with sensible defaults if parsing fails.
3. **Signal Normalisation**: `util/resume_summary_analyzer.analyze_resume_summary`
   - Accepts either the JSON payload or raw text.
   - Uses shared skill taxonomies to map skills by category.
   - Detects years of experience, knowledge mentions (with levels + associated skills), quantification gaps, and captures LLM ATS scores if present.
   - Synthesises a heuristic ATS score when the LLM omits one so downstream scoring always receives a blended value.

## Stage 2 – Job Description Structuring

1. **Prompt**: `prompt_jd_structured` asks for role title, seniority, summary, skill breakdown, tooling, experience expectations, knowledge statements, and domains.
2. **LLM Invocation**: `util/jd_structured_summary.summarise_job_description`
   - Calls the LLM with temperature 0 for deterministic JSON output.
   - Normalises skill lists, knowledge statements, and experience requirements.
   - Uses `util/constants.SKILL_CATEGORIES` to build category-wise skill maps.
   - Derives required experience band (entry → principal).
3. **Fallback Handling**:
   - If JSON parsing fails, returns a minimal structure using the raw JD text.
   - Combines `must_have`, `nice_to_have`, and `tooling` into a unified skills list for matching.

## Stage 3 – Comparison & Scoring

1. **Keyword Scoring**: `util/jd_resume_analyzer.process_jd_and_resume`
   - Accepts JD structured summary (or raw text) and resume signals.
   - Computes:
     - Skill coverage (matched vs missing skills)
     - Category coverage (e.g., programming languages, cloud, devops)
     - Experience alignment (candidate vs required years)
     - Knowledge alignment (JD statements vs resume mentions/skills)
   - Uses weighted formula (see `docs/scoring.md`):
     - `skill_match` 45%
     - `experience_alignment` 30%
     - `knowledge_level` 15%
     - `category_coverage` 10%
   - Blends keyword score with either the LLM-provided ATS score or the heuristic fallback (`final_ats_score = 0.6 * keyword + 0.4 * llm_score`), ensuring the final score always reflects both perspectives.
   - Generates additional insights: experience relevance message, quantification suggestions, next skills to learn.
   - Returns a JSON-ready dict with `jd_insights`, `resume_insights`, `matching_summary`, and `scores` (including per-component subscores).
2. **Narrative Fit**: `util/fit_comparator.compare_resume_to_jd`
   - Sends resume summary, JD summary, and scoring breakdown to the LLM using `prompt_fit_evaluation`.
   - Expects JSON with `overall_fit`, `ats_score`, `strengths`, `gaps`, `recommendations`, and `narrative`.
   - Falls back to a default structure if parsing fails.

## Pipeline Orchestration

`util/pipeline.evaluate_resume_against_jd` executes the full workflow:

```python
{
  "resume_summary": { ... },        # structured LLM summary of resume_text
  "resume_signals": { ... },        # deterministic signals for scoring
  "jd_summary": { ... },            # structured LLM summary of jd_text
  "matching_evaluation": { ... },   # deterministic matching + scores
  "llm_fit_report": { ... }         # narrative LLM report
}
```

The example runner `LLMTest.py` saves `matching_evaluation` as `testresult.json` for inspection.

## Configuration & Environment

- `.env` (root or `util/.env`) should set:
  - `llm_provider` (`gemini` or `mistral`)
  - `GEMINI_API_KEY` / `MISTRAL_API_KEY` (case insensitive)
  - Optional `GEMINI_MODEL` / `MISTRAL_MODEL`
- Virtual env activation (`genai_env`) is expected before installing dependencies or running scripts.
- `requirements.txt` keeps only the libraries required for Gemini/Mistral support and document parsing (`python-docx`, `pdfplumber`).

## Extending the System

- **Skills Taxonomy**: Add new categories or keywords in `util/constants.SKILL_CATEGORIES` to expand detection coverage.
- **Scoring Weights**: Tweak `SCORING_WEIGHTS` to adjust emphasis on skill vs experience vs knowledge.
- **New Providers**: Extend `util/simpleagent.py` to support additional LLMs; update `requirements.txt` accordingly.
- **UI/Service Integration**: Import `evaluate_resume_against_jd` and feed real resume/JD text; persist or display the returned JSON as needed.

## Related Documents

- `docs/scoring.md` – Detailed explanation of the ATS scoring formula and subscores.
- `README.md` – Installation steps, project overview, and directory structure.
