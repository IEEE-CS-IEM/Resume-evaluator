# Integration Guide (Step-by-Step)

This guide shows how to use the Resume Evaluator modules in every common scenario. Each recipe includes the function to call, the inputs to pass, and the output you will receive.

---

## 1. Resume → Structured Summary

**Goal:** Convert raw resume text into clean JSON via LLM.

```
from util.resume_summary_generator import generate_resume_summary

resume_text = "... raw resume content ..."
summary = generate_resume_summary(resume_text)
```

**Important notes**
- `resume_text` should be a single string (OCR noise is fine).
- Requires an active virtual environment with API keys (`llm_provider=gemini` or `mistral`).
- `summary` is a dictionary with keys such as `summary_text`, `core_skills`, `tooling`, `domain_experience`, `quantifiable_highlights`, `leadership_experience`, `total_years_experience`, `seniority`, `knowledge_statements`.
- If the LLM response cannot be parsed, the function returns a fallback dictionary with empty lists and the raw text.

**Common use-cases**
- UI teams: show the structured summary to users.
- Diagnosis: confirm the LLM extracted skills correctly before scoring.

---

## 2. Structured Summary → Deterministic Signals

**Goal:** Generate ATS-friendly signals and metrics from the summary JSON (or raw text).

```
from util.resume_summary_analyzer import analyze_resume_summary

resume_summary = { ... }  # output of generate_resume_summary()
signals = analyze_resume_summary(resume_summary)
```

**Input options**
- A dictionary from `generate_resume_summary()`.
- A plain string containing resume text (the analyzer will process it directly).

**Output (`signals`)**
- `skills`: sorted list of skills (lowercase, taxonomy aware).
- `skills_by_category`: dictionary of category → skills.
- `years_experience`: float or `None`.
- `experience_band`: entry/junior/mid-level/senior/principal/unknown.
- `knowledge_mentions`: list of snippets with inferred level/skills.
- `quantification_suggestions`: list of sentences that need metrics.
- `llm_score`: float ATS score (explicitly parsed from the LLM summary when present, otherwise a heuristic fallback).
- `llm_score_available`: always `True` because the analyzer guarantees a usable score.

**Common use-cases**
- Pipeline steps preceding JD comparison.
- Dashboards where deterministic metrics matter.

---

## 3. Job Description → Structured Summary

**Goal:** Normalize a JD into the same structure as the resume summary.

```
from util.jd_structured_summary import summarise_job_description

jd_text = "... job description ..."
jd_summary = summarise_job_description(jd_text)
```

**Input options**
- Raw JD string.
- The JD text may include bullet lists or plain paragraphs.

**Output (`jd_summary`)**
- `role_title`, `seniority_level`, `llm_summary`.
- `must_have_skills`, `nice_to_have_skills`, `tooling`, `domains`.
- `experience_requirement` (minimum/max years, described range).
- `knowledge_requirements`: list of verbatim sentences with skills/categories.
- `skills_by_category`: taxonomy-based categorization.
- `required_years`, `required_experience_band`.
- `raw_text` (original JD).

**Common use-cases**
- HR tooling: store the structured JD alongside job postings.
- Pipeline step before running ATS scoring.

---

## 4. Resume Signals + JD Summary → ATS Comparison

**Goal:** Compute keyword coverage, experience alignment, knowledge coverage, and ATS scores.

```
from util.jd_resume_analyzer import process_jd_and_resume

# Option A: pass the outputs of earlier steps
evaluation = process_jd_and_resume(jd_summary, signals)

# Option B: quick comparison from raw text
evaluation = process_jd_and_resume(jd_text, resume_text)
```

**Output (`evaluation`)**
- `jd_insights`: structured JD info (skills, categories, experience band, knowledge).
- `resume_insights`: structured resume data (skills, categories, experience, knowledge, quantification suggestions, llm score).
- `matching_summary`: matched/missing skills, category breakdown, experience alignment (0-1), knowledge alignment (0-1), category coverage (0-1), experience relevance narration.
- `scores`: keyword score, final ATS score, `llm_score_used` (always true because the analyzer guarantees a score), and subscores (skill coverage, experience alignment, knowledge alignment, category coverage).
- `recommendations`: quantification suggestions, next skills to learn.

**Common use-cases**
- Main ATS scoring engine.
- Analytics backend for ranking candidates vs JD.

---

## 5. Fit Narrative from Structured Data

**Goal:** Produce a human-readable report summarizing readiness.

```
from util.fit_comparator import compare_resume_to_jd

fit_report = compare_resume_to_jd(
    resume_summary=summary,
    jd_summary=jd_summary,
    scoring_breakdown=evaluation["scores"],
)
```

**Output (`fit_report`)**
- `overall_fit`: Ready / Almost Ready / Not Ready.
- `ats_score`: numeric value (mirrors `final_ats_score`).
- `strengths`, `gaps`, `recommendations`: bullet lists.
- `narrative`: paragraph providing context.

**Common use-cases**
- Email or PDF reports to candidates.
- Presentation layer for recruiters or hiring managers.

---

## 6. End-to-End Pipeline (All Steps Together)

**Goal:** Go from raw texts to a full evaluation in a single call.

```
from util.pipeline import evaluate_resume_against_jd

report = evaluate_resume_against_jd(resume_text, jd_text)
```

**Output (`report`)**
- `resume_summary`: JSON summary from Stage 1.
- `resume_signals`: deterministic signals from Stage 2.
- `jd_summary`: structured JD summary from Stage 2.
- `matching_evaluation`: Stage 3 keyword/ATS output.
- `llm_fit_report`: Stage 3 narrative fit.

`LLMTest.py` is an example script that runs this pipeline end-to-end and saves the `matching_evaluation` portion to `testresult.json`.

---

## 7. Partial Workflows & Swap Options

- Need **resume-only**? Call steps 1–2.
- Need **JD-only**? Call step 3.
- Need **scoring only**? Ensure you have resume signals (step 2) and JD summary (step 3), then run step 4.
- Want just the narrative? Provide resume/JD summaries and a scoring dictionary to step 5.
- Reuse the prompts? They live in `util/system_prompt.py`.

---

## 8. Environment Checklist

Before running any of the above:
- Activate your virtual environment (`source genai_env/bin/activate`).
- Install dependencies (`pip install -r requirements.txt`).
- Set environment variables (`.env` file or shell exports):
  ```
  llm_provider=gemini
  GEMINI_API_KEY=...
  # or MISTRAL_API_KEY if using mistral
  ```

---

## 9. Troubleshooting Tips

| Symptom | Check |
| --- | --- |
| `ModuleNotFoundError` for `google.generativeai` | `pip install google-generativeai` in the active env. |
| LLM call fails with 404 model | Ensure `GEMINI_MODEL` or `MISTRAL_MODEL` matches an available model. |
| Resume/JD parsing yields empty skills | Confirm text encoding, ensure taxonomy contains the skills you expect, or add new keywords to `util/constants.SKILL_CATEGORIES`. |
| `final_ats_score` looks low | Inspect `subscores` to see which component (skill, experience, knowledge, category coverage) is responsible. |
| Need faster responses | Reduce temperature (already set low), and make sure you are using the lightweight model variants (e.g., `gemini-1.5-flash`). |

---

You’re now ready to plug any part of the pipeline into your application. Pick the recipe that matches your use case, drop it into your code, and inspect the dictionary output to integrate with downstream systems. Happy building!
