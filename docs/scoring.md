# ATS Scoring Design

This document explains how the revamped JD/resume evaluation pipeline derives the final ATS score and the supporting signals that appear in `process_jd_and_resume`.

## Pipeline Overview

1. **Resume summarisation** – The LLM produces a narrative summary.  
2. **Resume signal extraction** – `util/resume_summary_analyzer.py` scans the summary for skills, experience, knowledge cues, and quantification gaps.  
3. **JD parsing** – `util/jd_resume_analyzer.py` normalises the job description (raw string or file) and extracts the same set of structured attributes.  
4. **Keyword matching** – Skills, knowledge statements, and experience requirements are aligned to surface matches and gaps.  
5. **Scoring** – Objective keyword scores and any LLM-provided score are blended into a final ATS score.

## Keyword Score (0–100)

The keyword score is a weighted combination of four subscores:

| Component              | Description                                                      | Weight |
| ---------------------- | ---------------------------------------------------------------- | ------ |
| `skill_match`          | Ratio of matched skills to the total skills requested in the JD. | 0.45   |
| `experience_alignment` | Alignment between candidate experience and JD requirement.       | 0.30   |
| `knowledge_level`      | Coverage of knowledge/familiarity statements in the JD.          | 0.15   |
| `category_coverage`    | Coverage of key competency buckets (cloud, data, devops, etc.).  | 0.10   |

Subscores are normalised to the range 0–1 before the weighted sum is converted to a percentage. The individual subscores are exported alongside the aggregate score for transparency.

## Final ATS Score

When the LLM summary contains an explicit numeric score (e.g. “ATS Score: 78”), it is detected and blended with the keyword score:

```
final_score = 0.6 * keyword_score + 0.4 * llm_score
```

If the LLM does not supply a score, a zero placeholder is used for the LLM score and the keyword score is reported as the final ATS score. A boolean flag indicates whether the LLM score contributed to the blend.

## Additional Insights

- **Experience relevance** summarises whether the candidate meets, exceeds, or falls short of the stated experience requirement.  
- **Quantification suggestions** flag impact statements that lack measurable data.  
- **Category breakdown** pinpoints matched/missing skills by competency area to guide resume edits or upskilling plans.  
- **Knowledge alignment** leverages both JD phrasing and skill overlaps from the resume to reduce false zero scores when wording differs.
