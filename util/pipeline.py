from typing import Any, Dict

from util.fit_comparator import compare_resume_to_jd
from util.jd_resume_analyzer import process_jd_and_resume
from util.jd_structured_summary import summarise_job_description
from util.resume_summary_analyzer import analyze_resume_summary
from util.resume_summary_generator import generate_resume_summary


def evaluate_resume_against_jd(
    resume_text: str,
    jd_text: str,
) -> Dict[str, Any]:
    """
    End-to-end pipeline that:
    1. Summarises the resume via LLM.
    2. Generates structured signals from the resume summary.
    3. Summarises the job description via LLM.
    4. Runs keyword/ATS scoring between resume and JD.
    5. Produces an LLM-driven fit narrative.
    """
    resume_summary = generate_resume_summary(resume_text)
    resume_signals = analyze_resume_summary(resume_summary)

    jd_summary = summarise_job_description(jd_text)
    evaluation = process_jd_and_resume(jd_summary, resume_signals)

    llm_evaluation = compare_resume_to_jd(
        resume_summary,
        jd_summary,
        evaluation["scores"],
    )

    def _prune_resume_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(summary, dict):
            return summary
        pruned = {
            key: value
            for key, value in summary.items()
            if key
            not in {
                "summary_text",
                "quantifiable_highlights",
                "leadership_experience",
                "knowledge_statements",
                "quantification_suggestions",
            }
        }
        return pruned

    def _prune_jd_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(summary, dict):
            return summary
        pruned = {
            key: value
            for key, value in summary.items()
            if key not in {"raw_text", "llm_summary"}
        }
        return pruned

    def _prune_fit_report(report: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(report, dict):
            return report
        pruned = {
            key: value
            for key, value in report.items()
            if key not in {"narrative"}
        }
        return pruned

    return {
        "resume_summary": _prune_resume_summary(resume_summary),
        "resume_signals": resume_signals,
        "jd_summary": _prune_jd_summary(jd_summary),
        "matching_evaluation": evaluation,
        "llm_fit_report": _prune_fit_report(llm_evaluation),
    }
