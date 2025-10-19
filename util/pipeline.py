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

    return {
        "resume_summary": resume_summary,
        "resume_signals": resume_signals,
        "jd_summary": jd_summary,
        "matching_evaluation": evaluation,
        "llm_fit_report": llm_evaluation,
    }
