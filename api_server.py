"""
FastAPI application exposing endpoints to upload resume PDFs and evaluate them
against a job description using the existing analysis utilities.

Endpoints
---------
- POST /users/{user_id}/resume
    Accepts a multipart/form-data request with a `file` field (PDF) and stores
    it on disk, mimicking Multer-style persistence.

- POST /users/{user_id}/evaluate
    Accepts an optional `jd_text` payload, reads the previously uploaded PDF,
    extracts text, and runs the deterministic scoring pipeline.
"""

from pathlib import Path
from typing import Optional

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from util.jd_resume_analyzer import extract_text_from_file, process_jd_and_resume

UPLOAD_ROOT = Path(__file__).resolve().parent / "uploads"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_JD_TEXT = """
Meta is seeking talented engineers to join our teams in building cutting-edge products, with the mission of connecting billions of people around the world. As a member of our team, you will have the opportunity to work on complex technical problems, build new features, and improve existing products across various platforms, including mobile devices and web applications. Our teams are constantly pushing the boundaries of user experience, and we're looking for passionate individuals who can help us advance the way people connect globally. If you're interested in joining a world-class team and working on exciting projects that have a significant impact, we encourage you to apply.

Software Engineer (University Grad) - Bangalore Responsibilities:

Develop a strong understanding of relevant product area, codebase, and/or systems
Demonstrate proficiency in data analysis, programming and software engineering
Produce high quality code with good test coverage, using modern abstractions and frameworks
Work independently, use available resources to get unblocked, and complete tasks on-schedule by exercising strong judgement and problem solving skills
Master Meta’s development standards from developing to releasing code in order to take on tasks and projects with increasing levels of complexity
Actively seek and give feedback in alignment with Meta’s Performance Philosophy

Minimum Qualifications:

Currently has, or is in the process of obtaining a Bachelor's degree in Computer Science, Computer Engineering, relevant technical field, or equivalent practical experience. Degree must be completed prior to joining Meta
Experience coding in an industry-standard language (e.g. Java, Python, C++, JavaScript)
Must obtain work authorization in country of employment at the time of hire, and maintain ongoing work authorization during employment

Preferred Qualifications:

Demonstrated software engineering experience from previous internship, work experience, coding competitions, or publications
Currently has, or is in the process of obtaining, a Bachelors or Masters degree in Computer Science or a related field
"""

app = FastAPI(title="Resume Evaluator API")


@app.post("/users/{user_id}/resume")
async def upload_resume(user_id: str, file: UploadFile = File(...)) -> JSONResponse:
    """
    Store the uploaded resume PDF on disk.

    Parameters
    ----------
    user_id: str
        Identifier used to namespace the stored file.
    file: UploadFile
        PDF file uploaded via multipart/form-data.
    """
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Only PDF resumes are supported.")

    user_dir = UPLOAD_ROOT / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    destination = user_dir / "resume.pdf"

    with destination.open("wb") as buffer:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            buffer.write(chunk)

    return JSONResponse(
        {
            "message": "Resume uploaded successfully.",
            "user_id": user_id,
            "stored_path": str(destination),
        }
    )


@app.post("/users/{user_id}/evaluate")
def evaluate_resume(
    user_id: str,
    jd_text: Optional[str] = Body(None, embed=True),
) -> JSONResponse:
    """
    Run the deterministic JD/resume comparison for the stored PDF.

    Parameters
    ----------
    user_id: str
        Identifier whose stored resume should be evaluated.
    jd_text: Optional[str]
        Raw job description text. When omitted, defaults to the Meta JD used in
        `LLMTest.py`.
    """
    resume_path = UPLOAD_ROOT / user_id / "resume.pdf"
    if not resume_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No resume has been uploaded for this user.",
        )

    try:
        resume_text = extract_text_from_file(str(resume_path))
    except Exception as exc:  # pragma: no cover - surface extraction issues
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read resume PDF: {exc}",
        ) from exc

    jd_source = jd_text.strip() if jd_text and jd_text.strip() else DEFAULT_JD_TEXT
    evaluation = process_jd_and_resume(jd_source, resume_text)

    return JSONResponse(evaluation)


@app.get("/health")
def healthcheck() -> JSONResponse:
    """Simple readiness probe."""
    return JSONResponse({"status": "ok"})
