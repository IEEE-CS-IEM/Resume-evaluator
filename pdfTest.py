import argparse
import json
import sys
from pathlib import Path

import requests

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


def upload_resume(base_url: str, user_id: str, resume_path: Path) -> None:
    """POST the resume to the FastAPI upload endpoint."""
    with resume_path.open("rb") as handle:
        files = {"file": (resume_path.name, handle, "application/pdf")}
        response = requests.post(f"{base_url}/users/{user_id}/resume", files=files, timeout=60)
    response.raise_for_status()
    print(f"[upload] {response.json()}")


def evaluate_resume(base_url: str, user_id: str, jd_text: str) -> dict:
    """POST to the evaluation endpoint and return the JSON response."""
    payload = {"jd_text": jd_text}
    response = requests.post(f"{base_url}/users/{user_id}/evaluate", json=payload, timeout=120)
    response.raise_for_status()
    result = response.json()
    print("[evaluate] Received evaluation payload.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Upload a PDF resume and evaluate it against the sample JD via FastAPI."
    )
    parser.add_argument("--resume", required=True, type=Path, help="Path to the resume PDF file.")
    parser.add_argument("--user-id", required=True, help="User identifier used for storage.")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL where the FastAPI app is running.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("pdf_test_result.json"),
        help="Optional path to write the evaluation JSON.",
    )
    args = parser.parse_args()

    if not args.resume.exists():
        print(f"Resume file not found: {args.resume}", file=sys.stderr)
        sys.exit(1)

    try:
        upload_resume(args.base_url, args.user_id, args.resume)
        evaluation = evaluate_resume(args.base_url, args.user_id, DEFAULT_JD_TEXT)
    except requests.HTTPError as exc:
        print(f"Request failed: {exc} – {exc.response.text}", file=sys.stderr)
        sys.exit(1)

    args.output.write_text(json.dumps(evaluation, indent=2, ensure_ascii=False))
    print(f"[evaluate] Saved evaluation to {args.output.resolve()}")


if __name__ == "__main__":
    main()
