import json

from dotenv import load_dotenv

from util.pipeline import evaluate_resume_against_jd


if __name__ == "__main__":
    load_dotenv()

    resume_text = """"Aarav  Mukh erjee
Software  Engineer
Ph: +91-98765O1234    •  Emal: aarav.mukherjeegmail.com
Linkedln:  ln/k/aarav-mukherjee   •  Location: Kol kata, India

SUMMARY
Highly motivated software engineer with 4+ years of experience in full-stack
web development, microservices and ML- integrated applications. Strong back-
ground in Python, Java, React and cloud platforms (AWS, GCP). Seeks to con-
tribute to scalable systems and data-driven products.

EXPERIENCE
Senior Software Engineer
NexGen Solutions Pvt. Ltd.                                 Janua ry 2021 - Present
- Led a 4-member team to design and implement a microservices-architec-
  ture for the company's payments platform, reducing latency by ~30%.
- Built RESTful APIs (Java Spring Boot) and deployed using Docker / Kuber-
  netes on AWS. Implemented CI/CD pipelines (Jenkins + AWS CodePipeline).
- Integrated ML inference service for fraud detection; throughput im-
  proved by 2x after model optimiz ation and batching.
- Mentored interns and conducted code reviews, improving code quality
  metrics (unit test coverage increase from 42% to 68%).

Software Engineer
ByteCraft Labs                                            Jul 2018 - Dec 2020
- Developed front-end features in React.js and redux; improved UX re-
  sponse times via code-splitting and lazy load.
- Implemented authentication & authorization using OAuth2 and JWT.
- Worked on data ETL jobs using Python and Airflow; reduced ETL runtimes
  by 25% through parallelisation and optimized queries.

EDUCATION
B.Tech., Computer Science & Engineering
Techno Institute of Engineering, Kolkata                        2014 - 2018
- CGPA: 8.12/10
- Relevant coursework: Data Structures, Algorithms, Databases, ML.

SKILLS
Languages: Python, Java, JavaScript (ES6+), SQL
Frameworks: Spring Boot, React, Node.js, Flask
Cloud / Infra: AWS (EC2, S3, RDS, Lambda), Docker, Kubernetes
Data / ML: pandas, scikit-learn, TensorFlow (basic), Airflow
Tools: Git, Jenkins, Jira, Postman, Redis, Elasticsearch

PROJECTS
Real-time Order Matching System (Side Project)
- Implemented a lightweight matching engine in Java that supports limit/
  market orders with concurrency control. Used Redis for order-book cache.

Invoice OCR & Processing (NexGen)
- Built a pipeline to extract invoice fields using Tesseract + custom NLP
  rules. Post-correction reduced manual review by ~60%.

CERTIFICATIONS
- AWS Certified Solutions Architect – Assoc (2022)
- Professional Certificate in Machine Learning (Coursera) (2020)

ACHIEVEMENTS
- Winner, Hackathon XYZ 2019 — Built a P2P marketplace prototype.
- Published blog on microservices best practices (medium.com/@aarav-m).

PERSONAL
Languages: English, Hindi, Bengali
Interests: Open-source contribution, chess, long-distance running

REFERENCES
Available on request.
"""
    jd_text = """

Software Development Engineer (SDE I / II)

Location: Bengaluru / Hyderabad / Remote (India)
Company: Oracle / FAANG-equivalent enterprise

About the Role

We are seeking passionate, inventive, and results-driven Software Development Engineers to design, develop, and deliver high-impact software systems used by millions of customers worldwide. As an SDE, you will work in an agile, collaborative environment and play a key role in building scalable distributed systems, optimizing backend performance, and improving developer productivity across teams.

You will own projects end-to-end — from design and implementation to testing and deployment — while working alongside some of the best engineers in the industry.

Responsibilities

Design, develop, test, and deploy scalable, reliable, and maintainable software systems.

Translate functional and technical requirements into detailed architecture and design.

Write clean, efficient, well-documented, and reusable code following best software engineering practices.

Participate in code reviews, architectural discussions, and team design sessions.

Collaborate closely with product managers, designers, and QA engineers to deliver high-quality solutions.

Contribute to continuous improvement in development processes, deployment pipelines, and system observability.

Analyze and improve the efficiency, scalability, and stability of distributed systems.

Debug production issues across services and multiple levels of the stack.

Required Qualifications

Bachelor’s or Master’s degree in Computer Science, Engineering, or related technical discipline.

Proficiency in at least one modern programming language such as Java, C++, Python, Go, or Kotlin.

Strong fundamentals in data structures, algorithms, object-oriented design, and system design.

Solid understanding of distributed systems, RESTful APIs, and microservices architecture.

Experience with SQL/NoSQL databases, version control (Git), and CI/CD pipelines.

Familiarity with cloud platforms (AWS, Azure, GCP, or Oracle Cloud).

Preferred Qualifications

Experience building and scaling backend systems with high availability and low latency.

Working knowledge of containerization and orchestration tools (Docker, Kubernetes).

Exposure to machine learning pipelines, streaming systems (Kafka, Spark), or large-scale data processing.

Experience with frontend frameworks (React, Angular, or Vue) is a plus.

Contributions to open-source projects or a strong GitHub/technical blog presence.

Soft Skills

Strong problem-solving and analytical skills.

Excellent communication and collaboration abilities.

Ownership mindset and ability to thrive in ambiguous, fast-paced environments.

Passion for learning, mentoring, and technical excellence.

Compensation & Benefits

Competitive base salary with performance-based bonuses.

Stock options / RSUs (depending on level and company).

Health and wellness benefits, flexible working hours, and learning reimbursements.

Opportunities to work on cross-functional, high-impact global projects.

Sample Job Titles under Similar Roles

Software Development Engineer (SDE I / SDE II)

Backend Engineer / Full-Stack Engineer

Cloud Application Developer

Systems Engineer – Distributed Computing

Software Engineer – Data Platform

Would you like me to also:

🔹 Generate an OCR-style extracted version of this JD (with typical OCR distortions)?

🔹 Or produce a JD–Resume matching pair, where this JD is paired with the earlier Aarav Mukherjee OCR resume for a realistic ATS or ML model test case?
"""
    if not resume_text or not jd_text:
        raise ValueError("Both resume text and job description are required.")

    report = evaluate_resume_against_jd(resume_text, jd_text)

    with open("testresult.json", "w", encoding="utf-8") as handle:
        json.dump(report["matching_evaluation"], handle, indent=2, ensure_ascii=False)

    print("Saved analysis to testresult.json")


    
