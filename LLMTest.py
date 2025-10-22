import json

from dotenv import load_dotenv

from util.pipeline import evaluate_resume_against_jd


if __name__ == "__main__":
    load_dotenv()

    resume_text = """"Aarav  Mukherjee
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

About Meta:

Meta builds technologies that help people connect, find communities, and grow businesses. When Facebook launched in 2004, it changed the way people connect. Apps like Messenger, Instagram and WhatsApp further empowered billions around the world. Now, Meta is moving beyond 2D screens toward immersive experiences like augmented and virtual reality to help build the next evolution in social technology. People who choose to build their careers by building with us at Meta help shape a future that will take us beyond what digital connection makes possible today—beyond the constraints of screens, the limits of distance, and even the rules of physics.

Individual compensation is determined by skills, qualifications, experience, and location. Compensation details listed in this posting reflect the base hourly rate, monthly rate, or annual salary only, and do not include bonus, equity or sales incentives, if applicable. In addition to base compensation, Meta offers benefits. Learn more about  benefits  at Meta.

"""
    if not resume_text or not jd_text:
        raise ValueError("Both resume text and job description are required.")

    report = evaluate_resume_against_jd(resume_text, jd_text)

    with open("testresult.json", "w", encoding="utf-8") as handle:
        json.dump(report["matching_evaluation"], handle, indent=2, ensure_ascii=False)

    print("Saved analysis to testresult.json")


    
