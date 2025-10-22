mistral_llm = "mistral-large-latest"
gemini_llm = "gemini-1.5-flash"
local_llm = "llama3.2:3b"

# Canonical skill dictionaries used for JD and resume parsing.
SKILL_CATEGORIES = {
    "programming_languages": {
        "python",
        "java",
        "javascript",
        "typescript",
        "c++",
        "c#",
        "c",
        "go",
        "rust",
        "kotlin",
        "scala",
        "matlab",
    },
    "frameworks": {
        "react",
        "angular",
        "vue",
        "spring boot",
        "django",
        "flask",
        "fastapi",
        "node.js",
        "express",
    },
    "cloud_platforms": {
        "aws",
        "azure",
        "gcp",
        "oracle cloud",
        "digital ocean",
    },
    "devops_tools": {
        "docker",
        "kubernetes",
        "terraform",
        "jenkins",
        "github actions",
        "gitlab ci",
        "ansible",
    },
    "data_platforms": {
        "postgresql",
        "mysql",
        "mongodb",
        "sql",
        "redis",
        "elasticsearch",
        "kafka",
        "spark",
        "hadoop",
    },
    "ml_ai": {
        "tensorflow",
        "pytorch",
        "scikit-learn",
        "huggingface",
        "mlflow",
    },
    "testing_quality": {
        "pytest",
        "junit",
        "selenium",
        "cypress",
        "postman",
    },
    "frontend_basics": {
        "html",
        "css",
    },
}

# Phrases indicating the depth of knowledge or expertise.
KNOWLEDGE_LEVEL_KEYWORDS = {
    "expert": {"expert", "expertise", "advanced", "deep understanding"},
    "proficient": {"proficient", "strong knowledge", "hands-on"},
    "familiar": {"familiar", "basic understanding", "exposure"},
}

EXPERIENCE_LEVEL_KEYWORDS = {
    "leadership": {"lead", "led", "managed", "mentored", "architected"},
    "delivery": {"implemented", "delivered", "shipped", "deployed"},
    "impact": {"improved", "reduced", "increased", "optimized"},
}

SCORING_WEIGHTS = {
    "skill_match": 0.45,
    "experience_alignment": 0.3,
    "knowledge_level": 0.15,
    "category_coverage": 0.1,
}

QUANTIFIABLE_VERBS = {
    "improved",
    "reduced",
    "increased",
    "optimized",
    "accelerated",
    "boosted",
    "cut",
    "saved",
    "grew",
}
