# LLM Resume Evaluator

An AI-powered resume evaluation system that analyses candidate profiles against job descriptions and provides detailed feedback on job readiness.
Current version changes the setup from the previous one to test based on the simpleagent module. 
LLMFactory is can also be used for the same. 

## Author : SATABARTO SARKAR

##  Features

- **Multi-LLM Support**: Compatible with multiple language models (Gemini, Mistral, local Ollama models)
- **Three-Stage Pipeline**: Dedicated LLM prompts for resume summarisation, JD extraction, and fit comparison
- **Structured Output**: Provides JSON-formatted evaluation reports plus deterministic ATS scoring
- **Keyword + LLM Scoring**: Blends objective keyword alignment with LLM readiness scores, with a deterministic fallback when the model omits one
- **Modular Architecture**: Clean separation of concerns with utility modules
- **Environment Management**: Secure API key management with environment variables

##  Project Structure

```
LLM-RESUME_EVALUATOR/
├── util/                       # Utility modules
│   ├── __pycache__/           # Python cache files
│   ├── constants.py           # Model configuration constants and skill taxonomy
│   ├── fit_comparator.py      # LLM-backed narrative fit evaluation
│   ├── jd_resume_analyzer.py  # Deterministic keyword + ATS scorer
│   ├── jd_structured_summary.py # LLM-based JD summariser
│   ├── pipeline.py            # Orchestrates resume→JD→comparison workflow
│   ├── resume_summary_analyzer.py # Extracts signals from LLM resume summary and scores heuristics
│   ├── resume_summary_generator.py # LLM-based resume summariser
│   ├── simpleagent.py         # LLM client wrapper
│   └── system_prompt.py       # System prompts for each pipeline stage
├── docs/
│   ├── architecture.md        # Full algorithm and module documentation
│   ├── integration-guide.md   # Hand-holding integration recipes
│   └── scoring.md             # Scoring methodology documentation
├── .venv/                     # Virtual environment (Python)
├── genai_env/                 # GenAI specific environment
├── langchain_env/             # LangChain specific environment
├── activate_genai.bat         # Windows batch script for GenAI env
├── activate_genai.ps1         # PowerShell script for GenAI env
├── activate_langchain.bat     # Windows batch script for LangChain env
├── activate_langchain.ps1     # PowerShell script for LangChain env
├── .gitignore                 # Git ignore rules
├── LLMTest.py                 # Main application entry point
├── requirements.txt           # Python dependencies
└── README.md                  # This documentation
```

##  Installation

### Prerequisites
- Python 3.8 or higher
- Windows environment (based on current setup)
- API keys for supported LLM providers

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/satabartosarkar123/resume_Evauator
   cd LLM-RESUME_EVALUATOR
   ```

2. **Create and activate virtual environment**
   
   For GenAI environment:
   ```powershell
   # Using PowerShell
   .\activate_genai.ps1
   
   # Using Command Prompt
   .\activate_genai.bat
   ```
   
   For LangChain environment:
   ```powershell
   # Using PowerShell
   .\activate_langchain.ps1
   
   # Using Command Prompt
   .\activate_langchain.bat
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   
   Create a `.env` file in the `util/` directory with your API keys:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   mistral_api_key=your_mistral_api_key_here
   gemini_api_key=your_gemini_api_key_here
   llm_provider=gemini  # or mistral
   local_model_url=http://localhost:11434  # for local Ollama models
   ```

##  Usage

### Basic Usage

Run the main application:
```bash
python LLMTest.py
```

The application will prompt you for:
1. **User Prompt**: Resume content or candidate profile information
2. **Job Description**: The job requirements and description

### Example Input

**User Prompt:**
```
Software Engineer with 3 years experience in Python, Django, and React. 
Bachelor's in Computer Science. Experience with AWS, Docker, and PostgreSQL.
```

**Job Description:**
```
Looking for a Senior Python Developer with 5+ years experience in Django, 
React, cloud technologies (AWS/Azure), and database management. 
Master's degree preferred.
```

### Output Format

The system returns a structured JSON evaluation:

```json
{
  "strength": [
    {"Relevant Education": "Bachelor's in Computer Science aligns with technical requirements"},
    {"Programming Skills": "Strong Python and Django experience matches core requirements"},
    {"Soft Skills": "Not clearly demonstrated"}
  ],
  "weakness": [
    "Experience level (3 years) below preferred requirement (5+ years)",
    "Missing Master's degree preference",
    "No mention of Azure cloud experience"
  ],
  "Area to Improve": [
    "Gain additional 2 years of relevant experience",
    "Consider pursuing advanced certifications in cloud technologies",
    "Develop Azure cloud platform skills"
  ],
  "readiness": "Almost Ready"
}
```

##  Architecture

### Core Components

1. **util/pipeline.py** – Coordinates the three-stage evaluation workflow (resume → JD → comparison).
2. **util/resume_summary_generator.py** – LLM-driven resume summariser that extracts ATS-ready signals.
3. **util/jd_structured_summary.py** – LLM-driven JD analyser that builds a structured requirement profile.
4. **util/jd_resume_analyzer.py** – Deterministic scorer that matches resume vs JD and computes ATS metrics.
5. **util/fit_comparator.py** – LLM narrative layer that fuses scoring data into actionable feedback.
6. **util/resume_summary_analyzer.py** – Normalises resume summary output and synthesises ATS fallback scores for the scoring engine.
7. **util/simpleagent.py / util/system_prompt.py** – Shared LLM client wrapper and prompt library.
8. **util/constants.py** – Skill taxonomies, knowledge keywords, and scoring weights.
9. **LLMTest.py** – Example script wiring the pipeline together for local testing.

### Supported LLM Providers

- **Google Gemini**: `gemini-1.5-flash` (Primary)
- **Mistral AI**: `mistral-large-latest`
- **Local Models**: `llama3.2:3b` via Ollama

##  Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GEMINI_API_KEY` | Google Gemini API key | Yes (for Gemini) |
| `mistral_api_key` | Mistral AI API key | Yes (for Mistral) |
| `gemini_api_key` | Alternative Gemini key format | Yes (for Gemini) |
| `llm_provider` | Active LLM provider (`gemini`/`mistral`) | Yes |
| `local_model_url` | Ollama server URL | Optional |

### Model Configuration

Edit `util/constants.py` to modify model versions:
```python
mistral_llm = "mistral-large-latest"
gemini_llm = "gemini-1.5-flash"
local_llm = "llama3.2:3b"
```

##  Development

### Adding New LLM Providers

1. Install the provider's SDK or LangChain integration inside your virtual environment.
2. Extend `util/simpleagent.py` to recognise the new provider and configure credentials.
3. Add the provider-specific prompt logic if different prompting is required.
4. Update `requirements.txt` with the dependency so deployments remain reproducible.
5. Document the new provider in `docs/architecture.md` for future contributors.

### Customizing Evaluation Criteria

Modify `util/system_prompt.py` to adjust:
- Evaluation criteria
- Output format
- Assessment categories
- Readiness levels

##  Dependencies

### Core Dependencies
- `langchain-mistralai` – MistralAI chat integration
- `langchain-google-genai` – Google Gemini chat integration
- `langchain-core>=0.2` – Core LangChain message abstractions
- `langchain-community` – Shared message and tool primitives
- `python-dotenv` – Environment variable management
- `python-docx` – DOC/DOCX parsing for job descriptions
- `pdfplumber` – PDF parsing for job descriptions
- `google-generativeai` – Native Gemini client used by `simpleagent`

### Optional Dependencies
- `langchain-community[redis,mongo]`, `langchain-openai`, or other ecosystem packages can be added as needed but are not required by the default pipeline.

##  Further Documentation

- `docs/architecture.md` – End-to-end explanation of the pipeline, module responsibilities, and algorithms.
- `docs/scoring.md` – Detailed ATS scoring formula and subscores.

*Last updated: July 2025*
