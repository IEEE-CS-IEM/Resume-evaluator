# LLM Resume Evaluator

An AI-powered resume evaluation system that analyses candidate profiles against job descriptions and provides detailed feedback on job readiness.
Current version changes the setup from the previous one to test based on the simpleagent module. 
LLMFactory is can also be used for the same. 

## Author : SATABARTO SARKAR

##  Features

- **Multi-LLM Support**: Compatible with multiple language models (Gemini, Mistral, local Ollama models)
- **Intelligent Resume Analysis**: Evaluates resumes against specific job requirements
- **Structured Output**: Provides JSON-formatted evaluation reports
- **Modular Architecture**: Clean separation of concerns with utility modules
- **Environment Management**: Secure API key management with environment variables

##  Project Structure

```
LLM-RESUME_EVALUATOR/
├── util/                       # Utility modules
│   ├── __pycache__/           # Python cache files
│   ├── .env                   # Environment variables (not in repo)
│   ├── constants.py           # Model configuration constants
│   ├── llm_factory.py         # LLM factory for multi-provider support
│   ├── simpleagent.py         # Simplified agent using Google Gemini
│   └── system_prompt.py       # System prompts for resume evaluation
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

1. **LLMTest.py**: Main application entry point
   - Handles user input collection
   - Orchestrates the evaluation process
   - Displays results

2. **util/simpleagent.py**: Google Gemini Agent
   - Simplified interface for Gemini API
   - Handles message formatting and API calls
   - Primary evaluation engine (currently active)

3. **util/llm_factory.py**: Multi-LLM Factory Pattern
   - Supports multiple LLM providers (Mistral, Gemini, Ollama)
   - Dynamic model selection based on environment configuration
   - Extensible architecture for adding new providers

4. **util/system_prompt.py**: Evaluation Logic
   - Contains expert-level resume evaluation prompts
   - Defines structured output format
   - Ensures consistent evaluation criteria

5. **util/constants.py**: Configuration Constants
   - Model name definitions
   - Version management for different LLM providers

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

1. Install the provider's LangChain integration
2. Add the provider to `util/llm_factory.py`
3. Update the model mapping in `get_model_name()`
4. Add API key handling in `get_api_key()`
5. Implement the model instantiation logic

### Customizing Evaluation Criteria

Modify `util/system_prompt.py` to adjust:
- Evaluation criteria
- Output format
- Assessment categories
- Readiness levels

##  Dependencies

### Core Dependencies
- `langchain-mistralai`: MistralAI integration
- `langchain-openai`: OpenAI integration (future use)
- `langchain-ollama`: Local model support
- `langchain-google-genai`: Google GenAI integration
- `langchain-core`: Core LangChain functionality
- `langchain-community`: Community extensions
- `python-dotenv`: Environment variable management
- `genai`: Google Generative AI library

*Last updated: July 22, 2025*
