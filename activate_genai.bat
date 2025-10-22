@echo off
echo Activating GenAI environment...
call genai_env\Scripts\activate.bat
echo GenAI environment activated!
echo Available packages: genai, ipython, openai (v0.27.10), tiktoken (v0.3.3)
cmd /k
